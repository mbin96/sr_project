import torch
from torch import nn
import torch.nn.functional as F

class RDB(nn.Module):
    def __init__(self, inChannel, growthRate, layerDepth):
        super(RDB, self).__init__()
        
        self.layerDepth = layerDepth
        outChannel = inChannel
        
        layers = []
        for i in range(layerDepth):
            # 3x3 conv
            layers.append(nn.Conv2d(inChannel, growthRate, kernel_size=3, padding = 1, bias = True))
            inChannel += growthRate
        self.rdbLayers = nn.ModuleList(layers)
     
        #local feature fusion
        self.lff=nn.Conv2d(inChannel, outChannel, kernel_size = 1)
        
    def forward(self, x):
        localRes = x
        
        for i in range(self.layerDepth):
            out = self.rdbLayers[i](x)
            out = F.relu(out)
            x = torch.cat((x,out),dim=1)
        
        #local feature fusion
        x = self.lff(x)
        
        #local residual learning
        x = localRes + x
        return x
    
class RDN(nn.Module):

    def __init__(self, colorChannel, scale):
        super(RDN, self).__init__()
        #parameter
        self.growthRate = 64
        self.g0 = 64
        self.blockDepth = 16
        self.convDepth = 8
        
        #SFE
        self.sfe1 = nn.Conv2d(colorChannel, self.g0 , kernel_size=3, padding = 1, bias = True)
        self.sfe2 = nn.Conv2d(self.g0, self.g0 , kernel_size=3, padding = 1, bias = True)

        #RDBs
        layers = []
        for i in range(self.blockDepth):
            layers.append(RDB(self.g0, self.growthRate, self.convDepth))
        self.rdbs = nn.ModuleList(layers)
        
        #DFF
        self.gff = nn.Sequential(nn.Conv2d(self.g0 * self.blockDepth, self.g0, kernel_size=1, bias = True),
            nn.Conv2d(self.g0, self.g0 , kernel_size=3, padding = 1, bias = True))
        
        #upnet
        if (scale == 4) :
            self.pixelShuffle = nn.Sequential(
                nn.Conv2d(self.g0, self.g0*(2**2), kernel_size=3, padding = 1, bias = True),
                nn.PixelShuffle(2),
                nn.Conv2d(self.g0, self.g0*(2**2), kernel_size=3, padding = 1, bias = True),
                nn.PixelShuffle(2))
        else :
            self.pixelShuffle = nn.Sequential(
                nn.Conv2d(self.g0, self.g0*(scale**2), kernel_size=3, padding = 1, bias = True),
                nn.PixelShuffle(scale))

        #HR
        self.convHR = nn.Conv2d(self.g0, colorChannel, kernel_size=3, padding = 1, bias = True)

    def forward(self, x):
        #SFENet
        x = self.sfe1(x)
        out = self.sfe2(x)
        #RDBs
        rdbResult=[]
        for i in range(self.blockDepth) :
            out = self.rdbs[i](out)
            rdbResult.append(out)

        #concat
        rdbResult = tuple(rdbResult)
        out = torch.cat(rdbResult,dim=1)

        #DFF
        out = self.gff(out)
        #Global residual
        x = x + out
        
        #UPNet
        x = self.pixelShuffle(x)

        #HR
        x = self.convHR(x)
        
        return x