import torch
from torch import nn
import torch.nn.functional as F

class RCAB(nn.Module):
    def __init__(self, inChannel, growthRate, layerDepth, reductRateCa):
        super(RCAB, self).__init__()
        
        self.layerDepth = layerDepth
        outChannel = inChannel
        self.conv = nn.Conv2d(inChannel, growthRate, kernel_size=3, padding = 1, bias = True)
        # self.fu = nn.Conv2d(growthRate, inChannel, kernel_size=1, padding = 1, bias = True)
        # layers = []
        # for i in range(layerDepth):
        #     # 3x3 conv
        #     layers.append(nn.Conv2d(inChannel, growthRate, kernel_size=3, padding = dilation, dilation = dilation, bias = True))
        #     inChannel += growthRate
        # self.rdbLayers = nn.ModuleList(layers)
     
        #local feature fusion
        # self.lff = nn.Conv2d(inChannel, outChannel, kernel_size = 1)
        
        #Channel attention
        # self.downScaleCa = nn.Conv2d(outChannel, int(outChannel/reductRateCa), kernel_size = 1)        
        # self.upScaleCa  = nn.Conv2d(int(outChannel/reductRateCa), outChannel, kernel_size = 1) 
        self.downScaleCa = nn.Conv2d(growthRate, int(growthRate/reductRateCa), kernel_size = 1)        
        self.upScaleCa  = nn.Conv2d(int(growthRate/reductRateCa), growthRate, kernel_size = 1) 
        
    def forward(self, x):
        
        x = self.conv(x)
        x = F.relu(x)
            
        #local feature fusion
        caParam = self.downScaleCa(F.avg_pool2d(x,x.size()[3]))
        caParam = F.relu(self.upScaleCa(torch.sigmoid(caParam)))
        x = x * caParam
        
        
        return x

class RDB(nn.Module):
    def __init__(self, inChannel, growthRate, layerDepth, dilation, reductRateCa):
        super(RDB, self).__init__()
        self.growthRate = growthRate
        self.layerDepth = layerDepth
        outChannel = inChannel
        
        layers = []
        for i in range(layerDepth):
            # 3x3 conv
            layers.append(RCAB(inChannel, growthRate, layerDepth, reductRateCa))
            inChannel += growthRate
        self.rdbLayers = nn.ModuleList(layers)
     
        #local feature fusion
        self.lff = nn.Conv2d(inChannel, outChannel, kernel_size = 1)

        #Channel attention
        # self.downScaleCa = nn.Conv2d(outChannel, int(outChannel/reductRateCa), kernel_size = 1)        
        # self.upScaleCa  = nn.Conv2d(int(outChannel/reductRateCa), outChannel, kernel_size = 1) 
        self.downScaleCa = nn.Conv2d(inChannel, int(inChannel/reductRateCa), kernel_size = 1)        
        self.upScaleCa  = nn.Conv2d(int(inChannel/reductRateCa), inChannel, kernel_size = 1) 
        
    def forward(self, x):
        localRes = x
        
        for i in range(self.layerDepth):
            out = self.rdbLayers[i](x)
            
            x = torch.cat((x,out),dim=1)
        
        #local feature fusion
        caParam = self.downScaleCa(F.avg_pool2d(x,x.size()[3]))
        caParam = F.relu(self.upScaleCa(torch.sigmoid(caParam)))
        x = x * caParam
        x = self.lff(x)

        #local residual learning
        x = localRes + x
        return x

# class RDBD(nn.Module):
#     def __init__(self, inChannel, growthRate, layerDepth):
#         super(RDBD, self).__init__()
        
#         self.layerDepth = layerDepth
#         outChannel = inChannel
        
#         layers = []
#         for i in range(layerDepth):
#             # 3x3 conv
#             layers.append(nn.Conv2d(inChannel, growthRate, kernel_size=3, padding = 2, dilation = 2, bias = True))
#             inChannel += growthRate
#         self.rdbLayers = nn.ModuleList(layers)
     
#         #local feature fusion
#         self.lff=nn.Conv2d(inChannel, outChannel, kernel_size = 1)
        
#     def forward(self, x):
#         localRes = x
        
#         for i in range(self.layerDepth):
#             out = self.rdbLayers[i](x)
#             out = F.relu(out)
#             x = torch.cat((x,out),dim=1)
        
#         #local feature fusion
#         x = self.lff(x)
        
#         #local residual learning
#         x = localRes + x
#         return x
    
class RDN(nn.Module):

    def __init__(self, colorChannel, scale, blockDepth, convDepth ):
        super(RDN, self).__init__()
        #parameter
        self.growthRate = 64
        self.g0 = 64
        self.blockDepth = blockDepth
        self.convDepth = convDepth
        reductRateCa = 16
        #SFE
        self.sfe1 = nn.Conv2d(colorChannel, self.g0 , kernel_size=3, padding = 1, bias = True)
        self.sfe2 = nn.Conv2d(self.g0, self.g0 , kernel_size=3, padding = 1, bias = True)

       
        self.sfe1f = nn.Conv2d(self.g0, self.g0 , kernel_size=1, padding = 0, bias = True)

        #RDBs
        layers = []
        for i in range(self.blockDepth):
            layers.append(RDB(self.g0, self.growthRate, self.convDepth, 1, reductRateCa))
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