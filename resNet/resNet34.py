import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import glob
import PIL
from PIL import Image
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
import random
import torchsummary

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
trainset = torchvision.datasets.ImageNet(root = './data', split='train', download=True)

class resNet(nn.Module):
    def __init__(self):
        super(resNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 7, stride = 2, padding = 3)
        self.bn0   = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 64, 3, padding = 1)
        self.bn1   = nn.BatchNorm2d(64)
        
        self.reDem12 = nn.Conv2d(64, 128, 1, stride = 2, padding = 1)
        self.conv2_t = nn.Conv2d(64, 128, 3, stride = 2, padding = 1)
        self.conv2_m = nn.Conv2d(128, 128, 3, padding = 1)
        self.bn2     = nn.BatchNorm2d(128)
        
        self.reDem23 = nn.Conv2d(128, 256, 1, stride = 2, padding = 1)
        self.conv3_t = nn.Conv2d(128, 256, 3, stride = 2, padding = 1)
        self.conv3_m = nn.Conv2d(256, 256, 3, padding = 1)
        self.bn3     = nn.BatchNorm2d(256)
        
        self.reDem34 = nn.Conv2d(256, 512, 1, stride = 2, padding = 1)
        self.conv4_t = nn.Conv2d(256, 512, 3, stride = 2, padding = 1)
        self.conv4_m = nn.Conv2d(512, 512, 3, padding = 1)
        self.bn4     = nn.BatchNorm2d(512)
        
        self.maxPool  = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.avg_pool = nn.AvgPool2d(7, stride = 1)

        self.fc = nn.Linear(512*7*7, 1000)

    def forward(self, x):
        #conv0 224 -> 56
        x = self.maxPool(F.relu(self.bn0(self.conv0(x))))

        #conv1 
        for i in range(3):
            i = i
            xpre = x
            x = self.bn1(self.conv1(F.relu(self.bn1(self.conv1(x)))))
            # residual addiction
            x = x + xpre
            # relu result
            x = F.relu(x)

        #conv2
        xpre = x
        x = self.bn2(self.conv2_t(F.relu(self.bn2(self.conv2(x)))))
        # residual addiction
        x = x + self.reDem12(xpre)
        # relu result
        x = F.relu(x)
        for i in range(3):
            xpre = x
            x = self.bn2(self.conv2_m(F.relu(self.bn2(self.conv2(x)))))
            # residual addiction
            x = x + xpre
            # relu result
            x = F.relu(x)

        #conv3
        xpre = x
        x = self.bn3(self.conv3_t(F.relu(self.bn3(self.conv3(x)))))
        # residual addiction
        x = x + self.reDem23(xpre)
        # relu result
        x = F.relu(x)
        for i in range(5):
            xpre = x
            x = self.bn3(self.conv3_m(F.relu(self.bn3(self.conv3(x)))))
            # residual addiction
            x = x + xpre
            # relu result
            x = F.relu(x)

        #conv4
        xpre = x
        x = self.bn4(self.conv4_t(F.relu(self.bn4(self.conv4(x)))))
        # residual addiction
        x = x + self.reDem34(xpre)
        # relu result
        x = F.relu(x)
        for i in range(2):
            xpre = x     
            x = self.bn4(self.conv4_m(F.relu(self.bn4(self.conv4(x)))))
            # residual addiction
            x = x + xpre
            # relu result
            x = F.relu(x)

        #fully connected
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


net = resNet()

#pre trained model load
net.load_state_dict(torch.load('./data/resnet34.pth'))
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, 
                    momentum=0.9, weight_decay=1e-4)

decay_epoch = [32000, 48000]
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, 
                 milestones=decay_epoch, gamma=0.1)
net.train()



net.eval()



