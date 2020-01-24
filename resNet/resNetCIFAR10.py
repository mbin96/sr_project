import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=700,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np



class resNet(nn.Module):
    def __init__(self):
        super(resNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, 3, padding = 1)
        self.bn0   = nn.BatchNorm2d(16)

        self.conv1 = nn.Conv2d(16, 16, 3, padding = 1)
        self.bn1   = nn.BatchNorm2d(16)
        
        self.reDem12 = nn.Conv2d(16, 32, 1, stride = 2)
        self.conv2_t = nn.Conv2d(16, 32, 3, stride = 2, padding = 1)
        self.conv2_m = nn.Conv2d(32, 32, 3, padding = 1)
        self.bn2     = nn.BatchNorm2d(32)
        
        self.reDem23 = nn.Conv2d(32, 64, 1, stride = 2)
        self.conv3_t = nn.Conv2d(32, 64, 3, stride = 2, padding = 1)
        self.conv3_m = nn.Conv2d(64, 64, 3, padding = 1)
        self.bn3     = nn.BatchNorm2d(64)
        
        #self.maxPool  = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.avg_pool = nn.AvgPool2d(8, stride = 1)

        self.fc = nn.Linear(8*8, 10)

    def forward(self, x):
        #conv0 224 -> 56
        x = F.relu(self.bn0(self.conv0(x)))

        #conv1 
        for i in range(9):
            i = i
            xpre = x
            x = self.bn1(self.conv1(F.relu(self.bn1(self.conv1(x)))))
            # residual addiction
            x = x + xpre
            # relu result
            x = F.relu(x)

        #conv2
        xpre = x
        x = self.bn2(self.conv2_m(F.relu(self.bn2(self.conv2_t(x)))))
        # residual addiction
        x = x + self.bn2(self.reDem12(xpre))
        # relu result
        x = F.relu(x)
        for i in range(8):
            xpre = x
            x = self.bn2(self.conv2_m(F.relu(self.bn2(self.conv2_m(x)))))
            # residual addiction
            x = x + xpre
            # relu result
            x = F.relu(x)

        #conv3
        xpre = x
        x = self.bn3(self.conv3_m(F.relu(self.bn3(self.conv3_t(x)))))
        # residual addiction
        x = x + self.bn3(self.reDem23(xpre))
        # relu result
        x = F.relu(x)
        for i in range(17):
            xpre = x
            x = self.bn3(self.conv3_m(F.relu(self.bn3(self.conv3_m(x)))))
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
print(net)

device = torch.device("cuda:0")
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, 
                    momentum=0.9, weight_decay=1e-4)

decay_epoch = [32000, 48000]
#step_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, gamma=0.1)
net.train()
print('hing')
for epoch in range(100):
    running_loss = 0.0
    print(epoch)
    for i, data in enumerate(trainloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data[0].to(device), data[1].to(device)

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i > 2000:    # print every 2000 mini-batches
            i = i % 2000
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    #step_lr_scheduler.step()

net.eval()




