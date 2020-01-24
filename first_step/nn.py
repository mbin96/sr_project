#Feed-forward network
#인풋을 받아 여러 계층에 파례로 전달 한 후 최졷적인 출력을 제공.



    # 학습 가능한 매개변수(또는 가중치(weight))를 갖는 신경망을 정의합니다.

    # 데이터셋(dataset) 입력을 반복합니다.

    # 입력을 신경망에서 전파(process)합니다.

    # 손실(loss; 출력이 정답으로부터 얼마나 떨어져있는지)을 계산합니다.

    # 변화도(gradient)를 신경망의 매개변수들에 역으로 전파합니다.

    # 신경망의 가중치를 갱신합니다. 일반적으로 다음과 같은 간단한 규칙을 사용합니다: 가중치(wiehgt) = 가중치(weight) - 학습율(learning rate) * 변화도(gradient)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__ (self):
        super(Net,self).__init__()
        # 1 input image channel - grayscale or BW, 6 output channels, 3x3 square convolution
        #kernel
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        # an affine operation : y = Wx + b
        self.fc1 = nn.Linear(16*6*6, 120) # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
device = torch.device("cuda") 
params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1,1,32,32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

#손실율
loss = criterion(output, target)
print(loss)

#loss 로 미분
#역전파 따라가기
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

###역전파###
#변화도는 실행하면서 누적되므로 수동으로 변화도 버퍼를 초기화
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

###정의된 규칙 사용하기
import torch.optim as optim

# Optimizer를 생성합니다.
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 학습 과정(training loop)에서는 다음과 같습니다:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update




