from __future__ import print_function
import torch

x = torch.rand(5, 3)
y = torch.rand(5, 3)
#tensor add
print(x+y)
#create empty tensor 
result = torch.empty(5, 3)
#add tensor at result
torch.add(x, y, out = result)

print(result)
#add tensor and replace
y.add_(x)

#resize and reshape tensor
x = torch.randn(4,4)
y = x.view(16)
#use -1 다른차원을 유추함.
z = x.view(-1,8) 
print(x.size(),y.size(),z.size())
print(x,y,z)

# 이 코드는 CUDA가 사용 가능한 환경에서만 실행합니다.
# ``torch.device`` 를 사용하여 tensor를 GPU 안팎으로 이동해보겠습니다.
if torch.cuda.is_available():
    device = torch.device("cuda")          # CUDA 장치 객체(device object)로
    y = torch.ones_like(x, device=device)  # GPU 상에 직접적으로 tensor를 생성하거나
    x = x.to(device)                       # ``.to("cuda")`` 를 사용하면 됩니다.
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 는 dtype도 함께 변경합니다!
