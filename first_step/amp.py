from __future__ import print_function
import torch
from apex import amp
N, D_in, D_out = 64, 1024, 512
x = torch.randn(N, D_in, device="cuda")
y = torch.randn(N, D_out, device="cuda")

model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
for to in range(500): 
    y_pred = model(x)
    loss = torch.nn.functinoal.mse_loss(y_pred, y)
    optimizer.zero_grad()
    with amp.scale_loss(loss, optimizer) as scaled_loss: 
        scaled_loss.backward()
    optimizer.step()
