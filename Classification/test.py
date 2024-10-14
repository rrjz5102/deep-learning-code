import torch
from torch import nn
a = torch.ones((3,224,224))

conv = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1)

res = conv(a)
print(res.shape)
