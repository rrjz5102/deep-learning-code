from torch import nn
import torch

class Model(nn.Module):
    def __init__(self):
        """
        If you have parameters in your model, which should be saved and restored in the state_dict , 
        but not trained by the optimizer, you should register them as buffers.
        """
        super().__init__()
        self.a = torch.ones(1)
        self.b= nn.Parameter(torch.ones(1))
        self.register_buffer("c", torch.ones(1))
        
    def forward(self):
        print(self.a.device, self.b.device, print(self.c.device))
        
model  = Model().to("cuda")
model()    