import torch

from torch import nn
from torchsummary import summary
import torchvision
import os


# 查看模型结构
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"
os.environ['TORCH_HOME'] = 'E:\learn\pyproject\Pretrained'

def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

'''
1.构造basci block
2.构造inverted residual block
3.构造整个model
'''
# initialzie sequential class  
class ConvBNReLU(nn.Sequential): 
    # automatic forward pass
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2 # maintain the shape of input feature if stride = 1
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6()
        )
        

class InvertedResidual(nn.Module):

    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        """_summary_
        Args:
            in_channel (_type_): _description_
            out_channel (_type_): _description_
            stride (_type_): The first layer of each sequence has a stride s and all others use stride 1.
            expand_ratio (_type_): set to 6 except for all the bottleneck layers
        """
        super().__init__()
        hidden_channel = in_channel*expand_ratio  # output channel of the intermidate layer in the bottleneck
        # check if has the shortcut path
        if stride ==1 and in_channel == out_channel:
            self.use_shortcut = True
        else:
            self.use_shortcut = False
        
        self.layer = []
        # 1*1. if t==1, no 1*1 conv
        if expand_ratio != 1:
            # maintain shape
            self.layer.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        # 3*3 DW conv, maintain channel
        self.layer.append(
            ConvBNReLU(hidden_channel,hidden_channel, kernel_size=3, stride=stride, groups=hidden_channel)
        )
        # 1*1 with linear activation
        self.layer.extend([
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
            
        ])
        
        self.conv = nn.Sequential(*self.layer)
    
    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super().__init__()
        print("------Initialize MobileNetV2------")
        self.features = []
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # first layer
        #in_channel =32
        #last_channel = 1280
        in_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)
        self.features.append(ConvBNReLU(3,in_channel,kernel_size=3,stride=2))
    
        # bottleneck layers
        for t,c,n,s in inverted_residual_setting:
            # num of repeated blocks
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i==0 else 1 # stride=s:only the first layer in the blocks
                self.features.append(InvertedResidual(in_channel,output_channel,stride,t))
                in_channel = c
                
        
        # conv2d
        self.features.append(ConvBNReLU(in_channel,last_channel,kernel_size=1))
        self.features = nn.Sequential(*self.features)
        # avg pool
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))  # (batch, channel, 1, 1)
        # fc 论文是conv2d?
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
                
if __name__ == "__main__":
    # model = InvertedResidual(3,3,2,6).to("cuda")
    device ="cuda"
    model = MobileNetV2(num_classes=1000).to(device)
    path = "E:\learn\pyproject\Pretrained\hub\checkpoints"
    print(model)
    summary(model, (3, 224, 224), device="cuda") 
    # model.load_state_dict(torch.load(os.path.join(path, "mobilenet_v2-b0353104.pth"),weights_only=True))
    
    #x = torch.randn(1,3,224,224).to("cuda")
    #y = model(x)
    #print(y.shape)
    
 