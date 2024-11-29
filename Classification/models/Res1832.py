import torch

from torch import nn
from torchsummary import summary
# 查看模型结构
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}"

class BasicBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample != None:
            identity = self.downsample(identity)  # 不要对x进行下采样
        x += identity
        x = self.relu(x)
        
        return x
        
        
        
class ResNet(nn.Module):
    def __init__(self, block, block_nums, include_top=True, num_classes=1000):
        """_summary_

        Args:
            block (_type_): _description_
            block_nums (_type_):  list
        """
        super().__init__()
        self.in_channel = 64
        self.num_classes = num_classes
        self.include_top = include_top
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
            )

        
        # conv2_x
        self.conv2 = self._make_layers(block, block_nums[0], 64, stride=1)
        # 3x
        self.conv3 = self._make_layers(block, block_nums[1], 128, stride=2)
        #4x
        self.conv4 = self._make_layers(block, block_nums[2], 256, 2)
        #5x
        self.conv5 = self._make_layers(block, block_nums[3], 512, 2)
        
        # global average pooling and num_classes-way fc
        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            
            self.fc = nn.Linear(512, num_classes)
        # init 
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        
    def _make_layers(self, block, block_num, channel, stride):
        downsample = None
        # downsample layer
        if stride!=1:
            # 对于res18 or res32,basic block不改变通道
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel, kernel_size=1, stride=2,bias=False),
                nn.BatchNorm2d(channel)
            )
        
        layers = []
        # construct the first basic block
        layers.append(block(self.in_channel,channel,stride,downsample))
        self.in_channel = channel # 需要调整basci block的输入通道
        # construct the reset basci block
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,channel, 1))  # 只有第一层basic block需要下采样
           
            
        return nn.Sequential(*layers)

                
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)  # (n, 512, 7,7)
        
        if self.include_top:
            x = self.avgpool(x)  # (n,512,1,1)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x
    
    
if __name__ == '__main__':
    net = ResNet(BasicBlock, [3,4,6,3])

    x = torch.randn(1,3,224,224)
    y = net(x)
    print(y.shape)

    # print(net)
