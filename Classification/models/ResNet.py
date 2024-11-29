import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1  # resnet50, resnet101, resnet152的conv2x通道会变
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        x = self.relu(x)
        
        return x
        
        
        
class Bottleneck(nn.Module):
    """_summary_
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    """
    expansion =4 
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        """_summary_
        3*3在resnext中要进行分组卷积，且通道数是resnet的两倍
        Args:
            in_channel (_type_): _description_
            out_channel (_type_): the number of filters(output channels) in 3*3 conv layer 
            stride (int, optional): _description_. Defaults to 1.
            downsample (_type_, optional): _description_. Defaults to None.
            groups (int, optional): resnext50 groups=32, width_per_group=4，通道数提升两倍
            width_per_group (int, optional): _description_. Defaults to 64.
        """
        super().__init__()
        
        # 分组卷积减少参数，假设in_channel = 3, out_channel = 64
        # group=1, width = out_channel, 用于resnet. group = 32
        width = int(out_channel * (width_per_group/64)) * groups #number of channels allocated per group in a grouped convolution. 
        
        #TODO：分组卷积
        self.expansion = 4
        self.downsample = downsample

        # conv2x不改变feature map的尺寸，conv3x-conv5x会改变，stride会变，不能设为1
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width, 
                               kernel_size=1,stride=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(width)
        
        
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width,groups=groups,
                               kernel_size=3,stride=stride,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        
        # change channel
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1,stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        # downsample中 channel要变为expansion倍
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        x += identity
        x = self.relu(x)
        
        return x
        
        
class ResNet(nn.Module):
    def __init__(self, block, block_nums, num_classes=1000, include_top=True, group=1,width_per_group=64):
        """_summary_

        Args:
            block (_type_): BasciBlock OR Bottleneck
            block_num (_type_): list[int]
            num_classes (int, optional): _description_. Defaults to 1000.
            include_top (bool, optional): _description_. Defaults to True.
            group (int, optional): _description_. Defaults to 1.
            width_per_group (int, optional): _description_. Defaults to 64.
        """
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.block_nums = block_nums
        self.group = group
        self.width_per_group = width_per_group
        
        #conv1_x
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #2x-5x
        self.conv2 = self._make_layer(block, 64, block_nums[0])  # 不改变feature map大小, s=1
        self.conv3 = self._make_layer(block, 128, block_nums[1], stride=2)
        self.conv4 = self._make_layer(block, 256, block_nums[2], stride=2)
        self.conv5 = self._make_layer(block, 512, block_nums[3], stride=2)
       
        # global average pooling and num_classes-way fc
        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512*block.expansion, num_classes)
            
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def _make_layer(self, block, channel, block_num, stride=1):
         
         # downsample layer
         # s!=1, feature map缩小2倍 OR resnet101中的通道改变也需要downsample调整
        downsample =None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        
        layers = []
        
        # 第一层basic block
        layers.append(block(self.in_channel, channel, stride, downsample, groups=self.group, width_per_group=self.width_per_group))
        
        # 其余basic block
        self.in_channel = channel * block.expansion 
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, stride=1,
                                downsample=None, groups=self.group, width_per_group=self.width_per_group
                                )
                          )
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
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
    # def __init__(self, block, block_nums, num_classes=1000, include_top=True, group=1,width_per_group=64):
    x = torch.randn(1,3,256,256)
    # model = ResNet(BasicBlock, [3,2,2,2])
    model = ResNet(Bottleneck, [3,4,6,3],group=32,width_per_group=4)
    y = model(x)
    print(model)
    print(y.shape)
        
        
