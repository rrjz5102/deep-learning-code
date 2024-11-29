import torch
from torch import nn
from torchsummary import summary


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, groups=1, add_relu=True):
        padding = (kernel_size - 1) // 2  # maintain the shape of input feature if stride = 1
        layers = [
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel)
        ]
        if add_relu:
            layers.append(nn.ReLU())
        super().__init__(*layers)


class BottleNeck(nn.Module):
    def __init__(self, in_channel, middle_channel, output_channel, stride=1, groups=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.groups = groups
        #! 在stage的第一个block，需要确保concat后维度符合最终的输出。
        ''' 
        #* g=1时,考虑stage1第一步block,out_channel=144
        #* 输入channel为24，identity channel为24。在主分支上前两个unit的输出通道是out_channel//4=36。
        #* 由于需要concat通道，那么主分支最后的输出通道数为out_channel - identity channel = 144 - 24 = 120。 在最后的1*1需要把通道数升为120
        '''
        if self.downsample:
            output_channel -= in_channel
        
        # s=2 only for the first block
        # stage1不使用gconv, 切每个stage的block1 stride=2，output channel翻四倍
        self.conv1 = ConvBNReLU(in_channel, middle_channel, kernel_size=1, stride=1, groups=groups)
        self.dwconv = ConvBNReLU(middle_channel, middle_channel, kernel_size=3, stride=stride, groups=middle_channel, add_relu=False)  # No ReLU for dwconv
        self.conv2 = ConvBNReLU(middle_channel, output_channel, kernel_size=1, stride=1, groups=groups, add_relu=False)
        self.relu = nn.ReLU()  
        
    def _channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()

        channels_per_group = num_channels // self.groups
    
        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)


        x = torch.transpose(x, 1, 2).contiguous()  #! view can only be applied to a contiguous tensor

        # flatten
        x = x.view(batchsize, -1, height, width)

        return x
    
    
    
    def forward(self, x):

        identity = x
        x = self.conv1(x)
        x = self._channel_shuffle(x)
        x = self.dwconv(x) 
        x = self.conv2(x)

        if self.downsample:
            # downsample twice 
            identity = self.downsample(identity)
            x = torch.concat((x, identity), dim=1)
        else:
            x += identity
        x = self.relu(x)

        return x
    
    
class ShuffNetv1(nn.Module):
    def __init__(self, cfg, block, num_classes=1000):
        """_summary_

        Args:
            cfg (_type_): dict
        """
        super().__init__()
        self.block = block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        
        self.stages = []
        input_channel = 24
        for stage_idx, (out_channel, num_blocks) in enumerate(zip(cfg['out_planes'], cfg['num_blocks'])):
            stage = []
            for block_idx in range(num_blocks):
                stride = 2 if block_idx == 0 else 1
                downsample = None
                if stride != 1:  #! No channel changes in downsample layer.
                    downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                
                
                
                stage.append(
                    block(
                        in_channel=input_channel,
                        middle_channel=out_channel // 4,
                        output_channel=out_channel,
                        stride=stride,
                        groups=1 if block_idx == 0 else cfg['groups'],
                        downsample=downsample
                    )
                )
                input_channel = out_channel
            self.stages.append(nn.Sequential(*stage))
        self.stages = nn.Sequential(*self.stages)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(input_channel, num_classes)  
        )
          
    def _make_layer(self, in_channel, out_channel, num_blocks, groups):
        # check if need downsample
        layers = []
        for i in range(num_blocks):
            stride =2 if i==0 else 1
            downsample = None
            if stride !=1:
                downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                
            # def __init__(self, in_channel, middle_channel, output_channel,stride=1, groups=1, downsample=None):
            layers.append(self.block(in_channel, int(in_channel/4), out_channel,stride, groups, downsample))
            in_channel = out_channel
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.stages(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
        
if __name__ == "__main__":
    # model = BottleNeck(in_channel=200, middle_channel=50, output_channel=400, stride=2, groups=1, downsample=downsample)
    cfg = {'out_planes': [200, 400, 800],
    'num_blocks': [4, 8, 4],
    'groups': 2
    }
    model = ShuffNetv1(cfg, BottleNeck) 
    # print(model)
    summary(model, (3, 224, 224), device="cpu")
    x = torch.randn(3, 3, 224, 224)
    y = model(x)
    print(y.shape)

    