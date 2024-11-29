import torch, sys,os
from torch import nn
from torchsummary import summary
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import copy
from collections import OrderedDict
from utils.util import drop_path, init_weights

class ConvBNSwish(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1,add_swish=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, groups=groups),
            nn.BatchNorm2d(out_channels)
        ]
        if add_swish:
            layers.append(nn.SiLU())
        super().__init__(*layers)
        


class SEBlock(nn.Module):
    
    def __init__(self, in_channels, mbconv_in_channels,reduction_ratio=4):
        """
            squeeze-and-excitation block
        Args:
            in_channels (_type_): SEBlock input channel
            mbconv_in_channels (_type_): the num of channels in the beginning of the block
            out_channelsreduction_ratio (int, optional): _description_. Defaults to 4.
        """
        super().__init__()
        """ 
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(in_channels, max(1,mbconv_in_channels//reduction_ratio)) # 输出节点个数等于最开始输入MBConv的输出节点个数/4
        self.fc2 = nn.Linear(max(1,mbconv_in_channels//reduction_ratio), in_channels)
        self.sigmoid = nn.Sigmoid()
        self.swish = nn.SiLU()
        """
        #* use 1*1 conv as a linear layer
        squeeze_c = max(1,in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Conv2d(in_channels,squeeze_c,1)
        self.swish = nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_c,in_channels,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.avg_pool(x)  #(b,c,1,1)
        # fc1
        out = self.fc1(out) 
        out = self.swish(out)
        # fc2
        out = self.fc2(out) 
        out = self.sigmoid(out)

        return out * x

def create_mbconv(in_channels, out_channels, kernel_size=1, 
                  stride=1, groups=1,add_swish=True, skip_conv=False):
    # create mbconv layer or identity layer
    # if skip_conv is True, skip all of the first 1*1 conv in each layer 
    if skip_conv:
        return nn.Identity()
    return ConvBNSwish(in_channels, out_channels, kernel_size, stride, groups, add_swish)
  
class DropPath(nn.Module):
    # drop path is a kind of dropout, but it is only applied during training.
    # constructed as a nn layer.
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
def adjust_channels(exp_ratio, channel):
    return round(exp_ratio * channel)

class MBConvConfig:
    """_summary_
        config of MBConv layer
    """
    def __init__(self,
                 kernel_size,
                 in_channels,
                 out_channels,
                 exp_ratio,
                 stride,
                 use_SE,
                 drop_connect_rate,
                 index,  # layer name
                 width_coefficient
                 ):
        # adjust channels
        self.in_channels = adjust_channels(in_channels, width_coefficient)
        self.kernel_size = kernel_size
        self.expanded_c = self.in_channels * exp_ratio
        self.out_channels = adjust_channels(out_channels, width_coefficient)
        self.stride = stride
        self.use_SE = use_SE
        self.drop_connect_rate = drop_connect_rate
        self.index = index
        self.exp_ratio = exp_ratio
        
class MBConv(nn.Module):
    def __init__(self,
                 cnf:MBConvConfig
                 ):
        """_summary_
            construct MBConv layer
        Args:
            cnf: config of MBConv layer
        """
        super(MBConv, self).__init__()
        # scale up channels in the first 1*1 conv 
       
        skip_conv = True if cnf.exp_ratio==1.0 else False 
        
        # check if have shortcut path
        self.use_shortcut = True if cnf.stride==1 and cnf.in_channels == cnf.out_channels else False
        #* dropout layer
        if self.use_shortcut and cnf.drop_connect_rate > 0:
            self.dropout = DropPath(cnf.drop_connect_rate)
        else:
            self.dropout = nn.Identity()        
        
        self.layers = nn.Sequential(
            # 1* 1
            create_mbconv(cnf.in_channels, cnf.expanded_c, 1, skip_conv=skip_conv),
            # DWConv
            create_mbconv(cnf.expanded_c, cnf.expanded_c, cnf.kernel_size, cnf.stride, groups=cnf.expanded_c),
            SEBlock(cnf.expanded_c, cnf.in_channels),
            # 1*1
            create_mbconv(cnf.expanded_c, cnf.out_channels, 1, add_swish=False),
        )

    def forward(self, x):
        identity = x if self.use_shortcut else None
        out = self.layers(x)
        if identity is not None:
            out += identity
            
        return out
'''       
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size,stride, 
                 channel_multiplier=1.0, dropout_rate=0.2
                 ):
        """_summary_

        Args:
            in_channels (_type_): intput channels. already adjusted by width_coefficient
            out_channels (_type_): output channels of the last 1*1 conv. already adjusted by width_coefficient
            stride (_type_): _description_
            channel_multiplier (float, optional): _description_. Defaults to 1.0.
            dropout_rate (float, optional): _description_. Defaults to 0.2.
        """
        super(MBConv, self).__init__()
        # scale up channels in the first 1*1 conv 
        hidden_channels = adjust_channel(channel_multiplier, in_channels)
        skip_conv = True if channel_multiplier==1.0 else False
        # check if have shortcut path
        self.use_shortcut = True if stride==1 and in_channels==out_channels else False
        #* dropout layer
        if self.use_shortcut and dropout_rate > 0:
            self.dropout = DropPath(dropout_rate)
        else:
            self.dropout = nn.Identity()        
        
        self.layers = nn.Sequential(
            # 1* 1
            create_mbconv(in_channels, hidden_channels, 1, skip_conv=skip_conv),
            # DWConv
            create_mbconv(hidden_channels, hidden_channels, kernel_size, stride=stride),
            SEBlock(hidden_channels, in_channels),
            # 1*1
            create_mbconv(hidden_channels, out_channels, 1, add_swish=False),
            # dropout layer
            self.dropout
        )

    def forward(self, x):
        identity = x if self.use_shortcut else None
        out = self.layers(x)
        if identity is not None:
            out += identity
            
        return out
'''  
  
def adjust_channel(exp_ratio, channel):
    return round(exp_ratio * channel)
      
#! only consider B0 now   
class EfficientNetV1(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 10,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 ):
        super().__init__()
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.stages = []
        # B0 config
        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        # exp_ratio=n
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],  # s2
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],  # s3
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],  # s4
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],  # s5
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],  # s6
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],  # s7
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]  # s8
        
        self.conv1 = ConvBNSwish(3,adjust_channel(self.width_coefficient,32),3, stride=2)
        
        
        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(np.ceil(depth_coefficient * repeats))
        
        # construct config for S2-S8, each line in it is the layer config of each stage
        # if B0, the mbconv_confighs has 16 lines, because there are 16 layers in S2-28
        # # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, layer_index

        mbconv_configs = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)  # default for B0
            for i in range(round_repeats(cnf.pop(-1))):
                # only the first block of each stage has stride=2 for dwconv
                if i>0:
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2] # change input channel if repeat>1
                
                #! the drop_connect_rate is updated for each block
                #TODO update dropout rate
                
                index = str(stage+1) + chr(i+97)  # 1a, 2a, 2b, ...

                mbconv_configs.append([*cnf,index, self.width_coefficient])
                
            
        
         
        self.stages = OrderedDict()
        for cnf in mbconv_configs:
            cnf = MBConvConfig(*cnf)
            self.stages[cnf.index] = self._make_layer(cnf)
            
            
        self.stages = nn.Sequential(self.stages)
        in_channel = mbconv_configs[-1][2]
        out_channel = adjust_channel(self.width_coefficient, 1280)
        
        self.conv2 = ConvBNSwish(in_channel,out_channel,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        if dropout_rate > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(out_channel, num_classes)
            )
        else:
            self.classifier = nn.Linear(out_channel, num_classes)
        init_weights(self)

    def _make_layer(self, cnf):
        # output channels: final output channels of the mbcov block
        return MBConv(cnf)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.stages(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.classifier(torch.flatten(x,1))
        return x

        

if __name__ == "__main__":
    torch.set_printoptions(precision=10, threshold=float('inf'), linewidth=120, sci_mode=False)

    a = torch.randn((1,3,224,224)).to("cuda")
    # model = MBConv(12,36,2, channel_multiplier=2) 
    model = EfficientNetV1(width_coefficient=1, depth_coefficient=1).to("cuda")
    res = model(a)
    #print(model)
    summary(model, input_size=(3, 224, 224),device="cuda")  # Add this line for detailed summary
    #print(res.shape)
    print(f"width_coefficient: {model.width_coefficient}, depth_coefficient: {model.depth_coefficient}")
    #print(model)


