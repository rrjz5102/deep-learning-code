import torch
from torch import nn
from torchsummary import summary

class Vgg16(nn.Module):
    def __init__(self, cfg, num_classes=1000,init_weights=False):
        super().__init__()
        self.num_classes = num_classes
        self.cfg = cfg
        self.features = self.make_features() # # 512,7,7
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def make_features(self):
        layers = []
        in_channels = 3
        for channel in self.cfg:
            if channel == 'M':
                layers += [nn.MaxPool2d(2,2)]
            else:
                # conv layer
                conv = nn.Conv2d(in_channels, channel, 3, 1, 1)
                layers += [conv, nn.ReLU()]
                in_channels = channel
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.features(x) 
        
        return x
        
if __name__ == '__main__':
    cfg = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']

    model = Vgg16(cfg).to("cuda")
    summary(model, (3,224,224))