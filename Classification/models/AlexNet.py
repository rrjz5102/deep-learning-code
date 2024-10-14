import torch
from torch import nn
from torchsummary import summary

# N = (W - kernel +2padding)/stride + 1
# input image: (3,224,224)
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  
            # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), 
            # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(),
            
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x) #(b,128,6,6)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1,3,224,224).to(device)
    model = AlexNet().to(device)
    model.eval()
    out = model(x)
