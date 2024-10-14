
from importlib import import_module
import os
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import matplotlib.pyplot as plt
''' 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
x = import_module('Config.AlexConfig')
print(x)
config = x.Config()
print(config)
'''

class Mydataset(Dataset):
    def __init__(self, root, subfolder,transform=None):
        # root : './data/flower_data'
        # subfolder : 'train' or 'val' or test
        self.root = root
        self.subfolder = subfolder
        self.transform = transform
        self.imgs = []
        self.labels = []
        self.label_dict = {"daisy":0, "dandelion":1, "roses":2, "sunflowers":3, "tulips":4}
        # read all images
        self.read_txtfile()
    def read_txtfile(self):
        with open(os.path.join(self.root,self.subfolder+'.txt'),'r') as f:
            for line in f:
                img, label = line.strip().split('\t')
                self.imgs.append(os.path.join(self.root,self.subfolder,img))
                self.labels.append(self.label_dict[label])
                assert len(self.imgs) == len(self.labels)
    
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[index]
        img = Image.open(img_path).convert('RGB') #[channel, height, width]
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label)
    
    
class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
if __name__ == '__main__':
    x = Mydataset(root='./data/flower_data', subfolder='train', transform=None, train=True)
    a,label = x[0] 
    plt.imshow(a)
    plt.show()
    