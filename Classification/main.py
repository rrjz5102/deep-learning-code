from utils.dataloader import Mydataset, TransformDataset
from utils.train import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
import argparse
from torch.utils.data import Subset
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os, sys
from torch import nn, optim
from torchvision import datasets
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from models.AlexNet import AlexNet  # Adjust this if the class/function name is different
from torch.utils.data import random_split
from torchsummary import summary

def main(args):
    # config

    data_transforms = {
        "train": v2.Compose([
            v2.ToImage(),
            v2.Resize(256),
            v2.RandomResizedCrop(224),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val":v2.Compose([
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    ''' 
    # flower data
    train_dataset = Mydataset(root='./data/flower_data', subfolder='train', transform=data_transforms['train'])
    val_dataset = Mydataset(root='./data/flower_data', subfolder='val', transform=data_transforms['val'])
    test_dataset = Mydataset(root='./data/flower_data', subfolder='test', transform=data_transforms['val'])
    indices = list(range(100))
    # train_dataset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    
    '''
    
    # cifar10 data
    train_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=data_transforms['val'])
    # split val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # train_dataset.transform = data_transforms['train'] transform doesnt work
    # val_dataset.transform = data_transforms['val']
    train_dataset = TransformDataset(train_dataset, transform=data_transforms['train'])
    # train_dataset = Subset(train_dataset, list(range(100)))
    val_dataset = TransformDataset(val_dataset, transform=data_transforms['val'])
    # dataloade
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,pin_memory=True)
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers


    print(f"training samples: {len(train_loader.dataset)}")
    print(f"validation samples: {len(val_loader.dataset)}")
    print(f"testing samples: {len(test_loader.dataset)}")
    
    model = AlexNet(num_classes=10).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # train(args,model, train_loader, val_loader, criterion, optimizer)
    # summary(model, (3,224,224))
    test(args,model, test_loader)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Start training")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default='alex')
    args = parser.parse_args()
    main(args)
