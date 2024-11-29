import torch 
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision

def train(args,model, train_loader, val_loader, criterion, optimizer):
    os.makedirs('./runs/'+args.model, exist_ok=True)
    writer = SummaryWriter('./runs/'+args.model)
    model.train()
    best_acc =0.0
    patience, counter = 10, 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        for X, y in tqdm(train_loader):
            X, y = X.to(args.device), y.to(args.device)
            optimizer.zero_grad()

            output = model(X)
            loss = criterion(output, y)
            
            loss.backward()
            optimizer.step()
            
            # calculate accuracy
            running_loss += loss.item() * X.size(0) # total loss for the batch
            # running_loss += loss.item() 
            # correct += (output.argmax(1) == y).type(torch.float).sum().item()
            correct += torch.eq(output.argmax(1), y).sum()            

            
        
            
       # train_loss = running_loss / len(train_loader)
        train_loss = running_loss / len(train_loader.dataset)  # Divide by total number of samples
        train_acc = correct / len(train_loader.dataset)
        val_loss, val_acc = validate(args, model, val_loader, criterion)
        writer.add_scalars(args.model+' Loss', {'train': train_loss, 'validation': val_loss}, epoch)
        writer.add_scalars(args.model+' Accuracy', {'train': train_acc, 'validation': val_acc}, epoch)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join('./save_dict',args.model + '.pth')
            torch.save(model.state_dict(), save_path)
            print("saving model")
            counter = 0
        else:
            counter += 1
            
        if counter > patience:
            print("Early stopping")
            break
    
    print(f"Best validation accuracy: {best_acc:.4f}")

def validate(args, model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(args.device), y.to(args.device)
            output = model(X)
            loss = criterion(output, y)
            
            running_loss += loss.item() * X.size(0)
            # correct += (output.argmax(1) == y).type(torch.float).sum().item()
            correct += torch.eq(output.argmax(1), y).sum()
            
    # torch.eq(output.argmax(1), y).sum()
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    return val_loss, val_acc

def test(args,model,test_loader):
    model.load_state_dict(torch.load(os.path.join('./save_dict', args.model + '.pth'), weights_only=True))
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in tqdm(test_loader):
            X, y = X.to(args.device), y.to(args.device)
            output = model(X)
            correct += torch.eq(output.argmax(1), y).sum()
    test_acc = correct / len(test_loader.dataset)
    print(f"Test accuracy: {test_acc:.4f}")
    




# Best validation accuracy: 0.8809