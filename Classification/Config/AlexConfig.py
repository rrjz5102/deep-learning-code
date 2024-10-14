import os
import torch
class Config:
    def __init__(self):
        # data config
        self.root = 'data/flower_data'
        self.train_path = os.path.join(self.root,'train')
        self.val_path = os.path.join(self.root,'val')
        self.test_path = os.path.join(self.root,'test')
        assert os.path.exists( self.train_path ), "Dataset not found"
        
        
        
        # model config
        

if __name__ == "__main__":    
    pass    
    