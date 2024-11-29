import sys
import torch
import os
from torch import nn


def create_model(args, model, pretrain_path):
    """ load model except classifier weights
    Args:
        args (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Load pretrained weights
    state_dict = torch.load(os.path.join(pretrain_path, args.load_pth), weights_only=True)
    
    # Remove classifier weights from state dict
    for key in list(state_dict.keys()):
        if "classifier" in key:
            state_dict.pop(key)
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    return model


def drop_entire_path(x, drop_prob: float =0., training=False):
    """
    drop the entire path with probability drop_prob
    Args:
        x (_type_): _description_
        drop_prob (float, optional): _description_. Defaults to 0..
        training (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if drop_prob == 0. or not training:
        return x
    if torch.rand(1).item() <= drop_prob:
        return torch.zeros_like(x)  # Drop the entire path by returning a tensor of zeros
    return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """_summary_
        dropout some value of x with probability drop_prob
    Args:
        x (_type_): output of a layer
        drop_prob (float, optional): _description_. Defaults to 0..
        training (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # x:(b,c,h,w), shape:(b,1,1,1) for broadcasting
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors
    # random_tensor: value of it is in [keep_prob, 1+keep prob]
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize: if value >= 1, set it to 1, else 0
    
    #! why divide by keep_prob??????
    output = x.div(keep_prob) * random_tensor 
    return output

def init_weights(model):
    print("*"*10+" initialzie weights "+ "*"*10)
    # initial weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
    


