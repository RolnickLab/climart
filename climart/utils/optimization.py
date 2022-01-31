import torch
import torch.nn as nn

def get_loss(name, reduction='mean'):
    name = name.lower().strip().replace('-', '_')
    if name in ['l1', 'mae', "mean_absolute_error"]:
        loss = nn.L1Loss(reduction=reduction)
    elif name in ['l2', 'mse', "mean_squared_error"]:
        loss = nn.MSELoss(reduction=reduction)
    elif name in ['smoothl1', 'smooth']:
        loss = nn.SmoothL1Loss(reduction=reduction)
    else:
        raise ValueError(f'Unknown loss function {name}')
    return loss


def get_trainable_params(model):
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return trainable_params
