"""
Author: Salva Rühling Cachay
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def get_optimizer(name, model, **kwargs):
    name = name.lower().strip()
    parameters = get_trainable_params(model)
    if name == 'adam':
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-4
        wd = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
        print('Using Adam optimizer: Lr=', lr, 'Wd=', wd)
        return optim.Adam(parameters, lr=lr, weight_decay=wd)
    if name == 'adamw':
        print('Using AdamW optimizer')
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-4
        wd = kwargs['weight_decay'] if 'weight_decay' in kwargs else 1e-6
        return optim.AdamW(parameters, lr=lr, weight_decay=wd)
    elif name == 'sgd':
        print('Using SGD optimizer')
        lr = kwargs['lr'] if 'lr' in kwargs else 0.01
        momentum = 0.9
        wd = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
        nesterov = kwargs['nesterov'] if 'nesterov' in kwargs else True
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=wd, nesterov=nesterov)
    elif name == 'rmsprop':
        lr = kwargs['lr'] if 'lr' in kwargs else 0.005
        return optim.RMSprop(parameters, lr=lr, momentum=0.0, eps=1e-10)
    else:
        raise ValueError("Unknown optimizer", name)


from abc import ABC


class Scheduler(ABC):
    def __init__(self, model, optimizer, total_epochs, *args, **kwargs):
        self.model = model
        self.n_epochs = total_epochs
        self.optimizer = optimizer

    def step(self, model, epoch, metric):
        pass

    def end(self, dataloader, device='cuda'):
        return self.model

    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']


class ReduceLROnPlateau(Scheduler):
    def __init__(self, model, optimizer, total_epochs, *args, **kwargs):
        super().__init__(model, optimizer, total_epochs, *args, **kwargs)
        patience = 10 if total_epochs <= 100 else 20 if total_epochs <= 250 else 50
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=patience,
                                                              verbose=True)
        self.start = int(total_epochs / 20)

    def step(self, model, epoch, metric):
        if epoch > self.start:
            self.scheduler.step(metric)

    def get_last_lr(self):
        return self.scheduler.optimizer.param_groups[0]['lr']


class Cosine(Scheduler):
    def __init__(self, model, optimizer, total_epochs, *args, **kwargs):
        super().__init__(model, optimizer, total_epochs, *args, **kwargs)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    def step(self, model, epoch, metric):
        self.scheduler.step()


class ExponentialLR(Scheduler):
    def __init__(self, model, optimizer, total_epochs, *args, **kwargs):
        super().__init__(model, optimizer, total_epochs, *args, **kwargs)
        self.gamma = 0.98 if 'gamma' not in kwargs else kwargs['gamma']
        self.min_lr = 1e-6 if 'min_lr' not in kwargs else kwargs['min_lr']
        self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.gamma, last_epoch=-1)

    def step(self, model, epoch, metric):
        if self.get_last_lr() < self.min_lr:
            return
        self.scheduler.step()


class StepLR(Scheduler):
    def __init__(self, model, optimizer, total_epochs, *args, **kwargs):
        super().__init__(model, optimizer, total_epochs, *args, **kwargs)
        step_size = kwargs['step_size'] if 'step_size' in kwargs else 30
        self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

    def step(self, model, epoch, metric):
        self.scheduler.step()


class SWA(Scheduler):
    def __init__(self, model, optimizer, total_epochs, *args, **kwargs):
        super().__init__(model, optimizer, total_epochs, *args, **kwargs)
        self.swa_model = optim.swa_utils.AveragedModel(model)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        self.swa_start = int(0.75 * total_epochs)
        self.swa_scheduler = optim.swa_utils.SWALR(optimizer, anneal_strategy="linear", anneal_epochs=20,
                                                   swa_lr=3e-5)  # swa_lr=0.05)

    def step(self, model, epoch, metric):
        if epoch >= self.swa_start:
            if epoch == self.swa_start:
                print('Switching to SWA scheduler now.')
            self.swa_model.update_parameters(model)
            self.swa_scheduler.step()
        else:
            self.scheduler.step()

    def end(self, dataloader, device='cuda'):
        # optim.swa_utils.update_bn(dataloader, self.swa_model, device=device)
        # Set the weights from SWA to our model (this is necessary since, AverageModel only inherits the parameters
        for param_model, param_SWA in zip(self.model.parameters(), self.swa_model.parameters()):
            param_model.data = param_SWA.data
        return self.model  # self.swa_model


class Weighted_loss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha  # Alpha is for rsdc and beta for rsuc
        self.beta = beta
        self.loss1 = nn.MSELoss(reduction='mean')
        self.loss2 = nn.MSELoss(reduction='mean')

    def forward(self, y_true, y_pred):
        return self.alpha * self.loss1(y_pred[:, :50], y_true[:, :50]) \
               + self.beta * self.loss2(y_pred[:, 50:], y_true[:, 50:])


class Constrained_loss(nn.Module):
    def __init__(self, uw_weight, dw_weight):
        super().__init__()
        self.uw_weight = uw_weight
        self.dw_weight = dw_weight
        self.loss1 = nn.MSELoss(reduction='mean')
        self.loss_flux = nn.MSELoss(reduction='mean')
        self.p1d = (1, 0)

    def flux_net(self, y_true, y_pred):
        y_pred_mod = torch.nn.functional.pad(y_pred, self.p1d, "constant",
                                             0)  # Pad to get the difference while subtracting
        y_pred_mod.requires_grad = True

        levels_diff = y_pred_mod[:, :50] - y_pred  # TOA - 1000 = Surface- 200, 49->940, 48 -> 920

        toa_sum = y_true[:, 0]  # Use y_true or y_pred?
        levels_sum = torch.sum(levels_diff[:, 1:], 1)  # Sum of fluxes lost in each layer
        surface_sum = y_pred[:, -1]

        assert (toa_sum.shape == levels_sum.shape == surface_sum.shape)
        ''' 
            Code to check if the physical law holds true
        y_true_mod = torch.nn.functional.pad(y_true, self.p1d, "constant", 0) 
        levels_diff_true = y_true_mod[:, :50] - y_true
        levels_true_sum = torch.sum(levels_diff_true[:, 1:], 1) 
        surface_true = y_true[:, -1]
        print(toa_sum - (levels_true_sum+surface_true))'''

        return toa_sum, levels_sum, surface_sum
        # Problem with batch-wise -> toa-sum = 10000 W/m2 surface+levels = 8000 W/m2 (20 = 0 , 20=10000)
        # Equality -> TOA = Surface + levels

    def forward(self, y_true, y_pred):
        td, ld, sd = self.flux_net(y_true[:, :50], y_pred[:, :50])
        tu, lu, su = self.flux_net(y_true[:, 50:], y_pred[:, 50:])

        downwelling = self.loss_flux(td, (ld + sd))
        upwelling = self.loss_flux(tu, (lu + su))

        # print(downwelling, upwelling)
        return self.loss1(y_pred, y_true) + self.dw_weight * downwelling + self.uw_weight * upwelling


def get_scheduler(name, model, optimizer, total_epochs, *args, **kwargs):
    name = name.lower().strip()
    if name is None or name in ['no', 'none']:
        return Scheduler(model, optimizer, total_epochs)
    elif name in ['lron', 'reducelron', 'lronplateau', 'reducelronplateau']:
        return ReduceLROnPlateau(model, optimizer, total_epochs, *args, **kwargs)
    elif name in ['swa']:
        return SWA(model, optimizer, total_epochs)
    elif name in ['cosine', 'cosineannealing']:
        return Cosine(model, optimizer, total_epochs, *args, **kwargs)
    elif name in ['steplr', 'step-lr']:
        return StepLR(model, optimizer, total_epochs, *args, **kwargs)
    elif name in ['exp', 'exponential', 'expdecay', 'explr']:
        return ExponentialLR(model, optimizer, total_epochs, *args, **kwargs)
    else:
        raise ValueError('Unknown scheduler!', name, ' (#epochs=', total_epochs, ')')


def get_loss(name, reduction='mean'):
    # Specify loss function
    name = name.lower().strip()
    if name in ['l1', 'mae']:
        loss = nn.L1Loss(reduction=reduction)
    elif name in ['l2', 'mse']:
        loss = nn.MSELoss(reduction=reduction)
    elif name in ['smoothl1', 'smooth']:
        loss = nn.SmoothL1Loss(reduction=reduction)
    elif name == 'weighted':
        loss = Weighted_loss(1, 1)
    elif name == 'constrained':
        loss = Constrained_loss(0.1, 0.1)
    else:
        raise ValueError('Unknown loss function')
    return loss


def get_trainable_params(model):
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    return trainable_params
