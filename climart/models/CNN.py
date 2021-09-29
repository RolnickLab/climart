from typing import Sequence, Optional, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from climart.data_wrangling.constants import LEVELS, LAYERS, GLOBALS
from climart.models.base_model import BaseModel, BaseTrainer
from climart.models.column_handler import ColumnPreprocesser
from climart.utils.utils import get_activation_function, get_normalization_layer
from climart.models.additional_layers import Multiscale_Module, GAP, SE_Block


class CNN_Net(BaseModel):
    def __init__(self,
                 channels_list: Sequence[int],
                 out_dim: int,
                 dilation: int = 1,
                 column_preprocesser: ColumnPreprocesser = None,
                 net_normalization: str = 'none',
                 gap: bool = False,
                 se_block: bool = False,
                 activation_function: str = 'relu',
                 dropout: float = 0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net_norm = net_normalization.lower()
        self.out_size = out_dim
        self.channel_list = channels_list
        self.linear_in_shape = 10
        self.use_linear = gap
        self.ratio = 16
        self.kernel_list = [20, 10, 5]
        self.stride_list = [2, 1, 1]# 221
        self.dilation = dilation
        self.global_average = GAP()

        feat_cnn_modules = []
        for i in range(len(self.channel_list) - 1):
            out = self.channel_list[i + 1]
            feat_cnn_modules.append(nn.Conv1d(in_channels=self.channel_list[i],
                                              out_channels=out, kernel_size=self.kernel_list[i], stride=self.stride_list[i],
                                              bias=self.net_norm != 'batch_norm', dilation=self.dilation))
            if se_block:
                feat_cnn_modules.append(SE_Block(out, self.ratio))
            if self.net_norm != 'none':
                feat_cnn_modules += [get_normalization_layer(self.net_norm, out)]
            feat_cnn_modules.append(get_activation_function(activation_function, functional=False))
            # TODO: Need to add adaptive pooling with arguments
            feat_cnn_modules.append(nn.Dropout(dropout))

        self.feat_cnn = nn.Sequential(*feat_cnn_modules)

#        input_dim = [self.channel_list[0], self.linear_in_shape]  # TODO: Need to pass input shape as argument

#        linear_input_shape = functools.reduce(operator.mul, list(self.feat_cnn(torch.rand(1, *input_dim)).shape))
#        print(linear_input_shape)
        linear_layers = []
        if not self.use_linear:
            linear_layers.append(nn.Linear(int(self.channel_list[-1]/100)*400, 256, bias=True))
            linear_layers.append(get_activation_function(activation_function, functional=False))
            linear_layers.append(nn.Dropout(dropout))
            linear_layers.append(nn.Linear(256, self.out_size, bias=True))
            self.ll = nn.Sequential(*linear_layers)

    @staticmethod
    def _input_transform(X: Dict[str, Tensor]) -> Tensor:
        X_levels = X['levels']

        # X_layers = rearrange(F.pad(rearrange(X['layers'], 'c f -> () c f'), (0,0,1,0),\
                # mode='reflect'), '() c f -> c f')
        npad = ((0, 1), (0, 0))
        X_layers = np.pad(np.array(X['layers']), pad_width=npad, mode='reflect')
        X_global = repeat(X['globals'], 'f -> c f', c = 50)
        X = torch.cat((torch.FloatTensor(X_levels), torch.FloatTensor(X_layers), torch.FloatTensor(X_global)), -1)
        return rearrange(X, 'c f -> f c')

    @staticmethod
    def _batched_input_transform(batch: Dict[str, np.ndarray]) -> Tensor:
        X_levels = torch.FloatTensor(batch[LEVELS])

        X_layers = rearrange(torch.nn.functional.pad(rearrange(torch.FloatTensor(batch[LAYERS]), 'b c f -> ()b c f'), (0,0,1,0),\
                mode='reflect'), '()b c f ->b c f')
        X_global = repeat(torch.FloatTensor(batch[GLOBALS]), 'b f ->b c f', c = 50)

        X = torch.cat((X_levels, X_layers, X_global), -1)
        return rearrange(X, 'b c f -> b f c').cpu().numpy()

    def forward(self, X: Union[Tensor, Dict[str, Tensor]]) -> Tensor:
        """
        input:
            Dict with key-values {GLOBALS: x_glob, LEVELS: x_lev, LAYERS: x_lay},
             where x_*** are the corresponding features.
        """ 
        X = self.feat_cnn(X)

        if not self.use_linear:
            X = rearrange(X, 'b f c -> b (f c)')
            X = self.ll(X)
        else:
            X = self.global_average(X)

        return X.squeeze(2)


class CNN_Multiscale(BaseModel):
    def __init__(self,
                 channels_list: Sequence[int],
                 out_dim: int,
                 dilation: int = 1,
                 gap: bool = False,
                 se_block: bool = False,
                 use_act: bool = False,
                 net_normalization: str = 'none',
                 activation_function: str = 'relu',
                 dropout: float = 0.0,
                 *args, **kwargs):
        # super().__init__(channels_list, out_dim, column_handler, projection, net_normalization,
                        #  gap, se_block, activation_function, dropout, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self.channels_per_layer = 200
        self.linear_in_shape = 10
        self.multiscale_in_shape = 10
        self.dilation = dilation
        self.ratio = 16
        self.kernel_list = [6, 4, 4]
        self.stride_list = [2, 1, 1]
        self.stride = 1
        self.use_linear = gap
        self.net_norm = net_normalization.lower()
        self.out_size = out_dim
        self.channel_list = channels_list

        feat_cnn_modules = []
        for i in range(len(self.channel_list) - 1):
            out = self.channel_list[i + 1]
            feat_cnn_modules.append(nn.Conv1d(in_channels=self.channel_list[i],
                                              out_channels=out, kernel_size=self.kernel_list[i], stride=self.stride_list[i],
                                              bias=self.net_norm != 'batch_norm'))
            if se_block:
                feat_cnn_modules.append(SE_Block(out, self.ratio))
            if self.net_norm != 'none':
                feat_cnn_modules += [get_normalization_layer(self.net_norm, self.channel_list[i+1])]
            feat_cnn_modules.append(get_activation_function(activation_function, functional=False))
            # TODO: Need to add adaptive pooling with arguments
            feat_cnn_modules.append(nn.Dropout(dropout))

        self.feat_cnn = nn.Sequential(*feat_cnn_modules)
        kwargs = {'in_channels': self.channel_list[-1], 'channels_per_layer': self.channels_per_layer,
                  'out_shape': self.linear_in_shape, 'dil_rate': self.dilation, 'use_act': use_act}
        self.pyramid = Multiscale_Module(**kwargs)

        input_dim = [self.channel_list[0], self.linear_in_shape]
        # TODO: Need to pass input shape as argument
        #linear_input_shape = functools.reduce(operator.mul, list(self.feat_cnn(torch.rand(1, *input_dim)).shape))
        linear_layers = []
        # linear_layers.append(nn.Linear(int(self.channel_list[-1]/100)*1000, 300, bias=True))
        linear_layers.append(nn.Linear(2800, 256, bias=True))
        linear_layers.append(get_activation_function(activation_function, functional=False))
        linear_layers.append(nn.Linear(256, self.out_size, bias=True))
        self.ll = nn.Sequential(*linear_layers)

    def forward(self, X: Union[Tensor, Dict[str, Tensor]]) -> Tensor:
        """
        input:
            Dict with key-values {GLOBALS: x_glob, LEVELS: x_lev, LAYERS: x_lay},
             where x_*** are the corresponding features.
        """
        if isinstance(X, dict):
            X_levels = X['levels']

            X_layers = rearrange(F.pad(rearrange(X['layers'], 'b c f -> () b c f'), (0,0,1,0),\
                 mode='reflect'), '() b c f -> b c f')
            X_global = repeat(X['globals'], 'b f -> b c f', c = X_levels.shape[1])

            X = torch.cat((X_levels, X_layers, X_global), -1)
            X = rearrange(X, 'b c f -> b f c')

        X = self.feat_cnn(X)
        X = rearrange(X, 'b f c -> b (f c)')
        X = self.ll(X)

        return X.squeeze(1)

class CNN_Trainer(BaseTrainer):
    def __init__(
            self, model_params, name="CNN", seed=None, verbose=False, model_dir="out/CNN",
            notebook_mode=False, model=None, output_normalizer=None, *args, **kwargs
    ):
        super().__init__(model_params, name=name, seed=seed, verbose=verbose, output_normalizer=output_normalizer,
                         model_dir=model_dir, notebook_mode=notebook_mode, model=model, *args, **kwargs)

        model_name = name.strip().lower()
        self.model_class = CNN_Net if model_name == 'cnn' else CNN_Multiscale
        print(self.model_class)
