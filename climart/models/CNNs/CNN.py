from typing import Sequence, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from climart.models.base_model import BaseModel
from climart.utils.utils import get_activation_function, get_normalization_layer
from climart.models.modules.additional_layers import Multiscale_Module, GAP, SE_Block


class CNN_Net(BaseModel):
    def __init__(self,
                 hidden_dims: Sequence[int],
                 dilation: int = 1,
                 net_normalization: str = 'none',
                 kernels: Sequence[int] = (20, 10, 5),
                 strides: Sequence[int] = (2, 1, 1),  # 221
                 gap: bool = False,
                 se_block: bool = False,
                 activation_function: str = 'relu',
                 dropout: float = 0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.output_dim = self.raw_output_dim
        self.channel_list = list(hidden_dims)
        input_dim = self.input_transform.output_dim
        self.channel_list = [input_dim] + self.channel_list

        self.linear_in_shape = 10
        self.use_linear = gap
        self.ratio = 16
        self.kernel_list = list(kernels)
        self.stride_list = list(strides)
        self.global_average = GAP()

        feat_cnn_modules = []
        for i in range(len(self.channel_list) - 1):
            out = self.channel_list[i + 1]
            feat_cnn_modules += [nn.Conv1d(in_channels=self.channel_list[i],
                                           out_channels=out, kernel_size=self.kernel_list[i],
                                           stride=self.stride_list[i],
                                           bias=self.hparams.net_normalization != 'batch_norm',
                                           dilation=self.hparams.dilation)]
            if se_block:
                feat_cnn_modules.append(SE_Block(out, self.ratio))
            if self.hparams.net_normalization != 'none':
                feat_cnn_modules += [get_normalization_layer(self.hparams.net_normalization, out)]
            feat_cnn_modules += [get_activation_function(activation_function, functional=False)]
            # TODO: Need to add adaptive pooling with arguments
            feat_cnn_modules += [nn.Dropout(dropout)]

        self.feat_cnn = nn.Sequential(*feat_cnn_modules)

        #        input_dim = [self.channel_list[0], self.linear_in_shape]  # TODO: Need to pass input shape as argument
        #        linear_input_shape = functools.reduce(operator.mul, list(self.feat_cnn(torch.rand(1, *input_dim)).shape))
        #        print(linear_input_shape)
        linear_layers = []
        if not self.use_linear:
            linear_layers.append(nn.Linear(int(self.channel_list[-1] / 100) * 400, 256, bias=True))
            linear_layers.append(get_activation_function(activation_function, functional=False))
            linear_layers.append(nn.Dropout(dropout))
            linear_layers.append(nn.Linear(256, self.output_dim, bias=True))
            self.ll = nn.Sequential(*linear_layers)

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
                 hidden_dims: Sequence[int],
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
        self.save_hyperparameters()
        self.channels_per_layer = 200
        self.linear_in_shape = 10
        self.multiscale_in_shape = 10
        self.ratio = 16
        self.kernel_list = [6, 4, 4]
        self.stride_list = [2, 1, 1]
        self.stride = 1
        self.use_linear = gap
        self.out_size = out_dim
        self.channel_list = hidden_dims

        feat_cnn_modules = []
        for i in range(len(self.channel_list) - 1):
            out = self.channel_list[i + 1]
            feat_cnn_modules.append(nn.Conv1d(in_channels=self.channel_list[i],
                                              out_channels=out, kernel_size=self.kernel_list[i],
                                              stride=self.stride_list[i],
                                              bias=self.hparams.net_normalization != 'batch_norm'))
            if se_block:
                feat_cnn_modules.append(SE_Block(out, self.ratio))
            if self.hparams.net_normalization != 'none':
                feat_cnn_modules += [get_normalization_layer(self.hparams.net_normalization, self.channel_list[i + 1])]
            feat_cnn_modules.append(get_activation_function(activation_function, functional=False))
            # TODO: Need to add adaptive pooling with arguments
            feat_cnn_modules.append(nn.Dropout(dropout))

        self.feat_cnn = nn.Sequential(*feat_cnn_modules)
        kwargs = {'in_channels': self.channel_list[-1], 'channels_per_layer': self.channels_per_layer,
                  'out_shape': self.linear_in_shape, 'dil_rate': self.hparams.dilation, 'use_act': use_act}
        self.pyramid = Multiscale_Module(**kwargs)

        input_dim = [self.channel_list[0], self.linear_in_shape]
        # TODO: Need to pass input shape as argument
        # linear_input_shape = functools.reduce(operator.mul, list(self.feat_cnn(torch.rand(1, *input_dim)).shape))
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

            X_layers = rearrange(F.pad(rearrange(X['layers'], 'b c f -> () b c f'), (0, 0, 1, 0), \
                                       mode='reflect'), '() b c f -> b c f')
            X_global = repeat(X['globals'], 'b f -> b c f', c=X_levels.shape[1])

            X = torch.cat((X_levels, X_layers, X_global), -1)
            X = rearrange(X, 'b c f -> b f c')

        X = self.feat_cnn(X)
        X = rearrange(X, 'b f c -> b (f c)')
        X = self.ll(X)

        return X.squeeze(1)
