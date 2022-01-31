from typing import Sequence, Optional, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from climart.utils.utils import get_activation_function, get_normalization_layer, get_logger

log = get_logger(__name__)


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int],
                 output_dim: int,
                 net_normalization: Optional[str] = None,
                 activation_function: str = 'gelu',
                 dropout: float = 0.0,
                 residual: bool = False,
                 output_normalization: bool = False,
                 output_activation_function: Optional[Union[str, bool]] = None,
                 out_layer_bias_init: Tensor = None,
                 name: str = ""
                 ):
        """
        Args:
            input_dim (int): the expected 1D input tensor dim
            output_activation_function (str, bool, optional): By default no output activation function is used (None).
                If a string is passed, is must be the name of the desired output activation (e.g. 'softmax')
                If True, the same activation function is used as defined by the arg `activation_function`.
        """

        super().__init__()
        self.name = name
        hidden_layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(1, len(dims)):
            hidden_layers += [MLP_Block(
                in_dim=dims[i - 1],
                out_dim=dims[i],
                net_norm=net_normalization.lower() if isinstance(net_normalization, str) else 'none',
                activation_function=activation_function,
                dropout=dropout,
                residual=residual
            )]
        self.hidden_layers = nn.ModuleList(hidden_layers)

        out_weight = nn.Linear(dims[-1], output_dim, bias=True)
        if out_layer_bias_init is not None:
            log.info(' Pre-initializing the MLP final/output layer bias.')
            out_weight.bias.data = out_layer_bias_init
        out_layer = [out_weight]
        if output_normalization and net_normalization != 'none':
            out_layer += [get_normalization_layer(net_normalization, output_dim)]
        if output_activation_function is not None and output_activation_function:
            if isinstance(output_activation_function, bool):
                output_activation_function = activation_function

            out_layer += [get_activation_function(output_activation_function, functional=False)]
        self.out_layer = nn.Sequential(*out_layer)

    def forward(self, X: Tensor) -> Tensor:
        for layer in self.hidden_layers:
            X = layer(X)

        Y = self.out_layer(X)
        return Y.squeeze(1)


class MLP_Block(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 net_norm: str = 'none',
                 activation_function: str = 'Gelu',
                 dropout: float = 0.0,
                 residual: bool = False
                 ):
        super().__init__()
        layer = [nn.Linear(in_dim, out_dim, bias=net_norm != 'batch_norm')]
        if net_norm != 'none':
            layer += [get_normalization_layer(net_norm, out_dim)]
        layer += [get_activation_function(activation_function, functional=False)]
        if dropout > 0:
            layer += [nn.Dropout(dropout)]
        self.layer = nn.Sequential(*layer)
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False
        elif residual:
            print('MLP block with residual!')

    def forward(self, X: Tensor) -> Tensor:
        X_out = self.layer(X)
        if self.residual:
            X_out += X
        return X_out
