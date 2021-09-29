from typing import Sequence, Optional, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from climart.models.base_model import BaseModel, BaseTrainer
from climart.utils.utils import get_activation_function, get_normalization_layer


class MLPNet(BaseModel):
    def __init__(self,
                 input_dim: Union[Dict[str, int], int],
                 hidden_dims: Sequence[int],
                 out_dim: int,
                 spatial_dim: Optional[Dict[str, int]] = None,
                 net_normalization: Optional[str] = None,
                 activation_function: str = 'relu',
                 dropout: float = 0.0,
                 residual: bool = False,
                 output_normalization: bool = False,
                 output_activation_function: Optional[Union[str, bool]] = None,
                 *args, **kwargs):
        """
        Args:
            input_dim must either be an int, i.e. the expected 1D input tensor dim, or a dict s.t.
                input_dim and spatial_dim have the same keys to compute the flattened input shape.
            output_activation_function (str, bool, optional): By default no output activation function is used (None).
                If a string is passed, is must be the name of the desired output activation (e.g. 'softmax')
                If True, the same activation function is used as defined by the arg `activation_function`.
        """
        super().__init__(*args, **kwargs)
        self.net_norm = net_normalization.lower() if isinstance(net_normalization, str) else 'none'
        self.out_size = out_dim
        self.residual = residual
        if isinstance(input_dim, dict):
            assert all([k in spatial_dim.keys() for k in input_dim.keys()])
            input_dim = sum([input_dim[k] * max(1, spatial_dim[k]) for k in input_dim.keys()])  # flattened
            self.log.info(f' Inferred a flattened input dim = {input_dim}')
        hidden_layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(1, len(dims)):
            hidden_layers += [MLP_Block(
                in_dim=dims[i - 1],
                out_dim=dims[i],
                net_norm=self.net_norm,
                activation_function=activation_function,
                dropout=dropout,
                residual=residual
            )]
        self.hidden_layers = nn.ModuleList(hidden_layers)

        out_weight = nn.Linear(dims[-1], self.out_size, bias=True)
        if self.out_layer_bias_init is not None:
            self.log.info(' Pre-initializing the MLP final/output layer bias.')
            out_weight.bias.data = self.out_layer_bias_init
        out_layer = [out_weight]
        if output_normalization and self.net_norm != 'none':  # in ['layer_norm', 'batch_norm']:
            out_layer += [get_normalization_layer(self.net_norm, self.out_size)]
        if output_activation_function is not None and output_activation_function:
            if isinstance(output_activation_function, bool):
                output_activation_function = activation_function

            out_layer += [get_activation_function(output_activation_function, functional=False)]
        self.out_layer = nn.Sequential(*out_layer)

    @staticmethod
    def _input_transform(X: Dict[str, Tensor]) -> Tensor:
        return torch.cat([torch.flatten(subX) for subX in X.values()], dim=0)

    @staticmethod
    def _batched_input_transform(X: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([subX.reshape((subX.shape[0], -1)) for subX in X.values()], axis=1)
        # return torch.cat([torch.flatten(subX, start_dim=1).unsqueeze(1) for subX in X.values()], dim=1)

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


class MLP_Trainer(BaseTrainer):
    def __init__(
            self, model_params, name='MLP', seed=None, verbose=False, model_dir="out/MLP",
            notebook_mode=False, model=None, output_normalizer=None, *args, **kwargs
    ):
        super().__init__(model_params, name=name, seed=seed, verbose=verbose, output_normalizer=output_normalizer,
                         model_dir=model_dir, notebook_mode=notebook_mode, model=model, *args, **kwargs)
        self.model_class = MLPNet
        self.name = name
