from typing import Sequence, Optional, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from climart.models.base_model import BaseModel
from climart.models.modules.mlp import MLP


class ClimartMLP(BaseModel):
    def __init__(self,
                 hidden_dims: Sequence[int],
                 datamodule_config: DictConfig = None,
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
        super().__init__(datamodule_config=datamodule_config, *args, **kwargs)
        self.save_hyperparameters()

        if isinstance(self.raw_input_dim, dict):
            assert all([k in self.raw_spatial_dim.keys() for k in self.raw_input_dim.keys()])
            self.input_dim = sum([self.raw_input_dim[k] * max(1, self.raw_spatial_dim[k]) for k in self.raw_input_dim.keys()])  # flattened
            self.log_text.info(f' Inferred a flattened input dim = {self.input_dim}')
        else:
            self.input_dim = self.raw_input_dim

        self.output_dim = self.raw_output_dim

        self.mlp = MLP(
            self.input_dim, hidden_dims, self.output_dim,
            net_normalization=net_normalization, activation_function=activation_function, dropout=dropout,
            residual=residual, output_normalization=output_normalization,
            output_activation_function=output_activation_function, out_layer_bias_init=self.out_layer_bias_init
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.mlp(X)
