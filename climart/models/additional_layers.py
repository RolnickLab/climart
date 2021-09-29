from collections import OrderedDict
from typing import Dict, Optional, Union, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from climart.models.MLP import MLPNet


class Multiscale_Module(nn.Module):

    def __init__(self, in_channels=None, channels_per_layer=None, out_shape=None,
                 dil_rate=1, use_act=False, *args, **kwargs):
        super().__init__()
        self.out_shape = out_shape
        self.use_act = use_act

        self.multi_3 = nn.Conv1d(in_channels=in_channels, out_channels=channels_per_layer,
                                 kernel_size=5, stride=1, dilation=dil_rate)
        self.multi_5 = nn.Conv1d(in_channels=in_channels, out_channels=channels_per_layer,
                                 kernel_size=6, stride=1, dilation=dil_rate)
        self.multi_7 = nn.Conv1d(in_channels=in_channels, out_channels=channels_per_layer,
                                 kernel_size=9, stride=1, dilation=dil_rate)
        self.after_concat = nn.Conv1d(in_channels=int(channels_per_layer * 3),
                                      out_channels=int(channels_per_layer / 2), kernel_size=1, stride=1)
        self.gap = GAP()

    def forward(self, x):
        x_3 = self.multi_3(x)
        x_5 = self.multi_5(x)
        x_7 = self.multi_7(x)
        x_3 = F.adaptive_max_pool1d(x_3, self.out_shape)
        x_5 = F.adaptive_max_pool1d(x_5, self.out_shape)
        x_7 = F.adaptive_max_pool1d(x_7, self.out_shape)
        x_concat = torch.cat((x_3, x_5, x_7), 1)
        x_concat = self.after_concat(x_concat)

        if self.use_act:
            return torch.sigmoid(self.gap(x)) * x_concat
        else:
            return x_concat


class GAP():
    def __init__(self):
        pass

    def __call__(self, x):
        x = F.adaptive_avg_pool1d(x, 1)
        return x


class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = GAP()
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, f = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class FeatureProjector(nn.Module):
    def __init__(
            self,
            input_name_to_feature_dim: Dict[str, int],
            projection_dim: int = 128,
            projector_n_layers: int = 1,
            projector_activation_func: str = 'Gelu',
            projector_net_normalization: str = 'none',
            projections_aggregation: Optional[Union[Callable[[Dict[str, Tensor]], Any], str]] = None,
            output_normalization: bool = True,
            output_activation_function: bool = False
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.input_name_to_feature_dim = input_name_to_feature_dim
        self.projections_aggregation = projections_aggregation

        input_name_to_projector = OrderedDict()
        for input_name, feature_dim in input_name_to_feature_dim.items():
            projector_hidden_dim = int((feature_dim + projection_dim) / 2)
            projector = MLPNet(
                input_dim=feature_dim,
                hidden_dims=[projector_hidden_dim for _ in range(projector_n_layers)],
                out_dim=projection_dim,
                activation_function=projector_activation_func,
                net_normalization=projector_net_normalization,
                dropout=0,
                output_normalization=output_normalization,
                output_activation_function=output_activation_function,
                name=f"{input_name}_MLP_projector"
            )
            input_name_to_projector[input_name] = projector

        self.input_name_to_projector = nn.ModuleDict(input_name_to_projector)

    def forward(self,
                inputs: Dict[str, Tensor]
                ) -> Union[Dict[str, Tensor], Any]:
        name_to_projection = dict()

        for name, projector in self.input_name_to_projector.items():
            # Project (batch-size, ..arbitrary_dim(s).., in_feat_dim)
            #      to (batch-size, ..arbitrary_dim(s).., projection_dim)
            in_feat_dim = self.input_name_to_feature_dim[name]
            input_tensor = inputs[name]
            shape_out = list(input_tensor.shape)
            shape_out[-1] = self.projection_dim

            projector_input = input_tensor.reshape((-1, in_feat_dim))
            # projector_input has shape (batch-size * #spatial-dims, features)
            flattened_projection = projector(projector_input)
            name_to_projection[name] = flattened_projection.reshape(shape_out)

        if self.projections_aggregation is None:
            return name_to_projection  # Dict[str, Tensor]
        else:
            return self.projections_aggregation(name_to_projection)  # Any


class PredictorHeads(nn.Module):
    """
    Module to predict (with one or more MLPs) desired output based on a 1D hidden representation.
    Can be used to:
        - readout a hidden vector to a desired output dimensionality
        - predict multiple variables with separate MLP heads (e.g. one for rsuc and rsdc each)
    """

    def __init__(
            self,
            input_dim: int,
            var_name_to_output_dim: Dict[str, int],
            separate_heads: bool = True,
            n_layers: int = 1,
            activation_func: str = 'Gelu',
            net_normalization: str = 'none',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.separate_heads = separate_heads

        predictor_heads = OrderedDict()
        mlp_shared_params = {
            'input_dim': input_dim,
            'activation_function': activation_func,
            'net_normalization': net_normalization,
            'output_normalization': False,
            'dropout': 0
        }
        if self.separate_heads:
            self.output_name_to_feature_dim = var_name_to_output_dim

            for output_name, var_out_dim in var_name_to_output_dim.items():
                predictor_hidden_dim = int((input_dim + var_out_dim) / 2)
                predictor = MLPNet(
                    hidden_dims=[predictor_hidden_dim for _ in range(n_layers)],
                    out_dim=var_out_dim,
                    **mlp_shared_params
                )
                predictor_heads[output_name] = predictor
        else:
            joint_out_dim = sum([out_dim for _, out_dim in var_name_to_output_dim.items()])
            self.output_name_to_feature_dim = {'joint_output': joint_out_dim}

            predictor_hidden_dim = int((input_dim + joint_out_dim) / 2)
            predictor = MLPNet(
                hidden_dims=[predictor_hidden_dim for _ in range(n_layers)],
                out_dim=joint_out_dim,
                **mlp_shared_params
            )
            predictor_heads['joint_output'] = predictor

        self.predictor_heads = nn.ModuleDict(predictor_heads)

    def forward(self,
                hidden_input: Tensor,  # (batch-size, hidden-dim) 1D tensor
                as_dict: bool = False,
                ) -> Union[Dict[str, Tensor], Tensor]:

        name_to_prediction = OrderedDict()
        for name, predictor in self.predictor_heads.items():
            name_to_prediction[name] = predictor(hidden_input)

        if self.separate_heads:
            if as_dict:
                return name_to_prediction
            else:
                joint_output = torch.cat(list(name_to_prediction.values()), dim=-1)
                return joint_output
        else:
            return name_to_prediction if as_dict else name_to_prediction['joint_output']


if __name__ == '__main__':
    x = torch.rand(64, 100, 15)
    # gp = GAP()
    # print(gp(x).shape)
    # in_channels = x.shape[1]
    # channels_per_layer = 200
    # out_shape = 10

    # kwargs = {'in_channels': in_channels, 'channels_per_layer': channels_per_layer, 'out_shape': out_shape,
    #           'dil_rate': 1, 'use_act': False}
    # mm = Multiscale_Module(**kwargs)
    # se = SE_Block(100, 15)
    out = SE_Block(x.shape[1], 15).forward(x)
    print(out.shape)
