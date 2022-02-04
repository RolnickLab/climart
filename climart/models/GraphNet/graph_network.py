from typing import Dict, Sequence, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from climart.data_transform.transforms import AbstractGraphTransform
from climart.models.GraphNet.constants import AggregationTypes, NODES, EDGES, GLOBALS, GRAPH_COMPONENTS, SPATIAL_DIM
from climart.models.GraphNet.graph_network_block import GraphNetBlock
from climart.models.base_model import BaseModel
from climart.models.modules.mlp import MLP


class GraphNetwork(BaseModel):
    def __init__(self,
                 # input_dim: Dict[str, int],
                 hidden_dims: Sequence[int],
                 input_transform: AbstractGraphTransform,
                 readout_which_output: Optional[str] = NODES,
                 update_mlp_n_layers: int = 1,
                 aggregator_funcs: Union[str, Dict[AggregationTypes, int]] = 'sum',
                 net_normalization: str = 'layer_norm',
                 residual: Union[bool, Dict[str, bool]] = True,
                 activation_function: str = 'Gelu',
                 output_activation_function: Optional[str] = None,
                 output_net_normalization: bool = True,
                 dropout: float = 0.0,
                 *args, **kwargs):
        """
        Args:
             readout_which_output: Which graph part to return (default: edges),
                                    can be {EDGES, NODES, GLOBALS, 'graph', None}
                                   If None or 'graph', the whole graph is returned.
        """
        super().__init__(input_transform=input_transform, *args, **kwargs)
        self.save_hyperparameters(ignore="verbose_mlp")
        assert len(self.hparams.hidden_dims) >= 1
        assert update_mlp_n_layers >= 1
        self.input_transform: AbstractGraphTransform = self.input_transform
        in_dim = self.input_transform.output_dim

        senders, receivers = self.input_transform.get_edge_idxs()
        gn_layers = []
        dims = [in_dim] + list(hidden_dims)
        for i in range(1, len(dims)):
            out_activation_function = output_activation_function if i == len(dims) - 1 else activation_function
            out_net_norm = output_net_normalization if i == len(dims) - 1 else True
            gn_layers += [
                GraphNetBlock(
                    in_dims=in_dim,
                    out_dims=dims[i],
                    senders=senders,
                    receivers=receivers,
                    n_layers=update_mlp_n_layers,
                    residual=residual,
                    net_norm=net_normalization,
                    activation=activation_function,
                    dropout=dropout,
                    output_normalization=out_net_norm,
                    output_activation_function=out_activation_function,
                    aggregator_funcs=aggregator_funcs,
                )]
            in_dim = dims[i]

        self.layers: nn.ModuleList[GraphNetBlock] = nn.ModuleList(gn_layers)
        self.output_type = readout_which_output
        if self.output_type not in [NODES, EDGES, GLOBALS, 'graph', None]:
            raise ValueError("Unsupported argument for GraphNetwork `output_type`", readout_which_output)
        if hasattr(self.input_transform, "n_edges"):
            err_msg = f"GN inferred {self.n_edges} edges, but input_tranform refers to {self.input_transform.n_edges}"
            assert self.n_edges == self.input_transform.n_edges, err_msg

    @property
    def n_edges(self):
        return self.layers[0].n_edges

    def update_graph_structure(self, senders: Sequence[int], receivers: Sequence[int]) -> None:
        for layer in self.layers:
            layer.update_graph_structure(senders, receivers)

    def forward(self, input: Dict[str, Tensor]):
        """
        input:
            Dict with key-values {GLOBALS: x_glob, LEVELS: x_lev, LAYERS: x_lay},
             where x_*** are the corresponding features.
        """
        graph_new = self.update_graph(input)
        if self.output_type is not None and self.output_type != 'graph':
            graph_component = graph_new[self.output_type]
            return graph_component.reshape(graph_component.shape[0], -1)
        else:
            return graph_new

    def update_graph(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        graph_net_input = input
        graph_new = self.layers[0](graph_net_input)
        for graph_net_block in self.layers[1:]:
            graph_new = graph_net_block(graph_new)

        return graph_new


class GN_withReadout(GraphNetwork):
    def __init__(self,
                 # input_dim,
                 # out_dim: int,
                 readout_which_output=NODES,
                 graph_pooling: str = 'mean',
                 *args, **kwargs):
        super().__init__(output_activation_function=kwargs[
                             'activation_function'] if 'activation_function' in kwargs else 'gelu',
                         *args, **kwargs)
        assert readout_which_output in GRAPH_COMPONENTS
        self.readout_which_output = readout_which_output
        self.graph_pooling = graph_pooling.lower()

        self.mlp_input_dim = self.hparams.hidden_dims[-1]
        if readout_which_output in [NODES, EDGES] and self.graph_pooling in ['sum+mean', 'mean+sum', 'mean&sum',
                                                                             'sum&mean']:
            self.mlp_input_dim = self.mlp_input_dim * 2

        self.readout_mlp = MLP(
            input_dim=self.mlp_input_dim,
            hidden_dims=[int((self.mlp_input_dim + self.raw_output_dim) / 2)],
            output_dim=self.raw_output_dim,
            dropout=kwargs['dropout'] if 'dropout' in kwargs else 0.0,
            activation_function=self.hparams.activation_function,
            net_normalization='layer_norm' if 'inst' in self.hparams.net_normalization else self.hparams.net_normalization,
            output_normalization=False, output_activation_function=None, out_layer_bias_init=self.out_layer_bias_init,
            name='GraphNet_Readout_MLP'
        )

    def forward(self, input: Dict[str, torch.Tensor]):
        final_graph = self.update_graph(input)
        output_to_use = final_graph[self.readout_which_output]

        # Graph pooling, e.g. take the mean over all node embeddings (dimension=1)
        if self.readout_which_output == GLOBALS:
            g_emb = output_to_use
        else:
            if self.graph_pooling == 'sum':
                g_emb = torch.sum(output_to_use, dim=SPATIAL_DIM)
            elif self.graph_pooling == 'mean':
                g_emb = torch.mean(output_to_use, dim=SPATIAL_DIM)
            elif self.graph_pooling == 'max':
                g_emb, _ = torch.max(output_to_use, dim=SPATIAL_DIM)  # returns (values, indices)
            elif self.graph_pooling in ['sum+mean', 'mean+sum', 'mean&sum', 'sum&mean']:
                xmean = torch.mean(output_to_use, dim=SPATIAL_DIM)
                xsum = torch.sum(output_to_use, dim=SPATIAL_DIM)  # (batch-size, out-dim)
                g_emb = torch.cat((xmean, xsum), dim=SPATIAL_DIM)  # (batch-size 2*out-dim)
            else:
                raise ValueError('Unsupported readout operation', self.graph_pooling)

        # After graph pooling: (batch-size, out-dim)
        # torch.Size([64, 100, 2])
        # torch.Size([64, 100])
        # After reshape: torch.Size([64, 200])
        out = self.readout_mlp(g_emb)
        return out

