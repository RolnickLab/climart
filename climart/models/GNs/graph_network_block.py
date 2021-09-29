from collections import OrderedDict
from functools import partial
from typing import Dict, Union, Optional, Sequence
from einops import rearrange, repeat

import numpy as np
import torch
import torch.nn as nn
import torch_scatter
from climart.models.MLP import MLPNet
import climart.models.GNs.constants as gn_constants
from climart.models.GNs.constants import GraphComponentToTensor, AggregationTypes, NODES, EDGES, GLOBALS


class GraphNetBlock(nn.Module):
    def __init__(self,
                 in_dims: Union[int, Dict[str, int]],
                 out_dims: Union[int, Dict[str, int]],
                 senders: Sequence[int],
                 receivers: Sequence[int],
                 n_layers: int = 1,
                 use_edge_features: bool = True,
                 use_global_features: bool = True,
                 residual: Union[bool, Dict[str, bool]] = False,
                 net_norm: str = 'none',
                 activation: str = 'relu',
                 dropout: float = 0,
                 output_normalization: bool = True,
                 output_activation_function: Optional[str] = None,
                 aggregator_funcs: Union[str, Dict[AggregationTypes, int]] = 'sum',
                 verbose: bool = False
                 ):
        super().__init__()
        if isinstance(in_dims, int):
            in_dims: dict = {c: in_dims for c in gn_constants.GRAPH_COMPONENTS}
        if isinstance(out_dims, int):
            out_dims: dict = {c: out_dims for c in gn_constants.GRAPH_COMPONENTS}

        self.components = gn_constants.GRAPH_COMPONENTS
        self.use_edge_features = use_edge_features
        self.use_global_features = use_global_features

        if not use_edge_features:
            in_dims[EDGES] = out_dims[EDGES] = 0
            self.components.remove(EDGES)
        if not use_global_features:
            in_dims[GLOBALS] = out_dims[GLOBALS] = 0
            self.components.remove(GLOBALS)

        n_feats_e = in_dims[EDGES]
        n_feats_n = in_dims[NODES]
        n_feats_u = in_dims[GLOBALS]
        self._n_edges = None
        self.update_graph_structure(senders, receivers)
        self.residual = {c: residual for c in self.components}

        in_dims = {
            EDGES: 2 * n_feats_n + n_feats_e + n_feats_u,
            NODES: n_feats_n + out_dims[EDGES] + n_feats_u,
            GLOBALS: out_dims[NODES] + out_dims[EDGES] + n_feats_u
        }

        update_funcs = OrderedDict()  # nn.ModuleDict()
        for component in self.components:
            c_in_dim = in_dims[component]
            out_dim = out_dims[component]
            if c_in_dim != out_dim:
                self.residual[component] = False

            in_dim = in_dims[component]
            hdim = int((in_dim + out_dim) / 2)

            update_funcs[component] = MLPNet(
                input_dim=in_dim,
                hidden_dims=[hdim for _ in range(n_layers)],
                out_dim=out_dim,
                activation_function=activation,
                net_normalization=net_norm,
                dropout=dropout,
                output_normalization=output_normalization,
                output_activation_function=output_activation_function,
                name=f'GN_{component}_update_MLP',
                verbose=verbose
            )
        self.update_funcs = nn.ModuleDict(update_funcs)

        self.aggregator_funcs = OrderedDict()  # nn.ModuleDict()
        for aggregation in AggregationTypes:
            agg_func = aggregator_funcs[aggregation] if isinstance(aggregator_funcs, dict) else aggregator_funcs
            agg_func = agg_func.lower()
            agg_dim = gn_constants.SPATIAL_DIM
            if agg_func == 'sum':
                if aggregation == AggregationTypes.AGG_E_TO_N:
                    # agg func returns a (batch-size, #nodes, #edge-feats) tensor
                    agg_func = partial(torch_scatter.scatter_sum, dim=agg_dim)
                else:
                    agg_func = partial(torch.sum, dim=agg_dim)
            elif agg_func == 'mean':
                if aggregation == AggregationTypes.AGG_E_TO_N:
                    agg_func = partial(torch_scatter.scatter_mean, dim=agg_dim)
                else:
                    agg_func = partial(torch.mean, dim=agg_dim)
            elif agg_func == 'max':
                if aggregation == AggregationTypes.AGG_E_TO_N:
                    def max_scatter_partial(x, index):
                        return torch_scatter.scatter_max(x, dim=agg_dim, index=index)[0]
                    agg_func = max_scatter_partial
                else:
                    agg_func = lambda x: torch.max(x, dim=agg_dim)[0]  # returns (values, indices)[0]
            # elif agg_func in ['sum+mean', 'mean+sum', 'mean&sum', 'sum&mean']:
            #    xmean = torch.mean(final_embs, dim=1)
            #    xsum = torch.sum(final_embs, dim=1)  # (batch-size, out-dim)
            #    g_emb = torch.cat((xmean, xsum), dim=1)  # (batch-size 2*out-dim)
            else:
                raise ValueError('Unsupported aggregation operation')
            self.aggregator_funcs[aggregation] = agg_func

    @property
    def n_edges(self):
        return self._n_edges

    def update_graph_structure(self, senders: Sequence[int], receivers: Sequence[int]) -> None:
        if not torch.is_tensor(senders):
            senders = torch.from_numpy(senders) if isinstance(senders, np.ndarray) else torch.tensor(senders)
        if not torch.is_tensor(receivers):
            receivers = torch.from_numpy(receivers) if isinstance(receivers, np.ndarray) else torch.tensor(receivers)
        self.register_buffer('_senders', senders.long())  # so that they are moved to correct device
        self.register_buffer('_receivers', receivers.long())
        assert len(self._receivers) == len(self._senders), "Sender and receiver must both have #edges indices"
        self._n_edges = len(self._senders)

    #   def _aggregate_edges_for_node(self, mode, edge_feats_old: Tensor) -> Tensor:
    #       """ Return a (batch-size, #nodes, #edge-feats) tensor, K, where K_bij = Agg({e_j | receiver_j = i}) """
    #       indices = self._senders if mode == gn_constants.SENDERS else self._receivers
    #       batch_size = edge_feats_old.shape[0]

    def forward(self, graph: GraphComponentToTensor) -> GraphComponentToTensor:
        if self.use_edge_features:
            edge_feats_old = graph[EDGES]  # edge_feats_old have shape (b, #edges, #edge-feats)
            if edge_feats_old.shape[gn_constants.SPATIAL_DIM] != self._n_edges:
                    raise ValueError(f"Edge features imply {edge_feats_old.shape[gn_constants.SPATIAL_DIM]} edges, "
                                     f"while sender and receiver lists imply {self._n_edges} edges.")
        node_feats_old = graph[NODES]  # node_feats_old have shape (b, #nodes, #node-feats)
        n_nodes = node_feats_old.shape[gn_constants.SPATIAL_DIM]
        if self.use_global_features:
            global_feats_old = graph[GLOBALS]  # global_feats_old have shape (b, #global-feats)
            batch_size, n_glob_feats = global_feats_old.shape

        out = {c: None for c in self.components}

        # ----------------------- Update edges
        # self.senders and self.receivers are a sequence of indices with #edges elements
        sender_feats = node_feats_old.index_select(index=self._senders, dim=gn_constants.SPATIAL_DIM)
        receiver_feats = node_feats_old.index_select(index=self._receivers, dim=gn_constants.SPATIAL_DIM)

        # print('E:', edge_feats_old.shape, 'V:', node_feats_old.shape, 'U:', global_feats_old.shape)
        # print('Senders', sender_feats.shape, 'Recvs', receiver_feats.shape, global_feats_unsqueezed.shape)
        mlp_input_e = torch.cat([
            edge_feats_old,  # (b, #edges, #edge-feats)
            sender_feats,  # (b, #edges, #node-feats)
            receiver_feats,  # (b, #edges, #node-feats)
            repeat(global_feats_old, 'b g -> b e g', e=self._n_edges)  # (1, self.n_edges, 1) (b, 1, #global-feats)
        ], dim=-1)

        mlp_input_e = rearrange(mlp_input_e, 'b e d1 -> (b e) d1')

        out[EDGES] = self.update_funcs[EDGES](mlp_input_e)
        out[EDGES] = rearrange(out[EDGES], '(b e) d2 -> b e d2', b=batch_size, e=self._n_edges)
        if self.residual[EDGES]:
            out[EDGES] += edge_feats_old
        # ----------------------- Update nodes
        aggregated_edge_feats_for_node = self.aggregator_funcs[AggregationTypes.AGG_E_TO_N](
            out[EDGES], index=self._receivers
        )

        mlp_input_n = torch.cat([
            aggregated_edge_feats_for_node,  # (b, #nodes, #edge-feats)
            node_feats_old,  # (b, #nodes, #node-feats)
            repeat(global_feats_old, 'b g -> b n g', n=n_nodes)  # (b, #nodes, #global-feats)
        ], dim=-1)
        mlp_input_n = rearrange(mlp_input_n, 'b n d1 -> (b n) d1')

        # print('Agg E:', aggregated_edge_feats_for_node.shape, 'V:', node_feats_old.shape, mlp_input_n.shape)
        out[NODES] = self.update_funcs[NODES](mlp_input_n)
        out[NODES] = rearrange(out[NODES], '(b n) d2 -> b n d2', b=batch_size, n=n_nodes)
        if self.residual[NODES]:
            out[NODES] += node_feats_old
        # ----------------------- Update global features
        aggregated_edge_feats_for_global = self.aggregator_funcs[AggregationTypes.AGG_E_TO_U](out[EDGES])
        aggregated_node_feats_for_global = self.aggregator_funcs[AggregationTypes.AGG_N_TO_U](out[NODES])
        # print('Agg EU:', aggregated_edge_feats_for_global.shape, 'VU:', aggregated_node_feats_for_global.shape)

        mlp_input_u = torch.cat([
            aggregated_edge_feats_for_global,  # (b, #edge-feats)
            aggregated_node_feats_for_global,  # (b, #node-feats)
            global_feats_old  # (b, #global-feats)
        ], dim=-1)
        out[GLOBALS] = self.update_funcs[GLOBALS](mlp_input_u)
        if self.residual[GLOBALS]:
            out[GLOBALS] += global_feats_old

        # for c in self.components:
        #    if self.residual[c]:
        #        out[c] += graph[c]  # residual connection

        return out



if __name__ == '__main__':
    nedges = 50
    nnodes = 49
    nfe, nfn, nfu = 2, 22, 82
    b = 64
    E = torch.randn((b, nedges, nfe))
    V = torch.randn((b, nnodes, nfn))
    U = torch.randn((b, nfu))

    senders = torch.arange(nedges) % nnodes
    receivers = (torch.arange(nedges) + 1) % nnodes

    gnl = GraphNetBlock(
        in_dims={NODES: nfn, EDGES: nfe, GLOBALS: nfu}, out_dims=128, senders=senders, receivers=receivers
    )

    X_in = {NODES: V, EDGES: E, GLOBALS: U}
    x = gnl(X_in)
    print([k + str(xx.shape) for k, xx in x.items()])
