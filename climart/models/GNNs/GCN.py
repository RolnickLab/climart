from typing import Dict, Sequence, Optional, Union

import torch
import torch.nn as nn
import numpy as np
import wandb
from torch import Tensor
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from climart.models.GNNs.structure_learner import EdgeStructureLearner
from climart.models.MLP import MLPNet
from climart.models.base_model import BaseModel, BaseTrainer
from climart.models.GNNs.graph_conv_layer import GraphConvolution
from climart.models.column_handler import ColumnPreprocesser
from climart.utils.utils import get_activation_function, adj_to_edge_indices


class GCN(BaseModel):
    def __init__(self,
                 input_dim: Union[int, Dict[str, int]],
                 hidden_dims: Sequence[int],
                 out_dim: int,
                 column_preprocesser: ColumnPreprocesser,
                 learn_edge_structure: bool = False,
                 degree_normalized_adj: bool = False,  # normalize by in-degree
                 improved_self_loops: bool = False,  # self loops get weighted twice as much as rest edges
                 net_normalization: str = 'none',
                 residual: bool = True,
                 activation_function: str = 'relu',
                 output_activation_function: Optional[str] = None,
                 dropout: float = 0.0,
                 pyg_builtin: bool = False,
                 device='cuda', verbose=True,
                 *args, **kwargs):
        super().__init__(*args, verbose=verbose, **kwargs)
        self.hidden_dims = hidden_dims
        assert len(self.hidden_dims) >= 1  # self.L > 1
        self.column_preprocesser = column_preprocesser
        self.preprocess_func = self.column_preprocesser.get_preprocesser(verbose=verbose)
        in_dim = self.column_preprocesser.out_dim

        self.out_dim = out_dim
        self.act = activation_function
        self.net_norm = net_normalization
        self.jumping_knowledge = False  # net_params['jumping_knowledge']
        self.line_graph_adj = column_preprocesser.get_adj(
                degree_normalized=degree_normalized_adj, improved=improved_self_loops
            ).float().to(device)
        self.num_nodes = self.line_graph_adj.shape[0]
        self.line_graph_adj = self.line_graph_adj.to_sparse()

        self.learn_edge_structure = learn_edge_structure
        if self.learn_edge_structure:
            max_num_edges = 0.05 * (self.num_nodes ** 2) + 2 * self.num_nodes
            if verbose:
                self.log.info(" Learning the graph structure!")

            self.register_buffer('learned_adj', None)
            self.structure_learner = EdgeStructureLearner(
                num_nodes=self.num_nodes, max_num_edges=max_num_edges, dim=50, static_feat=None, self_loops=True
            )
        if pyg_builtin:
            self.edge_idxs = adj_to_edge_indices(self.line_graph_adj)  # Torch geometric operates on edge lists
            self.line_graph_adj = None
        else:
            self.edge_idxs = None
        activation = get_activation_function(self.act, functional=True, num=1, device=device)

        shared_kwargs = {'residual': residual, 'pyg_builtin': pyg_builtin, 'improved_self_loops': improved_self_loops}
        conv_kwargs = {'activation': activation,
                       'net_norm': self.net_norm,
                       'dropout': dropout}
        gcn_layers = []
        dims = [in_dim] + list(self.hidden_dims)
        for i in range(1, len(dims)):
            gcn_layers += [
                GraphConvolution(dims[i - 1], dims[i], **shared_kwargs, **conv_kwargs)
            ]

        # Last output layer:
        final_conv_kwargs = {
            'activation': get_activation_function(output_activation_function, functional=True, device=device),
            'net_norm': None, 'dropout': 0
        }
        gcn_layers += [
            GraphConvolution(dims[-1], self.out_dim, **shared_kwargs, **final_conv_kwargs)
        ]
        self.layers = nn.ModuleList(gcn_layers)
        self.output_mask = self.column_preprocesser.get_output_mask()
        self.n_relevant_nodes = torch.count_nonzero(self.output_mask)

        if verbose:
            print('------------------->', self.output_mask.shape, self.n_relevant_nodes, self.out_dim)
            print([x for x in self.layers])

    def forward(self, input: Dict[str, Tensor]):
        """
        input:
            Dict with key-values {GLOBALS: x_glob, LEVELS: x_lev, LAYERS: x_lay},
             where x_*** are the corresponding features.
        """
        node_embs = self.get_node_embeddings(input)
        if self.output_mask is not None:
            node_embs = node_embs[:, self.output_mask, :]  # just take a subset of nodes

        return node_embs.reshape(node_embs.shape[0], -1)

    def get_node_embeddings(self, input: Dict[str, Tensor]) -> Tensor:
        projected_input = self.preprocess_func(input)
        if self.learn_edge_structure:
            if self.training or self.learned_adj is None:
                self.learned_adj = adj = self.structure_learner().to_sparse()
            else:
                adj = self.learned_adj
        else:
            adj = self.line_graph_adj
        # GCN forward pass --> Generate node embeddings
        node_embs = self.layers[0](projected_input, adj, self.edge_idxs)  # shape (batch-size, #nodes, #features)
        # print(0, input.shape, self.adj.shape, self.layers[0].weight.shape)
        # i=1
        for conv in self.layers[1:]:
            #  shape (batch-size, #nodes, #in-features) -> (batch-size, #nodes, #out-features)
            node_embs = conv(node_embs, adj, self.edge_idxs)
            # print(i, node_embs.shape, self.adj.shape, self.layers[i].weight.shape)
            # i+=1
        return node_embs


class GCN_withReadout(GCN):
    def __init__(self,
                 input_dim,
                 hidden_dims: Sequence[int],
                 out_dim: int,
                 graph_pooling: str = 'flatten',
                 jumping_knowledge: bool = False, *args, **kwargs):
        super().__init__(input_dim=input_dim,
                         hidden_dims=hidden_dims,
                         out_dim=hidden_dims[-1],
                         output_activation_function='gelu',
                         *args, **kwargs)
        self.graph_pooling = graph_pooling.lower()

        self.jumping_knowledge = jumping_knowledge
        self.mlp_input_dim = self.out_dim
        if self.jumping_knowledge:
            self.mlp_input_dim += + sum(self.hidden_dims)
        if self.graph_pooling in ['sum+mean', 'mean+sum', 'mean&sum', 'sum&mean']:
            self.mlp_input_dim = self.mlp_input_dim * 2
        elif self.graph_pooling == 'flatten':
            self.mlp_input_dim *= self.num_nodes

        mlp_params = {
            'input_dim': int(self.mlp_input_dim),
            'hidden_dims': [int(self.mlp_input_dim)],
            'out_dim': out_dim, #int(50*out_dim), #int(self.n_relevant_nodes * out_dim),
            'dropout': kwargs['dropout'] if 'dropout' in kwargs else 0.,
            'activation_function': self.act,
            'net_normalization': self.net_norm if 'inst' not in self.net_norm else 'layer_norm',
        }
        self.readout = MLPNet(**mlp_params, name='GCN_Readout_MLP', out_layer_bias_init=self.out_layer_bias_init)

    def forward(self, input: Dict[str, torch.Tensor]):
        final_embs = self.get_node_embeddings(input)

        # if self.jumping_knowledge:
        #        X_all_embeddings = torch.cat(, node_embs), dim=2)
        # final_embs = X_all_embeddings if self.jumping_knowledge else node_embs

        # Graph pooling, e.g. take the mean over all node embeddings (dimension=1)
        if self.graph_pooling == 'flatten':
            g_emb = final_embs.reshape(final_embs.shape[0], -1)
        elif self.graph_pooling == 'sum':
            g_emb = torch.sum(final_embs, dim=1)
        elif self.graph_pooling == 'mean':
            g_emb = torch.mean(final_embs, dim=1)
        elif self.graph_pooling == 'max':
            g_emb, _ = torch.max(final_embs, dim=1)  # returns (values, indices)
        elif self.graph_pooling in ['sum+mean', 'mean+sum', 'mean&sum', 'sum&mean']:
            xmean = torch.mean(final_embs, dim=1)
            xsum = torch.sum(final_embs, dim=1)  # (batch-size, out-dim)
            g_emb = torch.cat((xmean, xsum), dim=1)  # (batch-size 2*out-dim)
        else:
            raise ValueError('Unsupported readout operation')

        # After graph pooling: (batch-size, out-dim)
        # torch.Size([64, 100, 2])
        # torch.Size([64, 100])
        # After reshape: torch.Size([64, 200])
        out = self.readout(g_emb)
        return out


# -------------------------------------------------------------------------------- TRAINER
class GCN_Trainer(BaseTrainer):
    def __init__(
            self, model_params, column_preprocesser, name='GCN', seed=None, verbose=False, model_dir="out/GCN",
            notebook_mode=False, model=None, output_normalizer=None, *args, **kwargs
    ):
        super().__init__(model_params, name=name, seed=seed, verbose=verbose, output_normalizer=output_normalizer,
                         model_dir=model_dir, notebook_mode=notebook_mode, model=model, *args, **kwargs)
        name2 = self.name.lower().replace('&', '+').replace('gcn', 'gnn')
        if name2 in ['gnn+readout']:
            self.model_class = GCN_withReadout
            print('Using a GCN with readout...')
        elif name2 == 'gnn':
            self.model_class = GCN
        else:
            raise ValueError(name2 + ' is unknown')
        self.column_preprocesser = column_preprocesser

    def _model_init_kwargs(self):
        return {**super()._model_init_kwargs(), 'column_preprocesser': self.column_preprocesser}

    def on_train_end(self, logging_dict):
        if self.model.learn_edge_structure:
            adj = self.model.learned_adj.detach().to_dense()
            adj2 = self.model.line_graph_adj.to_dense()
            static_adj_overlap = torch.count_nonzero(adj == adj2)
            static_line_graph_overlap = torch.count_nonzero(
                adj[1:, 1:] == adj2[1:, 1:]
            )
            logging_dict['Learned_adj/static_adj_overlap'] = int(static_adj_overlap)
            logging_dict['Learned_adj/static_line_graph_overlap'] = int(static_line_graph_overlap)
            static_adj_lg = torch.count_nonzero(adj * adj2) - self.model.num_nodes
            static_lg_no_glob = torch.count_nonzero(adj[1:, 1:] * adj2[1:, 1:]) - (self.model.num_nodes - 1)
            logging_dict['Learned_adj/n_static_adj_edges'] = int(static_adj_lg)
            logging_dict['Learned_adj/n_line_graph_edges'] = int(static_lg_no_glob)
            logging_dict['Learned_adj/n_global_edges'] = int(torch.count_nonzero(adj[0, 1:]) + torch.count_nonzero(adj[1:, 0]))

            graph = get_graph_from_adj(adj.cpu().numpy(), min_weight=0.05)
            try:
                centrality = nx.eigenvector_centrality(graph, max_iter=200)
            except nx.PowerIterationFailedConvergence:
                try:
                    self.log.info('EV centrality computation failed to converge, trying with more iterations..')
                    centrality = nx.eigenvector_centrality(graph, max_iter=500, tol=1e-5)
                except nx.PowerIterationFailedConvergence:
                    self.log.info('EV centrality computation failed to converge.')
                    return logging_dict
            centrality = np.array([centrality[x] for x in range(self.model.num_nodes)])

            logging_dict['Learned_adj/eigv_global'] = centrality[0]
            if hasattr(self.column_preprocesser, 'LEVEL_NODES') and self.column_preprocesser.LEVEL_NODES is not None:
                logging_dict['Learned_adj/eigv_TOA'] = centrality[1]
                logging_dict['Learned_adj/eigv_TOA_layer'] = centrality[2]
                logging_dict['Learned_adj/eigv_surface_layer'] = centrality[-2]
                logging_dict['Learned_adj/eigv_surface'] = centrality[-1]
            else:
                logging_dict['Learned_adj/eigv_TOA_layer'] = centrality[1]
                logging_dict['Learned_adj/eigv_surface_layer'] = centrality[-1]

            if self.current_epoch % 5 == 1:
                fig, ax = plt.subplots(1)
                im = ax.imshow(centrality[np.newaxis, :], aspect="auto", cmap='Oranges')
                plt.colorbar(im)
                logging_dict['Learned_adj/Eigv_centrality'] = fig
                fig, ax = plt.subplots(1)
                sns.heatmap(adj.cpu().numpy(), linewidth=0.001)
                logging_dict['Learned_adj/Adjacency_matrix'] = wandb.Image(fig)

        return logging_dict


def get_graph_from_adj(adj: np.ndarray, min_weight=0.1, directed=True):
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(list(range(adj.shape[0])))
    rows, cols = np.where(adj > min_weight)
    edges = zip(rows.tolist(), cols.tolist())
    graph.add_edges_from(edges)
    return graph



