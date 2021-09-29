from typing import Dict, Optional, Tuple, Sequence, Any

import numpy as np
import torch
from torch import Tensor
from einops import repeat

from climart.models.GNs.constants import NODES, EDGES
from climart.data_wrangling.constants import LAYERS, LEVELS, GLOBALS
from climart.models.additional_layers import FeatureProjector
from climart.utils.utils import normalize_adjacency_matrix_torch, identity, get_logger

log = get_logger(__name__)


def get_dict_entry(dictio: dict, possible_keys: Sequence[Any]):
    for possible_key in possible_keys:
        if possible_key in dictio.keys():
            return dictio[possible_key]
        else:
            pass


class ColumnPreprocesser:
    ONLY_LAYER_NODES = ['duplication', 'graph_net_layer_nodes', 'identity']
    ONLY_LEVEL_NODES = ['graph_net_level_nodes']

    def __init__(self,
                 n_layers: int,
                 input_dims: Dict[str, int],
                 preprocessing: str,
                 projector_hidden_dim: int = 128,  # only if preprocessing == 'mlp'
                 projector_n_layers: int = 1,  # only if preprocessing == 'mlp'
                 projector_net_normalization: str = 'layer_norm',  # only if preprocessing == 'mlp'
                 drop_node_encoding: bool = True,  # only used if preprocessing == 'duplication'
                 node_encoding_len: int = 3,  # only used if preprocessing == 'duplication'
                 use_level_features: bool = True,
                 drop_last_level: bool = True  # only used if preprocessing == 'duplication'
                 ):
        self.input_dims = input_dims
        self.n_lay = n_layers
        self.n_lev = self.n_lay + 1
        self.out_dim = None
        self.as_string = ""
        self.preprocessing_type = preprocessing.lower()
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_n_layers = projector_n_layers
        self.projector_net_normalization = projector_net_normalization
        self.use_level_features = use_level_features
        if not self.use_level_features:
            log.info(' Dropping level features!')

        if self.preprocessing_type in self.ONLY_LAYER_NODES:
            self.n_nodes = self.n_lay
            self.LAYER_NODES = slice(0, self.n_nodes)  # all nodes are layer nodes
            if self.preprocessing_type in ['duplication']:
                self.drop_node_encoding = drop_node_encoding
                self.node_encoding_len = node_encoding_len
                self.drop_last_level = drop_last_level
        elif self.preprocessing_type in self.ONLY_LEVEL_NODES:
            self.n_nodes = self.n_lev
            self.LEVEL_NODES = slice(0, self.n_nodes)  # all nodes are layer nodes
        else:  # use global node
            self.n_nodes = self.n_lev + self.n_lay + 1 if self.use_level_features else self.n_lay + 1
            self.GLOBAL_NODE = 0
            if self.use_level_features:
                self.LEVEL_NODES = slice(1, self.n_nodes, 2)  # start at 1, then every second, [1, 3, 5, 7,...]
                self.LAYER_NODES = slice(2, self.n_nodes, 2)  # [2, 4, 6,...]
            else:
                self.LAYER_NODES = slice(1, self.n_nodes)  # [1, 2, 3,...]

        log.info(f" Inferred number of nodes/spatial dimension: {self.n_nodes}")
        self.FEATURE_DIM_IN = 2  # dim 1 is the node dimension

    def get_preprocesser(self, batched: bool = False, verbose: bool = True):
        if self.preprocessing_type == 'duplication':
            preprocesser = self.duplicate_features
            if not self.use_level_features:
                self.out_dim = sum([self.input_dims[GLOBALS], self.input_dims[LAYERS]])
            else:
                self.out_dim = sum(
                    [self.input_dims[GLOBALS], self.input_dims[LAYERS], self.input_dims[LEVELS]]
                )
            if self.drop_node_encoding:
                self.out_dim -= 2 * self.node_encoding_len if not self.use_level_features else 3 * self.node_encoding_len
            self.as_string = 'Duplicate global features at all layers'

        elif self.preprocessing_type == 'padding':
            preprocesser = self.pad_features
            self.out_dim = max(*[var_indim for var_indim in self.input_dims.keys()])
            self.as_string = 'Padding all var types to have same #features'

        elif self.preprocessing_type in ['mlp', 'mlp_projection']:
            in_dims = self.input_dims.copy()
            if not self.use_level_features:
                in_dims.pop(LEVELS)
            preprocesser = FeatureProjector(
                input_name_to_feature_dim=in_dims,
                projector_n_layers=self.projector_n_layers,
                projection_dim=self.projector_hidden_dim,
                projector_activation_func='Gelu',
                projector_net_normalization=self.projector_net_normalization,
                output_normalization=True,
                output_activation_function=False,
                projections_aggregation=self.intersperse if self.use_level_features else self.intersperse_no_levels)
            self.out_dim = self.projector_hidden_dim
            self.as_string = f'All var types are MLP-projected to a {self.projector_hidden_dim} hidden dimension'
        elif self.preprocessing_type == 'graph_net_layer_nodes':
            if batched:
                preprocesser = gn_input_dict_renamer_layer_nodes_batched
            else:
                preprocesser = gn_input_dict_renamer_layer_nodes  # just rename the dict keys
            self.out_dim = self.input_dims = {NODES: get_dict_entry(self.input_dims, [LAYERS, NODES]),
                                              EDGES: get_dict_entry(self.input_dims, [LEVELS, EDGES]),
                                              GLOBALS: get_dict_entry(self.input_dims, [GLOBALS, GLOBALS])}
            self.as_string = 'No preprocessing'
        elif self.preprocessing_type == 'graph_net_level_nodes':
            if batched:
                preprocesser = gn_input_dict_renamer_level_nodes_batched
            else:
                preprocesser = gn_input_dict_renamer_level_nodes  # just rename the dict keys

            self.out_dim = self.input_dims = {NODES: get_dict_entry(self.input_dims, [LEVELS, NODES]),
                                              EDGES: get_dict_entry(self.input_dims, [LAYERS, EDGES]),
                                              GLOBALS: self.input_dims[GLOBALS]}
            self.as_string = 'No preprocessing'
        elif self.preprocessing_type == 'drop_levels':
            def gn_input_dict_renamer_no_levels(x: Dict[str, Tensor]):
                x[NODES] = x.pop(LAYERS)
                return x

            preprocesser = gn_input_dict_renamer_no_levels  # just rename the dict keys
            self.out_dim = {NODES: get_dict_entry(self.input_dims, [LAYERS, NODES]),
                            EDGES: 0, GLOBALS: self.input_dims[GLOBALS]}
            self.as_string = 'Dropping levels'
        elif self.preprocessing_type in [None, 'identity']:
            preprocesser = identity
            self.out_dim = self.input_dims
            self.as_string = 'No preprocessing'
        else:
            raise ValueError(f"Preprocessing type {self.preprocessing_type} not known.")
        if verbose and preprocesser != identity:
            s = f' {self.preprocessing_type} pre-processing'
            if 'No preprocessing' not in self.as_string:
                s += f': {self.as_string}'
            log.info(s)
        return preprocesser

    def get_adj(self, degree_normalized: bool = False, improved: bool = False) -> Tensor:
        """ Adjacency matrix of a line graph with global node and self-loops """
        adj = torch.zeros((self.n_nodes, self.n_nodes))

        if hasattr(self, 'GLOBAL_NODE'):
            adj[:, self.GLOBAL_NODE] = 1
            adj[self.GLOBAL_NODE, :] = 1

        for i in range(1, self.n_nodes):
            adj[i, i - 1:i + 2] = 1
            adj[i - 1:i + 2, i] = 1

        if degree_normalized:
            print("------------------------> DEGREE NORMED A", improved)
            return normalize_adjacency_matrix_torch(adj, improved=improved, add_self_loops=True)
        return adj

    def get_edge_idxs(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.preprocessing_type in ['graph_net_level_nodes', 'graph_net_layer_nodes']
        # one-way: node i has an edge to node i+1
        one_way_senders = np.arange(self.n_nodes - 1)  # n_lay - 1 = n_lev - 2 edges
        one_way_receivers = one_way_senders + 1
        # one-way: node i+1 has an edge to node i
        other_way_senders = np.arange(1, self.n_nodes)
        other_way_receivers = other_way_senders - 1

        senders = np.concatenate((one_way_senders, other_way_senders))
        receivers = np.concatenate((one_way_receivers, other_way_receivers))

        return senders, receivers

    def intersperse(self,
                    name_to_array: Optional[Dict[str, Tensor]] = None,
                    global_node: Optional[Tensor] = None,  # shape (b, #feats)
                    levels: Optional[Tensor] = None,  # shape (b, #levels, #feats)
                    layers: Optional[Tensor] = None  # shape (b, #layers, #feats)
                    ) -> Tensor:
        """
        Either name_to_array dict OR ALL OF global_node, levels, layers mustnt be None
        """
        global_node, levels, layers = self.get_data_types(name_to_array, global_node, levels, layers)

        if global_node.shape[-1] != layers.shape[-1] or levels.shape[-1] != layers.shape[-1]:
            raise ValueError("Expected all node types to have same dimensions. Project them first or pad them instead!")

        batch_size, _, n_feats = levels.shape

        interspersed_data = torch.empty((batch_size, self.n_nodes, n_feats))
        interspersed_data[:, self.GLOBAL_NODE, :] = global_node
        interspersed_data[:, self.LEVEL_NODES, :] = levels
        interspersed_data[:, self.LAYER_NODES, :] = layers

        interspersed_data = interspersed_data.to(global_node.device)
        return interspersed_data

    def intersperse_no_levels(self,
                              name_to_array: Optional[Dict[str, Tensor]] = None,
                              global_node: Optional[Tensor] = None,  # shape (b, #feats)
                              layers: Optional[Tensor] = None, **kwargs  # shape (b, #layers, #feats)
                              ) -> Tensor:
        """
        Either name_to_array dict OR ALL OF global_node, layers mustnt be None
        """
        global_node, _, layers = self.get_data_types(name_to_array, global_node, None, layers)

        if global_node.shape[-1] != layers.shape[-1]:
            raise ValueError("Expected all node types to have same dimensions. Project them first or pad them instead!")

        batch_size, _, n_feats = layers.shape
        interspersed_data = torch.empty((batch_size, self.n_nodes, n_feats))
        interspersed_data[:, self.GLOBAL_NODE, :] = global_node
        interspersed_data[:, self.LAYER_NODES, :] = layers
        interspersed_data = interspersed_data.to(global_node.device)
        return interspersed_data

    def pad_features(self,
                     name_to_array: Optional[Dict[str, Tensor]] = None,
                     global_node: Optional[Tensor] = None,  # shape (b, #feats)
                     levels: Optional[Tensor] = None,  # shape (b, #levels, #feats)
                     layers: Optional[Tensor] = None,  # shape (b, #layers, #feats)
                     padding: float = 0.0
                     ) -> Tensor:
        """
        Either name_to_array dict OR ALL OF global_node, levels, layers mustnt be None
        """
        global_node, levels, layers = self.get_data_types(name_to_array, global_node, levels, layers)

        data_size = global_node.shape[0]
        n_global_feats, n_level_feats, n_layer_feats = global_node.shape[-1], levels.shape[-1], layers.shape[-1]
        max_features = max(n_global_feats, n_level_feats, n_layer_feats)

        padded_data = torch.ones((data_size, self.n_nodes, max_features))
        padded_data *= padding  # set all values to padding by default
        padded_data[:, self.GLOBAL_NODE, :n_global_feats] = global_node
        padded_data[:, self.LEVEL_NODES, :n_level_feats] = levels
        padded_data[:, self.LAYER_NODES, :n_layer_feats] = layers

        padded_data = padded_data.to(global_node.device)
        return padded_data

    def duplicate_features(
            self,
            name_to_array: Optional[Dict[str, Tensor]] = None,
            global_node: Optional[Tensor] = None,  # shape (b, #feats)
            levels: Optional[Tensor] = None,  # shape (b, #levels, #feats)
            layers: Optional[Tensor] = None,  # shape (b, #layers, #feats)
    ) -> Tensor:
        """ Duplicate global (and optionally level) features across all layers

        level_policy: If 'drop', level features are simply dropped,
                      If 'concat', all levels but one are concatenated to its adjacent layer
        drop_last_level: Only used if level_policy is concat, i.e. all levels but one are concatenated to the layers.

        Returns:
            A (b, #layers, d) array/Tensor,
                where d = #layer-feats + #global-feats (+ #levels-feats, if not drop_levels)
        """
        global_node, levels, layers = self.get_data_types(name_to_array, global_node, levels, layers)

        if self.drop_node_encoding:
            global_node = global_node[:, :-self.node_encoding_len]
            layers = layers[:, :, :-self.node_encoding_len]
            levels = levels[:, :, :-self.node_encoding_len]

        data_size, n_layers, n_layer_feats = layers.shape
        n_global_feats, n_level_feats = global_node.shape[-1], levels.shape[-1]
        n_layers_feats_with_duplicates = n_layer_feats + n_global_feats
        # print(data_size, n_layers, n_layer_feats, n_global_feats, n_level_feats, n_layers_feats_with_duplicates)
        if self.use_level_features:
            n_layers_feats_with_duplicates += n_level_feats

        all_data = torch.empty((data_size, n_layers, n_layers_feats_with_duplicates))
        all_data[:, :, :n_layer_feats] = layers
        all_data[:, :, n_layer_feats:n_layer_feats + n_global_feats] = global_node[:, None, :]
        if self.use_level_features:
            # Will concat level features to adjacent layers, but will drop information from one (the first/last) level!
            for i in range(n_layers):
                level_i = i if self.drop_last_level else i + 1
                all_data[:, i, n_layer_feats + n_global_feats:] = levels[:, level_i, :]

        all_data = all_data.to(global_node.device)
        return all_data

    def get_data_types(
            self,
            name_to_array: Optional[Dict[str, Tensor]] = None,
            global_node: Optional[Tensor] = None,  # shape (b, #feats)
            levels: Optional[Tensor] = None,  # shape (b, #levels, #feats)
            layers: Optional[Tensor] = None,  # shape (b, #layers, #feats)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if name_to_array is not None:
            global_node = name_to_array[GLOBALS]
            levels = name_to_array[LEVELS] if LEVELS in name_to_array.keys() else None
            layers = name_to_array[LAYERS]

        return global_node, levels, layers

    def get_output_mask(self) -> Tensor:
        # GCN output shape out = (n, self.n_nodes, gcn-out-dim)
        # in GCN gcn-out-dim = 2
        # out[:, get_output_mask(), :] has shape (n, len(self.LEVEL_NODES), gcn-out-dim)
        if not hasattr(self, 'LEVEL_NODES') or self.preprocessing_type in self.ONLY_LEVEL_NODES:
            return torch.ones(self.n_nodes).bool()
        else:
            level_mask = torch.zeros(self.n_nodes)
            level_mask[self.LEVEL_NODES] = 1
        return level_mask.bool()

    def __str__(self):
        return self.as_string


def gn_input_dict_renamer_layer_nodes(x: Dict[str, Tensor]):
    x[NODES] = x.pop(LAYERS)
    x[EDGES] = x.pop(LEVELS)[1:-1, :]  # remove the surface and toa level
    x[EDGES] = repeat(x[EDGES], "e d -> (repeat e) d", repeat=2)  # bidirectional edges
    # x[EDGES] = repeat(x[EDGES], "b e d -> b (repeat e) d", repeat=2)  # bidirectional edges
    return x


def gn_input_dict_renamer_layer_nodes_batched(x: Dict[str, Tensor]):
    x[NODES] = x.pop(LAYERS)
    x[EDGES] = x.pop(LEVELS)[:, 1:-1, :]  # remove the surface and toa level
    x[EDGES] = repeat(x[EDGES], "b e d -> b (repeat e) d", repeat=2)  # bidirectional edges
    return x


def gn_input_dict_renamer_level_nodes(x: Dict[str, Tensor]):
    x[NODES] = x.pop(LEVELS)
    x[EDGES] = x.pop(LAYERS)
    x[EDGES] = repeat(x[EDGES], "e d -> (repeat e) d", repeat=2)  # bidirectional edges
    return x


def gn_input_dict_renamer_level_nodes_batched(x: Dict[str, Tensor]):
    x[NODES] = x.pop(LEVELS)
    x[EDGES] = x.pop(LAYERS)
    # x[GLOBALS] = x.pop(GLOBALS)
    x[EDGES] = repeat(x[EDGES], "b e d -> b (repeat e) d", repeat=2)  # bidirectional edges
    return x
