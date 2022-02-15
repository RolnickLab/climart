from abc import ABC
from typing import Dict, Any, Tuple, Union, Optional

import einops
import numpy as np
import torch
from torch import Tensor

from climart.data_loading.constants import LEVELS, LAYERS, GLOBALS, get_data_dims, INPUT_TYPES
from climart.models.GraphNet.constants import NODES, EDGES
from climart.models.modules.additional_layers import FeatureProjector
from climart.utils.utils import get_logger, normalize_adjacency_matrix_torch


class AbstractTransform(ABC):
    def __init__(self, exp_type: str):
        input_output_dimensions = get_data_dims(exp_type=exp_type)
        self.spatial_input_dim: Dict[str, int] = input_output_dimensions['spatial_dim']
        self.input_dim: Dict[str, int] = input_output_dimensions['input_dim']
        self.log = get_logger(__name__)

    @property
    def output_dim(self) -> Union[int, Dict[str, int]]:
        """
        Returns:
            The number of feature dimensions that the transformed data will have.
            If the transform returns an array, output_dim should be an int.
            If the transform returns a dict of str -> array, output_dim should be a dict str -> int, that
                described the number of features for each key in the transformed output.

        """
        return self._out_dim

    def transform(self, X: Dict[str, np.ndarray]) -> Any:
        """
        How to transform dict
            X = {
                'layer': layer_array,   # shape (#layers, #layer-features)
                'levels': levels_array, # shape (#levels, #level-features)
                'globals': globals_array (#global-features,)
                }
        to the form the model will use/receive it in forward.
        Implementation will be applied (with multi-processing) in the _get_item(.) method of the dataset
            --> IMPORTANT: the arrays in X will *not* have the batch dimension!
        """
        raise NotImplementedError

    def batched_transform(self, X: Dict[str, np.ndarray]) -> Any:
        """
        How to transform dict
            X = {
                'layer': layer_array,   # shape (batch-size, #layers, #layer-features)
                'levels': levels_array, # shape (batch-size, #levels, #level-features)
                'globals': globals_array (batch-size, #global-features,)
                }
        to the form the model will use/receive it in forward.
        """
        raise NotImplementedError


class IdentityTransform(AbstractTransform):
    def __init__(self, exp_type: str):
        super().__init__(exp_type)
        self._out_dim = self.input_dim

    def transform(self, X_not_batched: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return X_not_batched

    def batched_transform(self, X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return X


class FlattenTransform(AbstractTransform):
    """ Flattens a dict with array's as values into a 1D vector (or 2D with batch dimension)"""

    def __init__(self, exp_type: str):
        super().__init__(exp_type)
        self._out_dim = sum(self.input_dim.values())

    def transform(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([X[key].flatten() for key in INPUT_TYPES], axis=0)
        # return np.concatenate([torch.flatten(subX) for subX in X.values()], dim=0)

    def batched_transform(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([X[key].reshape((X[key].shape[0], -1)) for key in INPUT_TYPES], axis=1)
        # return torch.cat([torch.flatten(subX, start_dim=1).unsqueeze(1) for subX in X.values()], dim=1)


class RepeatGlobalsTransform(AbstractTransform):
    """
    c - spatial dimension
    f - feature dimension
    b -  batch dimension
    """
    def __init__(self, exp_type: str):
        super().__init__(exp_type)
        self._out_dim = sum(self.input_dim.values())

    def transform(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        X_levels = X[LEVELS]
        # X_layers = rearrange(F.pad(rearrange(X['layers'], 'c f -> () c f'), (0,0,1,0), mode='reflect'), '() c f ->
        # c f')
        npad = ((0, 1), (0, 0))
        X_layers = np.pad(X[LAYERS], pad_width=npad, mode='reflect')
        X_global = einops.repeat(X[GLOBALS], 'f -> c f', c=50)
        X = np.concatenate([X_levels, X_layers, X_global], axis=-1)
        return einops.rearrange(X, 'c f -> f c')

    def batched_transform(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        X_levels = batch[LEVELS]
        X_global = einops.repeat(batch[GLOBALS], 'b f -> b c f', c=50)
        X_layers = einops.rearrange(batch[LAYERS], 'b c f -> ()b c f')
        X_layers = np.pad(X_layers, pad_width=((0, 0), (0, 0), (0, 1), (0, 0)), mode='reflect')
        # X_layers = np.pad(X_layers, pad_width=(0, 0, 1, 0), mode='reflect')
        X_layers = einops.rearrange(X_layers, '()b c f -> b c f')

        X = np.concatenate([X_levels, X_layers, X_global], axis=-1)
        return einops.rearrange(X, 'b c f -> b f c')


# -------------------------------------------- Graph specific transforms
class AbstractGraphTransform(AbstractTransform, ABC):
    def __init__(self, exp_type: str):
        super().__init__(exp_type)

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    def get_edge_idxs(self) -> Tuple[np.ndarray, np.ndarray]:
        # one-way: node i has an edge to node i+1
        one_way_senders = np.arange(self.n_nodes - 1)  # n_lay - 1 = n_lev - 2 edges
        one_way_receivers = one_way_senders + 1
        # one-way: node i+1 has an edge to node i
        other_way_senders = np.arange(1, self.n_nodes)
        other_way_receivers = other_way_senders - 1

        senders = np.concatenate((one_way_senders, other_way_senders))
        receivers = np.concatenate((one_way_receivers, other_way_receivers))
        return senders, receivers

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
            self.log.info("-> Adjacency matrix is normalized by in-degree")
            return normalize_adjacency_matrix_torch(adj, improved=improved, add_self_loops=True)
        return adj

    def get_level_nodes_mask(self) -> Tensor:
        """
        Indexing an array 'A' with shape (n, self.n_nodes, out-dim)
        leads to A[:, get_level_nodes_mask(), :] have shape (n, len(self.LEVEL_NODES), out-dim)
        :return: A torch.Tensor mask that indexes all the level nodes
        """

        if not hasattr(self, 'LEVEL_NODES'):
            raise ValueError(f"This transform {self} does not have level nodes.")
        else:
            level_mask = torch.zeros(self.n_nodes)
            level_mask[self.LEVEL_NODES] = 1
        return level_mask.bool()


class OnlyLayersAreNodesTransforms(AbstractGraphTransform, ABC):
    def __init__(self, exp_type: str):
        super().__init__(exp_type)
        self._n_nodes = self.spatial_input_dim[LAYERS]
        self.LAYER_NODES = slice(0, self.n_nodes)  # all nodes are layer nodes


class OnlyLevelsAreNodesTransforms(AbstractGraphTransform, ABC):
    def __init__(self, exp_type: str):
        super().__init__(exp_type)
        self._n_nodes = self.spatial_input_dim[LEVELS]
        self.LEVEL_NODES = slice(0, self.n_nodes)  # all nodes are level nodes


class AllAreNodesTransforms(AbstractGraphTransform, ABC):
    def __init__(self, exp_type: str, use_level_features: bool = True):
        super().__init__(exp_type)
        self.use_level_features = use_level_features
        if self.use_level_features:
            self._n_nodes = self.spatial_input_dim[LAYERS] + self.spatial_input_dim[LEVELS] + 1
            self.LEVEL_NODES = slice(1, self.n_nodes, 2)  # start at 1, then every second, [1, 3, 5, 7,...]
            self.LAYER_NODES = slice(2, self.n_nodes, 2)  # [2, 4, 6,...]
        else:
            self._n_nodes = self.spatial_input_dim[LAYERS] + 1
            self.LAYER_NODES = slice(1, self.n_nodes)  # [1, 2, 3,...]        self.padding_value = padding_value
        self.GLOBAL_NODE = 0


class EdgesAndNodesTransforms(AbstractGraphTransform, ABC):
    def __init__(self, exp_type: str):
        super().__init__(exp_type)

    @property
    def n_edges(self) -> int:
        return self._n_edges


class DuplicateFeaturesTransform(OnlyLayersAreNodesTransforms):
    """
    Duplicate global features at all layers
    """

    def __init__(self,
                 level_policy: str = "drop",
                 drop_last_level: bool = True,
                 use_level_features: bool = True,
                 drop_node_encoding: bool = True,
                 node_encoding_len: int = 3,
                 *args, **kwargs
                 ):
        """
        Duplicate global (and optionally level) features across all layers

        Args:
            level_policy: If 'drop', level features are simply dropped,
                          If 'concat', all levels but one are concatenated to its adjacent layer
            drop_last_level: Only used if level_policy is concat, i.e. all levels but one are concatenated to the layers.
        """
        super().__init__(*args, **kwargs)
        self.level_policy = level_policy
        self.drop_last_level = drop_last_level
        self.use_level_features = use_level_features
        self.drop_node_encoding = drop_node_encoding
        self.node_encoding_len = node_encoding_len

        if self.use_level_features:
            self._out_dim = sum([self.input_dim[GLOBALS], self.input_dim[LAYERS], self.input_dim[LEVELS]])
        else:
            self.log.info(' Dropping level features!')
            self._out_dim = sum([self.input_dim[GLOBALS], self.input_dim[LAYERS]])

        if self.drop_node_encoding:
            self._out_dim -= 2 * self.node_encoding_len if not self.use_level_features else 3 * self.node_encoding_len

    def transform(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Returns:
            A (b, #layers, d) array, where d = #layer-feats + #global-feats (+ #levels-feats, if not drop_levels)
        """
        global_node, levels, layers = X[GLOBALS], X[LEVELS], X[LAYERS]
        if self.drop_node_encoding:
            global_node = global_node[:, :-self.node_encoding_len]
            layers = layers[:, :, :-self.node_encoding_len]
            levels = levels[:, :, :-self.node_encoding_len]
        data_size, n_layers, n_layer_feats = layers.shape
        n_global_feats, n_level_feats = global_node.shape[-1], levels.shape[-1]
        n_layers_feats_with_globals = n_layer_feats + n_global_feats

        if self.use_level_features:
            n_layers_feats_with_globals += n_level_feats

        all_data = np.zeros((data_size, n_layers, n_layers_feats_with_globals))
        all_data[:, :, :n_layer_feats] = layers
        all_data[:, :, n_layer_feats:n_layer_feats + n_global_feats] = global_node[:, None, :]
        if self.use_level_features:
            # Will concat level features to adjacent layers, but will drop information from one (the first/last) level!
            for i in range(n_layers):
                level_i = i if self.drop_last_level else i + 1
                all_data[:, i, n_layer_feats + n_global_feats:] = levels[:, level_i, :]
        return all_data

    def batched_transform(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        return self.transform(X)


class PadTransform(AllAreNodesTransforms):
    """
    Padding all var types to have same #features
    """

    def __init__(self, exp_type: str, use_level_features: bool = True, padding_value: float = 0.0):
        super().__init__(exp_type, use_level_features=use_level_features)
        self._out_dim = max(*[var_indim for var_indim in self.input_dim.values()])
        self.padding_value = padding_value

    def transform(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        global_node, levels, layers = X[GLOBALS], X[LEVELS], X[LAYERS]
        # set all values to padding by default
        padded_data = self.padding_value * np.ones((self.n_nodes, self.output_dim))
        padded_data[self.GLOBAL_NODE, :self.input_dim[GLOBALS]] = global_node
        padded_data[self.LAYER_NODES, :self.input_dim[LAYERS]] = layers
        if self.use_level_features:
            padded_data[self.LEVEL_NODES, :self.input_dim[LEVELS]] = levels
        return padded_data

    def batched_transform(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        global_node, levels, layers = X[GLOBALS], X[LEVELS], X[LAYERS]
        data_size = global_node.shape[0]
        # set all values to padding by default
        padded_data = self.padding_value * np.ones((data_size, self.n_nodes, self.output_dim))
        padded_data[:, self.GLOBAL_NODE, :self.input_dim[GLOBALS]] = global_node
        padded_data[:, self.LAYER_NODES, :self.input_dim[LAYERS]] = layers
        if self.use_level_features:
            padded_data[:, self.LEVEL_NODES, :self.input_dim[LEVELS]] = levels
        return padded_data


class LayerEdgesAndLevelNodesGraph(EdgesAndNodesTransforms, OnlyLevelsAreNodesTransforms):
    """
    graph_net_level_nodes: gn_input_dict_renamer_level_nodes
    """

    def __init__(self, exp_type: str):
        super().__init__(exp_type)
        self._n_edges = self.spatial_input_dim[LAYERS] * 2  # bi-directional
        self._out_dim = {NODES: self.input_dim[LEVELS], EDGES: self.input_dim[LAYERS], GLOBALS: self.input_dim[GLOBALS]}

    def transform(self, X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        X[NODES] = X.pop(LEVELS)
        X[EDGES] = X.pop(LAYERS)
        X[EDGES] = einops.repeat(X[EDGES], "e d -> (repeat e) d", repeat=2)  # bidirectional edges
        return X

    def batched_transform(self, X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        X[NODES] = X.pop(LEVELS)
        X[EDGES] = X.pop(LAYERS)  # x[GLOBALS] = x.pop(GLOBALS)
        X[EDGES] = einops.repeat(X[EDGES], "b e d -> b (repeat e) d", repeat=2)  # bidirectional edges
        return X


class LevelEdgesAndLayerNodesGraph(EdgesAndNodesTransforms, OnlyLayersAreNodesTransforms):
    """
    graph_net_layer_nodes: gn_input_dict_renamer_layer_nodes
    """

    def __init__(self, exp_type: str):
        super().__init__(exp_type)
        self._n_edges = self.spatial_input_dim[LEVELS] * 2  # bi-directional
        self._out_dim = {NODES: self.input_dim[LAYERS], EDGES: self.input_dim[LEVELS], GLOBALS: self.input_dim[GLOBALS]}

    def transform(self, x: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        x[NODES] = x.pop(LAYERS)
        x[EDGES] = x.pop(LEVELS)[1:-1, :]  # remove the surface and toa level
        x[EDGES] = einops.repeat(x[EDGES], "e d -> (repeat e) d", repeat=2)  # bidirectional edges
        return x

    def batched_transform(self, x: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        x[NODES] = x.pop(LAYERS)
        x[EDGES] = x.pop(LEVELS)[:, 1:-1, :]  # remove the surface and toa level
        x[EDGES] = einops.repeat(x[EDGES], "b e d -> b (repeat e) d", repeat=2)  # bidirectional edges
        return x


class ColumnPreprocesser:
    ONLY_LAYER_NODES = ['duplication', 'graph_net_layer_nodes', 'identity']
    ONLY_LEVEL_NODES = ['graph_net_level_nodes']

    def __init__(self,
                 preprocessing: str,
                 exp_type: str,
                 projector_hidden_dim: int = 128,  # only if preprocessing == 'mlp'
                 projector_n_layers: int = 1,  # only if preprocessing == 'mlp'
                 projector_net_normalization: str = 'layer_norm',  # only if preprocessing == 'mlp'
                 use_level_features: bool = True,
                 ):
        self.preprocessing_type = preprocessing.lower()
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_n_layers = projector_n_layers
        self.projector_net_normalization = projector_net_normalization
        self.use_level_features = use_level_features

    def get_preprocesser(self, batched: bool = False, verbose: bool = True):
        if self.preprocessing_type in ['mlp', 'mlp_projection']:
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

        return preprocesser

    def intersperse(self,
                    name_to_array: Optional[Dict[str, Tensor]] = None,
                    global_node: Optional[Tensor] = None,  # shape (b, #feats)
                    levels: Optional[Tensor] = None,  # shape (b, #levels, #feats)
                    layers: Optional[Tensor] = None  # shape (b, #layers, #feats)
                    ) -> Tensor:
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
