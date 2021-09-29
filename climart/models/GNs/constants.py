from enum import Enum
from typing import Dict

from torch import Tensor


GLOBALS = "globals"
NODES = "nodes"
EDGES = "edges"
GRAPH_COMPONENTS = [GLOBALS, NODES, EDGES]

GraphComponentToTensor = Dict[str, Tensor]


class AggregationTypes(Enum):
    AGG_E_TO_N = "edge_to_node_aggregation"
    AGG_E_TO_U = "edge_to_global_aggregation"
    AGG_N_TO_U = "node_to_global_aggregation"
    AGGREGATORS = [AGG_E_TO_N, AGG_E_TO_U, AGG_N_TO_U]


N_NODES = "n_nodes"
N_EDGES = "n_edges"
SPATIAL_DIM = 1
