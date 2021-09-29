import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from climart.utils.utils import get_normalization_layer



class GraphConvolution(nn.Module):
    """
    This GCN layer was adapted from the PyTorch version by T. Kipf, see README.
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features,
                 out_features,
                 pyg_builtin: bool = False,
                 pyg_normalize_adj: bool = True,
                 improved_self_loops: bool = False,
                 residual: bool = False,
                 net_norm: str = 'none',
                 activation=F.relu,
                 dropout=0,
                 bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pyg_builtin = pyg_builtin
        if self.pyg_builtin:
            from torch_geometric.nn import GCNConv
            self.pyg_conv = GCNConv(self.in_features, self.out_features, cached=False,
                                    add_self_loops=True, improved=improved_self_loops,
                                    normalize=pyg_normalize_adj, bias=bias)
        else:
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

        self.residual = residual

        if self.in_features != self.out_features:
            self.residual = False

        self.net_norm = get_normalization_layer(net_norm, self.out_features)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, edge_index):
        """ Either adj or edge_index must be provided"""
        # AXW
        if self.pyg_builtin:
            node_repr = self.pyg_conv(input, edge_index=edge_index)
        else:
            support = torch.matmul(input, self.weight)  # XW matmul --> (batch-size, #nodes, #out-dim)
            node_repr = sparse_matmul(adj, support)  # (batch-size, #nodes, #out-dim)
            if self.bias is not None:
                node_repr = node_repr + self.bias

        # Post convolution
        if self.net_norm is None:
            pass
        elif self.net_norm in ['batch_norm', 'instance_norm']:
            node_repr = node_repr.transpose(1, 2)  # --> (batch-size, #out-dim, #nodes)
            node_repr = self.net_norm(node_repr)  # batch normalization over feature/node embedding dim.
            node_repr = node_repr.transpose(1, 2)
        elif self.net_norm == 'layer_norm':
            node_repr = self.net_norm(node_repr)  # batch normalization over feature/node embedding dim.

        if self.activation is not None:
            node_repr = self.activation(node_repr)

        if self.residual:
            node_repr = input + node_repr  # residual connection

        node_repr = self.dropout(node_repr)
        return node_repr

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def sparse_matmul(matrix, matrix_batch):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return torch.sparse.mm(matrix, vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)
