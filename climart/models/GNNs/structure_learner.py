"""
Author: Salva Rühling Cachay
"""

import torch
import torch.nn as nn


class EdgeStructureLearner(nn.Module):
    def __init__(self, num_nodes, max_num_edges: int, dim: int, static_feat, alpha1=0.1, alpha2=2.0,
                 self_loops=True):
        super().__init__()
        self.num_nodes = num_nodes
        if static_feat is None:
            self.emb1 = nn.Embedding(num_nodes, dim)
            self.emb2 = nn.Embedding(num_nodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)
            self.static_feat = None
        else:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)

            self.static_feat = static_feat if torch.is_tensor(static_feat) else torch.from_numpy(static_feat)
            self.register_buffer('static_feat', self.static_feat.float())

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.num_edges = int(max_num_edges)
        self.self_loops = self_loops
        self.diag = torch.eye(self.num_nodes).bool()

    def forward(self):
        if self.static_feat is None:
            all_n = torch.arange(self.num_nodes).long().to(self.lin1.weight.device)
            nodevec1 = self.emb1(all_n)
            nodevec2 = self.emb2(all_n)
        else:
            nodevec1 = nodevec2 = self.static_feat

        nodevec1 = torch.tanh(self.alpha1 * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha1 * self.lin2(nodevec2))

        adj = torch.sigmoid(self.alpha2 * nodevec1 @ nodevec2.T)
        adj = adj.flatten()
        mask = torch.zeros(self.num_nodes * self.num_nodes).to(adj.device)
        _, strongest_idxs = torch.topk(adj, self.num_edges, sorted=False)  # Adj to get the strongest weight value indices
        mask[strongest_idxs] = 1
        adj = adj * mask
        adj = adj.reshape((self.num_nodes, self.num_nodes))
        self.diag = self.diag.to(adj.device)
        if self.self_loops:
            adj[self.diag] = adj[self.diag].clamp(min=0.5)

        return adj
