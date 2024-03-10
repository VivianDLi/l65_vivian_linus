from typing import Union
import torch
import torch_scatter
from graphein.protein.tensor.types import CoordTensor, EdgeTensor
from torch.nn import Linear, Dropout, Sequential
from torch_geometric.typing import OptPairTensor
from torch_geometric.nn import MessagePassing

from proteinworkshop.models.utils import get_activations
from proteinworkshop.types import NodeFeatureTensor


class EGNN(MessagePassing):
    def __init__(
        self,
        emb_dim: int,
        activation: str = "relu",
        norm: str = "layer",
        aggr: str = "mean",
        dropout: float = 0.1,
        **kwargs,
    ):
        """E(n) Equivariant GNN Layer support bipartite graphs.

        Paper: E(n) Equivariant Graph Neural Networks, Satorras et al.

        Args:
            emb_dim: (int) - hidden dimension `d`
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            dropout: (float) - dropout rate
        """
        # Set the aggregation function
        super().__init__(aggr=aggr, **kwargs)

        self.emb_dim = emb_dim
        self.activation = get_activations(activation)
        self.norm = {
            "layer": torch.nn.LayerNorm,
            "batch": torch.nn.BatchNorm1d,
        }[norm]

        # MLP `\psi_h` for computing messages `m_ij`
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + 1, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Dropout(dropout),
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Dropout(dropout),
        )
        # MLP `\psi_x` for computing messages `\overrightarrow{m}_ij`
        self.mlp_pos = Sequential(
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Dropout(dropout),
            Linear(emb_dim, 1),
        )
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Dropout(dropout),
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Dropout(dropout),
        )

    def forward(
        self,
        x: Union[torch.Tensor, OptPairTensor],
        edge_index: EdgeTensor,
        pos: Union[torch.Tensor, OptPairTensor],
    ):
        """
        Args:
            x: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
        Returns:
            out: [(n, d),(n,3)] - updated node features
        """
        if isinstance(x, torch.Tensor):
            x = (x, x)
        if isinstance(pos, torch.Tensor):
            pos = (pos, pos)

        msg_aggr, pos_aggr = self.propagate(edge_index, x=x, pos=pos)
        msg_aggr = self.mlp_upd(torch.cat([x[1], msg_aggr], dim=-1))
        return msg_aggr, pos_aggr

    def message(self, x_i, x_j, pos_i, pos_j):
        # Compute messages
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1, keepdim=True)
        msg = torch.cat([x_i, x_j, dists], dim=-1)
        msg = self.mlp_msg(msg)
        # Scale magnitude of displacement vector
        pos_diff = pos_diff / (dists + 1) * self.mlp_pos(msg)
        return msg, pos_diff

    def aggregate(self, inputs, index):
        msgs, pos_diffs = inputs
        # Aggregate messages
        msg_aggr = torch_scatter.scatter(
            msgs, index, dim=self.node_dim, reduce=self.aggr
        )
        # Aggregate displacement vectors
        pos_aggr = torch_scatter.scatter(
            pos_diffs, index, dim=self.node_dim, reduce="mean"
        )
        return msg_aggr, pos_aggr

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"
