from typing import Union
from math import pi as PI
import torch
from graphein.protein.tensor.types import EdgeTensor
from torch.nn import Linear, Sequential
from torch_geometric.typing import OptPairTensor
from torch_geometric.nn import MessagePassing


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
        **kwargs
    ):
        """Convolutional layer implemented in SchNet that supports bipartite inputs (i.e., tuple of (x, x)).

        Args:
            in_channels (int): input dimension of tensors (i.e., d in (n, d))
            out_channels (int): output dimension of tensors (i.e., d in (n, d))
            num_filters (int): number of Gaussian filters to use
            nn (Sequential): MLP used
            cutoff (float): cutoff distance for interatomic interactions.
        """
        super().__init__(aggr="add", **kwargs)
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(
        self,
        x: Union[torch.Tensor, OptPairTensor],
        edge_index: EdgeTensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x = (x, x)

        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = (self.lin1(x[0]), x[1])
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return x_j * W
