from typing import Any, Dict, List, Mapping, Optional, Set, Union

import torch
import torch.nn as nn
import torch_scatter
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from torch_geometric.data import Batch
from torch_geometric.nn.conv import HeteroConv
from torch_geometric.nn import Sequential, Linear
from torch_geometric.nn.models import SchNet
from torch_geometric.nn.models.schnet import (
    CFConv,
    ShiftedSoftplus,
)

from proteinworkshop.models.graph_encoders.gnn import get_gnn_layer
from proteinworkshop.models.utils import get_activations, get_aggregation
from proteinworkshop.types import EncoderOutput

import logging


class VirtualSchNet(SchNet):
    def __init__(
        self,
        edge_list: List[str],
        hidden_channels: int = 128,
        out_dim: int = 1,
        num_filters: int = 128,
        num_layers: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10,
        max_num_neighbors: int = 32,
        readout: str = "add",
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: Optional[torch.Tensor] = None,
    ):
        """Initializes an instance of the SchNetModel class with the provided
        parameters and virtual nodes.

        :param hidden_channels: Number of channels in the hidden layers
            (default: ``128``)
        :type hidden_channels: int
        :param out_dim: Output dimension of the model (default: ``1``)
        :type out_dim: int
        :param num_filters: Number of filters used in convolutional layers
            (default: ``128``)
        :type num_filters: int
        :param num_layers: Number of convolutional layers in the model
            (default: ``6``)
        :type num_layers: int
        :param num_gaussians: Number of Gaussian functions used for radial
            filters (default: ``50``)
        :type num_gaussians: int
        :param cutoff: Cutoff distance for interactions (default: ``10``)
        :type cutoff: float
        :param max_num_neighbors: Maximum number of neighboring atoms to
            consider (default: ``32``)
        :type max_num_neighbors: int
        :param readout: Global pooling method to be used (default: ``"add"``)
        :type readout: str
        """
        super().__init__(
            hidden_channels,
            num_filters,
            num_layers,
            num_gaussians,
            cutoff,  # None, # Interaction graph is not used
            max_num_neighbors,
            readout,
            dipole,
            mean,
            std,
            atomref,
        )
        self.edge_list = edge_list

        self.readout = readout
        # Overwrite embbeding
        self.embedding = torch.nn.LazyLinear(hidden_channels)
        # Overwrite atom embedding and final predictor
        self.lin2 = torch.nn.LazyLinear(out_dim)

        # Replace interaction layers with Hetero versions
        self.interactions = nn.ModuleList()
        for _ in range(num_layers):
            block = HeteroInteractionBlock(
                edge_list, hidden_channels, num_gaussians, num_filters, cutoff
            )
            self.interactions.append(block)

    @property
    def required_batch_attributes(self) -> Set[str]:
        """Required batch attributes for this encoder.

        - ``x_dict``: Dictionary of node features (shape: :math:`(n, d)`) for each edge type
        - ``pos_dict``: Dictionary of node positions (shape: :math:`(n, 3)`) for each edge type
        - ``edge_index_dict``: Dictionary of edge indices (shape: :math:`(2, e)`) for each edge type
        - ``batch_dict``: Dictionary of batch indices (shape: :math:`(n,)`) for each edge type

        :return: Set of required batch attributes
        :rtype: Set[str]
        """
        return {"pos_dict", "edge_index_dict", "x_dict", "batch_dict"}

    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the GNN encoder.

        Returns the node embedding and graph embedding in a dictionary.

        :param batch: Batch of data to encode.
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings. Contains
            ``node_embedding`` and ``graph_embedding`` fields. The node
            embedding is of shape :math:`(|V|, d)` and the graph embedding is
            of shape :math:`(n, d)`, where :math:`|V|` is the number of nodes
            and :math:`n` is the number of graphs in the batch and :math:`d` is
            the dimension of the embeddings.
        :rtype: EncoderOutput
        """
        h_dict = {
            edge_name: self.embedding(batch.x_dict[edge_name])
            for edge_name in batch.x_dict
        }

        u_dict = {
            edge_name: batch.edge_index_dict[edge_name][0]
            for edge_name in batch.edge_index_dict
        }
        v_dict = {
            edge_name: batch.edge_index_dict[edge_name][1]
            for edge_name in batch.edge_input_dict
        }
        edge_weight_dict = {
            edge_name: (
                batch.pos_dict[edge_name][u_dict[edge_name]]
                - batch.pos_dict[edge_name][v_dict[edge_name]]
            ).norm(dim=-1)
            for edge_name in batch.pos_dict
        }
        edge_attr_dict = {
            edge_name: self.distance_expansion(edge_weight_dict[edge_name])
            for edge_name in batch.edge_weight_dict
        }

        for interaction in self.interactions:
            h_update_dict = interaction(
                h_dict, batch.edge_index_dict, edge_weight_dict, edge_attr_dict
            )
            h_dict = {
                edge_name: h_dict[edge_name] + h_update_dict[edge_name]
                for edge_name in h_update_dict
            }

        h_dict = {
            edge_name: self.lin1(h_dict[edge_name]) for edge_name in h_dict
        }
        h_dict = {
            edge_name: self.act(h_dict[edge_name]) for edge_name in h_dict
        }
        h_dict = {
            edge_name: self.lin2(h_dict[edge_name]) for edge_name in h_dict
        }

        return EncoderOutput(
            {
                "node_embedding": h_dict["real"],
                "graph_embedding": torch_scatter.scatter(
                    h_dict["real"],
                    batch.batch["real"],
                    dim=0,
                    reduce=self.readout,
                ),
            }
        )


class HeteroInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        edge_list: List[str],
        hidden_channels: int,
        num_gaussians: int,
        num_filters: int,
        cutoff: float,
    ):
        super().__init__()
        self.edge_list = edge_list

        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )

        conv_dict = {
            edge_name: CFConv(
                hidden_channels, hidden_channels, num_filters, self.mlp, cutoff
            )
            for edge_name in edge_list
        }
        self.conv = HeteroConv(conv_dict, aggr="sum")
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(
        self,
        x_dict: Dict[str, nn.Tensor],
        edge_index_dict: Dict[str, nn.Tensor],
        edge_weight_dict: Dict[str, nn.Tensor],
        edge_attr_dict: Dict[str, nn.Tensor],
    ) -> Dict[str, nn.Tensor]:
        x_dict = self.conv(
            x_dict,
            edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            edge_attr_dict=edge_attr_dict,
        )
        x_dict = {
            edge_name: self.act(x_dict[edge_name])
            for edge_name in self.edge_list
        }
        x_dict = {
            edge_name: self.lin(x_dict[edge_name])
            for edge_name in self.edge_list
        }
        return x_dict
