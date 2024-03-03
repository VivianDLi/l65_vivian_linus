from typing import Any, Mapping, Set, Tuple, Union

import torch.nn as nn
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from torch_geometric.data import Batch
from torch_geometric.nn.conv import MessagePassing, HeteroConv
from torch_geometric.nn import GCNConv

from proteinworkshop.models.graph_encoders.layers.egnn import EGNNLayer
from proteinworkshop.models.graph_encoders.layers.gvp import GVPConv
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput

import logging


def get_gnn_layer(
    layer_name: str,
    emb_dim: Union[int, Tuple[int, int]],
    activation: str = "relu",
    norm: str = "layer",
    aggr: str = "mean",
    dropout: float = 0.1,
) -> MessagePassing:
    """Returns a GNN layer for heterogeneous convolution.

    :param layer_name: Name of the layer.
    :type layer_name: str
    :param emb_dim: Number of hidden dimensions to the layer.
    :type emb_dim: int
    :param activation: Name of the activation function to use for the layer.
    :type activation: str
    :param norm: Name of the normalisation function to use after the layer.
    :type norm: str
    :param aggr: Name of the aggregation function to use for message-passing layers.
    :type aggr: str
    :param dropout: Probability to use for dropout after the layer.
    :type dropout: float
    :return: A GNN layer.
    :rtype: MessagePassing
    """
    if layer_name == "GCN":
        return GCNConv(emb_dim, emb_dim)  # type: ignore
    elif layer_name == "EGNN":
        return EGNNLayer(
            emb_dim, activation, norm, aggr, dropout
        )  # type: ignore
    else:
        raise ValueError(f"Unknown layer: {layer_name}")


class VirtualGNN(nn.Module):
    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        layer_types: Mapping[str, Any],
        activation: str = "relu",
        aggr: str = "sum",
        pool: str = "mean",
        dropout: float = 0.1,
        pos_features: bool = True,
        edge_features: bool = False,
        residual: bool = True,
    ):
        """Initializes a GNN encoder with the provided parameters.

        :param num_layers: Number of GNN layers per/between each hierarchy.
        :type num_layers: int
        :param layer_types: Type of GNN layer to use for each edge type between hierarchies.
        :type layer_types: Mapping[str, Dict[str, List[Union[str, int, List[str]]]]]
        :param activation: Type of activation function to use for all layers. Defaults to "relu".
        :type activation: str
        :param aggr: Type of aggregation function to use to group node embeddings from different hierarchies. Defaults to "sum".
        :type aggr: str
        :param dropout: Probability to use for all layer dropout functions. Defaults to 0.1.
        :type dropout: float
        :param pos_features: Whether to use geometric features. Defaults to True.
        :type pos_features: bool
        :param edge_features: Whether to use edge features. Defaults to False.
        :type edge_features: bool
        :param residual: Whether to use residual connections. Defaults to True.
        :type residual: bool
        """
        super().__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.layer_types = layer_types
        self.activation = activation
        self.aggr = aggr
        self.dropout = dropout
        self.pos_features = pos_features
        self.edge_features = edge_features
        self.residual = residual

        self.emb_in = nn.LazyLinear(emb_dim)
        self.layers = self._build_gnn_encoder()
        self.pool = get_aggregation(pool)

    def _build_gnn_encoder(self) -> nn.Sequential:
        """Builds a GNN encoder for a hierarchical virtual node model."""
        # Define separate GNN layer per heter-node type
        conv_dict = {}
        for edge_name, layer_config in self.layer_types.items():
            assert not edge_name.startswith(
                "_"
            ), "Edge names cannot start with _, as they are reserved for internal use"
            if edge_name == "r_to_r":
                if "pairs" in layer_config.keys():
                    logging.warning(
                        "Edge type `r_to_r` is reserved for real node connections, ignoring pairs value"
                    )
                pairs = [["real", "real"]]
            else:
                assert (
                    "pairs" in layer_config.keys()
                ), f"Pairs must be specified for non-real edge type {edge_name}"
                pairs = layer_config["pairs"]

            # Add layers
            assert (
                "n_dims" in layer_config.keys()
            ), f"Hidden dimensions must be specified for edge type {edge_name}"
            assert (
                "layers" in layer_config.keys()
            ), f"GNN Layer type must be specified for edge type {edge_name}"
            assert len(layer_config["n_dims"]) == len(
                pairs
            ), f"Number of hidden dimensions must match the number of pairs for edge type {edge_name}"
            assert len(layer_config["layers"]) == len(
                pairs
            ), f"Number of layer types must match the number of pairs for edge type {edge_name}"
            for i, n_from, n_to in enumerate(pairs):
                conv_dict[n_from, edge_name, n_to] = get_gnn_layer(
                    layer_config["layers"][i],
                    self.emb_dim,
                    activation=self.activation,
                    dropout=self.dropout,
                )

        # Stack of Heterogeneous Conv Layers
        convs = nn.ModuleList()
        for _ in self.num_layers:
            convs.append(HeteroConv(conv_dict, aggr=self.aggr))

        return convs

    @property
    def required_batch_attributes(self) -> Set[str]:
        """Required batch attributes for this encoder.

        :return: Set of required attributes
        :rtype: Set[str]
        """
        feature_set = {"x_dict", "edge_index_dict", "batch_dict"}
        if self.pos_features:
            feature_set.add("pos_dict")
        if self.edge_features:
            feature_set.add("edge_attr_dict")
        return feature_set

    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the VirtualGNN encoder.

        Returns the node embedding and graph embedding in a dictionary.

        :param batch: Batch of data to encode.
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings. Contains
            ``node_embedding`` and ``graph_embedding`` fields. The node
            embedding is of shape :math:`(|V|, d)` and the graph embedding is
            of shape :math:`(n, d)`, where :math:`|V|` is the number of nodes in the original graph
            and :math:`n` is the number of graphs in the batch and :math:`d` is
            the dimension of the embeddings.
        :rtype: EncoderOutput
        """
        h_dict = {
            type_name: self.emb_in(batch.x_dict[type_name])
            for type_name in batch.x_dict
        }
        pos_dict = {
            type_name: batch.pos_dict[type_name]
            for type_name in batch.pos_dict
        }

        for layer in self.layers:
            if self.edge_features and self.pos_features:
                # Message passing layer
                h_update_dict, pos_update_dict = layer(
                    h_dict,
                    pos_dict,
                    batch.edge_index_dict,
                    batch.edge_attr_dict,
                )
                # Update node features (n, d) -> (n, d)
                h_dict = {
                    type_name: (
                        h_dict[type_name] + h_update_dict[type_name]
                        if self.residual
                        else h_update_dict[type_name]
                    )
                    for type_name in h_update_dict
                }
                # Update node coordinates (n, 3) -> (n, 3)
                pos_dict = {
                    type_name: (
                        pos_dict[type_name] + pos_update_dict[type_name]
                        if self.residual
                        else pos_update_dict[type_name]
                    )
                    for type_name in pos_update_dict
                }
            elif self.edge_features:
                # Message passing layer
                h_update_dict = layer(
                    h_dict,
                    pos_dict,
                    batch.edge_index_dict,
                    batch.edge_attr_dict,
                )
                # Update node features (n, d) -> (n, d)
                h_dict = {
                    type_name: (
                        h_dict[type_name] + h_update_dict[type_name]
                        if self.residual
                        else h_update_dict[type_name]
                    )
                    for type_name in h_update_dict
                }
            elif self.pos_features:
                # Message passing layer
                h_update_dict, pos_update_dict = layer(
                    h_dict,
                    pos_dict,
                    batch.edge_index_dict,
                )
                # Update node features (n, d) -> (n, d)
                h_dict = {
                    type_name: (
                        h_dict[type_name] + h_update_dict[type_name]
                        if self.residual
                        else h_update_dict[type_name]
                    )
                    for type_name in h_update_dict
                }
                # Update node coordinates (n, 3) -> (n, 3)
                pos_dict = {
                    type_name: (
                        pos_dict[type_name] + pos_update_dict[type_name]
                        if self.residual
                        else pos_update_dict[type_name]
                    )
                    for type_name in pos_update_dict
                }
            else:
                # Message passing layer
                h_update_dict = layer(h_dict, pos_dict, batch.edge_index_dict)
                # Update node features (n, d) -> (n, d)
                h_dict = {
                    type_name: (
                        h_dict[type_name] + h_update_dict[type_name]
                        if self.residual
                        else h_update_dict[type_name]
                    )
                    for type_name in h_update_dict
                }

        return EncoderOutput(
            {
                "node_embedding": h_dict["real"],
                "graph_embedding": self.pool(
                    h_dict["real"], batch.batch_dict["real"]
                ),
                "pos": pos_dict["real"],  # Position
            }
            if self.pos_features
            else {
                "node_embedding": h_dict["real"],
                "graph_embedding": self.pool(
                    h_dict["real"], batch.batch_dict["real"]
                ),
            }
        )
