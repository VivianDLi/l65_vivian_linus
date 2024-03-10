from typing import Any, Mapping, Set, Tuple, Union

import torch.nn as nn
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from torch_geometric.data import Batch
from torch_geometric.nn.conv import GraphConv, SAGEConv, GATConv, GATv2Conv, SGConv, TransformerConv, MessagePassing

from proteinworkshop.models.utils import get_aggregation, get_activations
from proteinworkshop.types import EncoderOutput
from proteinvirtual.models.graph_encoders.layers import EGNN, HeteroConv

import logging


def get_gnn_layer(
    layer_name: str,
    emb_dim: Union[int, Tuple[int, int]],
    aggr: str = "mean",
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
        return GraphConv(emb_dim, emb_dim, aggr=aggr)  # type: ignore
    elif layer_name == "SAGE":
        return SAGEConv(emb_dim, emb_dim, aggr=aggr)
    elif layer_name == "GAT":
        return GATConv(emb_dim, emb_dim, aggr=aggr) 
    elif layer_name == "GATv2":
        return GATv2Conv(emb_dim, emb_dim, aggr=aggr)
    elif layer_name == "SGC":
        return SGConv(emb_dim, emb_dim, aggr=aggr)
    elif layer_name == "Transformer":
        return TransformerConv(emb_dim, emb_dim, aggr=aggr)
    elif layer_name == "EGNN":
        return EGNN(emb_dim, aggr=aggr)
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
        pos_features: bool = False,
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
        :param pos_features: Whether to use geometric features. Defaults to False.
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
        self.aggr = aggr
        self.pos_features = pos_features
        self.edge_features = edge_features
        self.residual = residual

        self.embeddings, self.layers = self._build_gnn_encoder()
        self.activation = get_activations(activation)
        self.dropout = nn.Dropout(dropout)
        self.pool = get_aggregation(pool)

    def _build_gnn_encoder(self) -> nn.Sequential:
        """Builds a GNN encoder for a hierarchical virtual node model."""
        # Define separate GNN layer per heter-node type and find node types
        node_names = set()
        conv_config_dict = {}
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
                "layers" in layer_config.keys()
            ), f"GNN Layer type must be specified for edge type {edge_name}"
            assert len(layer_config["layers"]) == len(
                pairs
            ), f"Number of layer types must match the number of pairs for edge type {edge_name}"
            for (n_from, n_to), conv_t in zip(pairs, layer_config["layers"]):
                conv_config_dict[n_from, edge_name, n_to] = dict(
                   layer_name=conv_t,
                   emb_dim=self.emb_dim, 
                )
                node_names.update([n_from, n_to])
        # Stack of Embeddings per node type
        embs = nn.ModuleDict()
        for node in node_names:
            embs[node] = nn.LazyLinear(self.emb_dim)

        # Stack of Heterogeneous Conv Layers
        convs = nn.ModuleList()
        for _ in range(self.num_layers):
            conv_dict = {key: get_gnn_layer(**conv_config_dict[key]) for key in conv_config_dict.keys()}
            convs.append(HeteroConv(conv_dict, aggr=self.aggr))

        return embs, convs

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
            node_name: self.embeddings[node_name](batch.x_dict[node_name])
            for node_name in batch.x_dict
        }
        if self.pos_features:
            pos_dict = batch.pos_dict

        for layer in self.layers:
            if self.edge_features and self.pos_features:
                # Message passing layer
                h_update_dict, pos_update_dict = layer(
                    h_dict,
                    batch.edge_index_dict,
                    pos_dict=pos_dict,
                    edge_attr_dict=batch.edge_attr_dict,
                )
                h_update_dict = {
                    node_name: self.dropout(
                        self.activation(h_update_dict[node_name])
                    )
                    for node_name in h_update_dict
                }
                # Update node features (n, d) -> (n, d)
                h_dict = {
                    node_name: (
                        h_dict[node_name] + h_update_dict[node_name]
                        if self.residual
                        else h_update_dict[node_name]
                    )
                    for node_name in h_update_dict
                }
                # Update node coordinates (n, 3) -> (n, 3)
                pos_dict = {
                    node_name: (
                        pos_dict[node_name] + pos_update_dict[node_name]
                        if self.residual
                        else pos_update_dict[node_name]
                    )
                    for node_name in pos_update_dict
                }
            elif self.edge_features:
                # Message passing layer
                h_update_dict = layer(
                    h_dict,
                    batch.edge_index_dict,
                    edge_attr_dict=batch.edge_attr_dict,
                )
                h_update_dict = {
                    node_name: self.dropout(
                        self.activation(h_update_dict[node_name])
                    )
                    for node_name in h_update_dict
                }
                # Update node features (n, d) -> (n, d)
                h_dict = {
                    node_name: (
                        h_dict[node_name] + h_update_dict[node_name]
                        if self.residual
                        else h_update_dict[node_name]
                    )
                    for node_name in h_update_dict
                }
            elif self.pos_features:
                # Message passing layer
                h_update_dict, pos_update_dict = layer(
                    h_dict,
                    batch.edge_index_dict,
                    pos_dict=pos_dict,
                )
                h_update_dict = {
                    node_name: self.dropout(
                        self.activation(h_update_dict[node_name])
                    )
                    for node_name in h_update_dict
                }
                # Update node features (n, d) -> (n, d)
                h_dict = {
                    node_name: (
                        h_dict[node_name] + h_update_dict[node_name]
                        if self.residual
                        else h_update_dict[node_name]
                    )
                    for node_name in h_update_dict
                }
                # Update node coordinates (n, 3) -> (n, 3)
                pos_dict = {
                    node_name: (
                        pos_dict[node_name] + pos_update_dict[node_name]
                        if self.residual
                        else pos_update_dict[node_name]
                    )
                    for node_name in pos_update_dict
                }
            else:
                # Message passing layer
                h_update_dict = layer(h_dict, batch.edge_index_dict)
                h_update_dict = {
                    node_name: self.dropout(
                        self.activation(h_update_dict[node_name])
                    )
                    for node_name in h_update_dict
                }
                # Update node features (n, d) -> (n, d)
                h_dict = {
                    node_name: (
                        h_dict[node_name] + h_update_dict[node_name]
                        if self.residual
                        else h_update_dict[node_name]
                    )
                    for node_name in h_update_dict
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
