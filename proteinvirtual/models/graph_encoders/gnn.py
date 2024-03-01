from typing import Any, Mapping, Set, Union

import torch.nn as nn
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from torch_geometric.data import Batch
from torch_geometric.nn.conv import HeteroConv

from proteinworkshop.models.graph_encoders.gnn import get_gnn_layer
from proteinworkshop.models.utils import get_activations, get_aggregation
from proteinworkshop.types import EncoderOutput

import logging

class VirtualGNN(nn.Module):
    def __init__(
        self,
        num_layers: int,
        layer_types: Mapping[str, Any],
        activation: str = "relu",
        aggr: str = "sum",
        pool: str = "mean",
        dropout: float = 0.1,
    ):
        """Initializes a GNN encoder with the provided parameters.

        :param num_layers: Number of GNN layers per/between each hierarchy.
        :type num_layers: int
        :param layer_types: Type of GNN layer to use for each edge type between hierarchies.
        :type layer_types: Mapping[str, Dict[str, List[Union[str, int, List[str]]]]]
        :param activation: Type of activation function to use for all layers
        :type activation: str
        :param aggr: Type of aggregation function to use to group node embeddings from different hierarchies.
        :type aggr: str
        :param dropout: Probability to use for all layer dropout functions
        :type dropout: float
        """
        super().__init__()
        self.num_layers = num_layers
        self.layer_types = layer_types
        self.activation = activation
        self.aggr = aggr
        self.dropout = dropout

        self.layers = self._build_gnn_encoder()
        self.pool = get_aggregation(pool)
        
    def _build_gnn_encoder(self) -> nn.Sequential:
        """Builds a GNN encoder for a hierarchical virtual node model."""
        # Define separate GNN layer per heter-node type
        conv_dict = {}
        for edge_name, layer_config in self.layer_types.items():
            assert not edge_name.startswith("_"), "Edge names cannot start with _, as they are reserved for internal use"
            if edge_name == "r_to_r":
                if 'pairs' in layer_config.keys():
                    logging.warning("Edge type `r_to_r` is reserved for real node connections, ignoring pairs value")
                pairs = [['real', 'real']]
            else:
                assert 'pairs' in layer_config.keys(), f"Pairs must be specified for non-real edge type {edge_name}"
                pairs = layer_config['pairs']
            
            # Add layers
            assert "n_dims" in layer_config.keys(), f"Hidden dimensions must be specified for edge type {edge_name}"
            assert "layers" in layer_config.keys(), f"GNN Layer type must be specified for edge type {edge_name}"
            assert len(layer_config["n_dims"]) == len(pairs), f"Number of hidden dimensions must match the number of pairs for edge type {edge_name}"
            assert len(layer_config["layers"]) == len(pairs), f"Number of layer types must match the number of pairs for edge type {edge_name}"
            for i, n_from, n_to in enumerate(pairs):
                layer = get_gnn_layer(layer_config["layers"][i])
                conv_dict[n_from, edge_name, n_to] = layer(layer_config["n_dims"][i], layer_config["n_dims"][min(i + 1, len(pairs) - 1)])
        
        # Stack of Heterogeneous Conv Layers
        convs = nn.ModuleList()
        for _ in self.num_layers:
            convs.append(nn.Sequential(
                HeteroConv(conv_dict, aggr=self.aggr),
                get_activations(self.activation),
                nn.Dropout(self.dropout)
            ))
        
        return convs

    @property
    def required_batch_attributes(self) -> Set[str]:
        """Required batch attributes for this encoder.

        :return: Set of required attributes
        :rtype: Set[str]
        """
        return {"x", "pos", "edge_index", "batch"}
        
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
        emb_dict = self.layers(batch.x_dict, batch.edge_index_dict)

        return EncoderOutput(
            {
                "node_embedding": emb_dict["real"],
                "graph_embedding": self.readout(emb_dict["real"], batch.batch_dict["real"]),
            }
        )
