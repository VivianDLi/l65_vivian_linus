import torch
from torch import nn
from torch_geometric.data import *
from proteinvirtual.constants import *
from proteinworkshop.features.factory import *
from proteinvirtual.features.nodes import *
from proteinvirtual.features.edges import *
from proteinvirtual.features.features import *

from typing import *
import logging

# Type definitions
PositionConfig = Union[str, Mapping[str, str]]
VirtualNodeConfig = Mapping[str, Union[int, PositionalEncoding]]

class VirtualProteinFeaturiser(nn.Module):
    def __init__(self,
                 representation: StructureRepresentation,
                 virtual_nodes: Mapping[str, VirtualNodeConfig],
                 edge_types: Mapping[str, Any],
                 scalar_node_features: Mapping[str, List[ScalarNodeFeature]],
                 vector_node_features: Mapping[str, List[VectorNodeFeature]],
                 scalar_edge_features: Mapping[str, List[ScalarEdgeFeature]],
                 vector_edge_features: Mapping[str, List[VectorEdgeFeature]],           
    ):
        super(VirtualProteinFeaturiser, self).__init__()
        self.representation = representation
        self.virtual_nodes = virtual_nodes
        self.edge_types = edge_types
        self.scalar_node_features = scalar_node_features
        self.vector_node_features = vector_node_features
        self.scalar_edge_features = scalar_edge_features
        self.vector_edge_features = vector_edge_features
    
    def forward(self, batch: Union[Batch, ProteinBatch]) -> Union[Batch, ProteinBatch]:
        # Representation
        batch = transform_representation(batch, self.representation) 
        
        out_batch = []
        for data in batch.to_data_list():
            # Convert to hetero data
            if isinstance(data, Data):
                data = data.to_heterogeneous(node_type_names=["real"], edge_type_names=["r_to_r"])
        
            assert isinstance(data, HeteroData), "data should be HeteroData by this point"
            
            # Add virtual nodes
            virtual_node_names = []
            node_counts: Dict[str, int] = {}
            if self.virtual_nodes:
                for node_name, node_config in self.virtual_nodes.items():
                    assert node_name != "real", "Cannot name a virtual node 'real'"
                    assert not node_name.startswith("_"), "Node names cannot start with _, as they are reserved for internal use"
                    assert 'n_nodes' in node_config.keys(), "n_nodes must be specified for virtual nodes"
                    n_nodes = node_config['n_nodes']
                    virtual_node_names.append(node_name)
                    node_counts[node_name] = n_nodes
                    # Add position if available
                    if 'position' in node_config.keys():
                        data = add_vnode_positions(node_config['position'], n_nodes, node_name, data)
            
            # Add real annd virtual edges
            edge_from_to = {}
            if self.edge_types:
                for edge_name, edge_config in self.edge_types.items():
                    assert not edge_name.startswith("_"), "Edge names cannot start with _, as they are reserved for internal use"
                    if edge_name == "r_to_r":
                        if 'pairs' in edge_config.keys():
                            logging.warning("Edge type `r_to_r` is reserved for real node connections, ignoring pairs value")
                        pairs = [['real', 'real']]
                    else:
                        assert 'pairs' in edge_config.keys(), f"Pairs must be specified for non-real edge type {edge_name}"
                        pairs = edge_config['pairs']

                    
                    # SHORTCUT 1: shorter pairs statement
                    if isinstance(pairs, str):
                        if pairs == "_all_":
                            pairs = [(i, j) for i in virtual_node_names + ['real'] for j in virtual_node_names + ['real']]
                        elif pairs == "_allv_":
                            pairs = [(i, j) for i in virtual_node_names for j in virtual_node_names]
                        elif pairs == "_allv2r_":
                            pairs = [(i, 'real') for i in virtual_node_names]
                        elif pairs == "_allr2v_":
                            pairs = [('real', i) for i in virtual_node_names]
                        else:
                            raise ValueError(f"Invalid pairs value: {pairs} for edge type {edge_name}")
                    
                    # Add edges
                    assert "strategies" in edge_config.keys(), f"Strategies must be specified for edge type {edge_name}"
                    for n_from, n_to in pairs:
                        if edge_name not in edge_from_to:
                            edge_from_to[edge_name] = []
                        edge_from_to[edge_name].append((n_from, n_to))
                        data = add_edges(edge_config['strategies'], n_from, edge_name, n_to, data)
            
            # Add scalar node features
            if self.scalar_edge_features:
                for node_name, features in self.scalar_node_features.items():
                    if node_name == "_notreal_":
                        for vname in virtual_node_names:
                            data = add_scalar_node_features(features, vname, data)
                    elif node_name == "_all_":
                        for vname in virtual_node_names + ['real']:
                            data = add_scalar_node_features(features, vname, data)
                    else:
                        data = add_scalar_node_features(features, node_name, data)
                    
            # Add vector node features
            if self.vector_node_features:
                for node_name, features in self.vector_node_features.items():
                    if node_name == "_notreal_":
                        for vname in virtual_node_names:
                            data = add_vector_node_features(features, vname, data)
                    elif node_name == "_all_":
                        for vname in virtual_node_names + ['real']:
                            data = add_vector_node_features(features, vname, data)
                    else:
                        data = add_vector_node_features(features, node_name, data)
            
            # Add scalar edge_features
            if self.scalar_edge_features:
                for edge_name, features in self.scalar_edge_features.items():
                    for n_from, n_to in edge_from_to[edge_name]:
                        data = add_scalar_edge_features(features, n_from, edge_name, n_to, data)
            
            # Add vector edge_features
            if self.vector_edge_features:
                for edge_name, features in self.vector_edge_features.items():
                    for n_from, n_to in edge_from_to[edge_name]:
                        data = add_vector_edge_features(features, n_from, edge_name, n_to, data)
        
            out_batch.append(data)
        
        out_batch = Batch.from_data_list(out_batch)
        return out_batch