import torch
from torch import nn
from torch_geometric.data import *
from torch_geometric.data.batch import DynamicInheritanceGetter
from proteinworkshop.features.factory import *
from proteinvirtual.features.nodes import *
from proteinvirtual.features.edges import *
from proteinvirtual.features.node_features import *
from proteinvirtual.features.edge_features import *

from loguru import logger as log
from typing import *
import logging

# Type definitions
PositionConfig = Union[str, Mapping[str, str]]
VirtualNodeConfig = Mapping[str, Union[int, PositionalEncoding]]


def convert_to_hetereo(data: Data) -> HeteroData:
    attr_dict = data.to_dict()
    attributes = set(attr_dict.keys())
    node_attributes = set(data.node_attrs())
    edge_attributes = set(data.edge_attrs())
    other_attributes = attributes - node_attributes - edge_attributes
    out = HeteroData()
    
    if len(node_attributes) > 0:
        out["real"].update({k: attr_dict[k] for k in node_attributes})
    
    if len(edge_attributes) > 0:
        out["real", "r_to_r", "real"].update({k: attr_dict[k] for k in edge_attributes})
    
    if len(other_attributes) > 0:
        for k in other_attributes:
            setattr(out, k, attr_dict[k])

    return out


def convert_to_hetero_batch(batch: Batch, group_attributes_by_element=False) -> Batch:
    out = DynamicInheritanceGetter()(Batch, HeteroData)
    
    attr_dict = batch.to_dict()
    if group_attributes_by_element:
        assert batch.num_graphs > 0, "Batch must contain at least one graph if group_attributes_by_element"
        _attr_src = batch[0]
    else:
        _attr_src = batch
    
    attributes = set(attr_dict.keys())
    node_attributes = set(_attr_src.node_attrs())
    edge_attributes = set(_attr_src.edge_attrs())
    other_attributes = attributes - node_attributes - edge_attributes
    
    if len(node_attributes) > 0:
        out["real"].update({k: attr_dict[k] for k in node_attributes})
        if hasattr(batch, "_slice_dict"):
            setattr(out['real'], "_slice_dict", batch._slice_dict)
        
        if hasattr(batch, "_inc_dict"):
            setattr(out['real'], "_inc_dict", batch._inc_dict)
    
    if len(edge_attributes) > 0:
        out["real", "r_to_r", "real"].update({k: attr_dict[k] for k in edge_attributes})
    
    if len(other_attributes) > 0:
        for k in other_attributes:
            setattr(out, k, attr_dict[k])
    
    return out


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
        if hasattr(batch, "coords"):
            batch = transform_representation(batch, self.representation) 
        
        if isinstance(batch, Data):
            batch = convert_to_hetero_batch(batch)
            # batch = batch.to_heterogeneous(node_type_names=["real"], edge_type_names=[["real", "r_to_r", "real"]])
        
        assert isinstance(batch, HeteroData), "batch should be HeteroData by this point"
        
        # Add virtual nodes
        virtual_node_names = []
        if self.virtual_nodes:
            for node_name, node_config in self.virtual_nodes.items():
                assert node_name != "real", "Cannot name a virtual node 'real'"
                assert not node_name.startswith("_"), "Node names cannot start with _, as they are reserved for internal use"
                
                has_n_nodes = 'n_nodes' in node_config.keys()
                has_p_nodes = 'p_nodes' in node_config.keys()
                
                if has_n_nodes and has_p_nodes:
                    log.wanring(f"Both n_nodes and p_nodes specified for virtual node {node_name}, p_nodes will be ignored!")
                
                if not has_n_nodes and not has_p_nodes:
                    raise ValueError(f"one of n_node or p_node must be specified for virtual node {node_name}")
                
                batch_device = batch['real'].batch.device
                bsz = batch.num_graphs
            
                if has_n_nodes:
                    n_nodes: torch.Tensor = torch.tensor([node_config['n_nodes']] * bsz, dtype=torch.long, device=batch_device)
                else:
                    batch_array = batch['real'].batch
                    p_nodes = float(node_config['p_nodes'])
                    n_real_nodes = torch.tensor([(batch_array==i).sum() for i in range(bsz)], dtype=torch.float, device=batch_device)
                    n_nodes = torch.round(n_real_nodes * p_nodes).to(dtype=torch.long)
                
                batch[node_name].batch = torch.repeat_interleave(
                    torch.arange(bsz).to(dtype=torch.long, device=batch_device), 
                    n_nodes
                )

                virtual_node_names.append(node_name)
                
                # Add position if available
                if 'positions' in node_config.keys():
                    batch = add_vnode_positions_batch(node_config['positions'], n_nodes, node_name, batch)
        
        # Add scalar node features
        if self.scalar_node_features:
            for node_name, features in self.scalar_node_features.items():
                if len(features) == 0:
                    continue
                
                if node_name == "_notreal_":
                    for vname in virtual_node_names:
                        batch = add_scalar_node_features(features, vname, batch)
                elif node_name == "_all_":
                    for vname in virtual_node_names + ['real']:
                        batch = add_scalar_node_features(features, vname, batch)
                else:
                    batch = add_scalar_node_features(features, node_name, batch)
                
        # Add vector node features
        if self.vector_node_features:
            for node_name, features in self.vector_node_features.items():
                if len(features) == 0:
                    continue
                
                if node_name == "_notreal_":
                    for vname in virtual_node_names:
                        batch = add_vector_node_features(features, vname, batch)
                elif node_name == "_all_":
                    for vname in virtual_node_names + ['real']:
                        batch = add_vector_node_features(features, vname, batch)
                else:
                    batch = add_vector_node_features(features, node_name, batch)
        
        # Add edges
        edge_from_to = {}
        if self.edge_types:
            for edge_name, edge_config in self.edge_types.items():
                assert not edge_name.startswith("_"), "Edge names cannot start with _, as they are reserved for internal use"
                if edge_name == "r_to_r":
                    if 'pairs' in edge_config.keys():
                        log.warning("Edge type `r_to_r` is reserved for real node connections, ignoring pairs value")
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
                    batch = add_edge_batch(edge_config['strategies'], n_from, edge_name, n_to, batch)
        
        # Add scalar edge_features
        if self.scalar_edge_features:
            for edge_name, features in self.scalar_edge_features.items():
                if len(features) == 0:
                    continue
                edges_to_add = []
                if edge_name == "_all_":
                    edges_to_add = edge_from_to.keys()
                else:
                    edges_to_add = [edge_name]
                for e_name in edges_to_add:
                    for n_from, n_to in edge_from_to[e_name]:
                        batch = add_scalar_edge_features(features, n_from, e_name, n_to, batch)
        
        # Add vector edge_features
        if self.vector_edge_features:
            for edge_name, features in self.vector_edge_features.items():
                if len(features) == 0:
                    continue
                edges_to_add = []
                if edge_name == "_all_":
                    edges_to_add = edge_from_to.keys()
                else:
                    edges_to_add = [edge_name]
                for e_name in edges_to_add:
                    for n_from, n_to in edge_from_to[e_name]:
                        batch = add_vector_edge_features(features, n_from, e_name, n_to, batch)
        
        return batch