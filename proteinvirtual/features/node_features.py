# This file contains the functions to add node features to the graph
from typing import *
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

import torch
from torch_geometric.data import HeteroData, Data, Batch
from torch_geometric.data.storage import NodeStorage, EdgeStorage
from proteinworkshop.features.node_features import *
from proteinworkshop.features.edge_features import *
from proteinvirtual.features.utils import *


def add_scalar_node_features(features, node_name, batch: HeteroData):
    assert node_name in batch.node_types, f"Node type {node_name} not found in batch"
    subgraph = batch[node_name]
    
    workshop_features = []
    output = []
    n_nodes = subgraph.num_nodes
    device = subgraph.batch.device
    for f in features:
        f: str = f.lower().strip()
        if f.startswith("random"):
            size = int(f.split("_")[1])
            output.append(generate_random_features(n_nodes, size, device, torch.float))
        elif f.startswith("zero"):
            size = int(f.split("_")[1])
            output.append(generate_zero_features(n_nodes, size, device, torch.float))
        else:
            workshop_features.append(f)
    
    if len(workshop_features) > 0:
        workshop_output = compute_scalar_node_features(fool_typechecker_node(subgraph), workshop_features)
        output.append(workshop_output)
    
    scalar_node_features = torch.cat(output, dim=1)
    batch[node_name].x = scalar_node_features
    batch[node_name].x = torch.nan_to_num(batch[node_name].x, nan=0.0, posinf=0.0, neginf=0.0)
    return batch


def add_vector_node_features(features, node_name, batch: HeteroData):
    assert node_name in batch.node_types, f"Node type {node_name} not found in batch"

    vector_node_features = []
    for feature in features:
        if feature == "orientation":
            assert node_name == "real", "Orientation only supported for real nodes"
            vector_node_features.append(orientations(batch[node_name].coords, batch[node_name]._slice_dict["coords"]))
        elif feature == "virtual_cb_vector":
            raise NotImplementedError("Virtual CB vector not implemented yet.")
        else:
            raise ValueError(f"Vector feature {feature} not recognised.")
    batch.x_vector_attr = torch.cat(vector_node_features, dim=0)
    
    return batch
