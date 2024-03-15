# This file contains the functions to add node features to the graph
from typing import *
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data, Batch
from torch_geometric.data.storage import NodeStorage, EdgeStorage
from proteinworkshop.features.node_features import *
from proteinworkshop.features.edge_features import *
from proteinvirtual.features.utils import *


def neighbour_average_feature(node_pos: torch.Tensor,
                              basis_pos: torch.Tensor, 
                              basis_x: torch.Tensor, 
                              k: int,
                              dist_p: float = 2.0,
                              weight_p: float = 0.0):
    """
    Calculates the weighted average of the basis features for each node based on their k nearest neighbors.
    
    Args:
        node_pos (torch.Tensor): Tensor of shape (n_nodes, 3) representing the positions of the nodes.
        basis_pos (torch.Tensor): Tensor of shape (n_basis, 3) representing the positions of the basis nodes.
        basis_x (torch.Tensor): Tensor of shape (n_basis, d) representing the features of the basis nodes.
        k (int): Number of nearest neighbors to consider.
        dist_p (float, optional): The p-norm to compute the pairwise distances between nodes and basis nodes. Defaults to 2.0.
        weight_p (float, optional): The power to raise the inverse distances to compute the weights. Defaults to 0.0.
    
    Returns:
        torch.Tensor: Tensor of shape (n_nodes, d) representing the weighted average of the basis features for each node.
    """
     
    assert node_pos.shape[1] == 3, "Node position tensor must have shape (n_nodes, 3)"
    assert basis_pos.shape[1] == 3, "Basis position tensor must have shape (n_basis, 3)"
    assert basis_pos.shape[0] == basis_x.shape[0], "Basis position and basis feature tensors must have the same number of elements"
    
    dist = torch.cdist(node_pos, basis_pos, p=dist_p)
    sel_dist, sel_indices = torch.topk(dist, min(basis_pos.shape[0], k), largest=False)
    
    if weight_p == 0:
        weights = torch.ones_like(sel_dist)
    else:
        weights = 1 / (sel_dist + 1e-6) ** weight_p
    
    weights = weights / torch.sum(weights, dim=1, keepdim=True)
    weighted_x = torch.einsum("ik,ikj->ij", weights, basis_x[sel_indices])
    return weighted_x
    

def add_scalar_node_features(features, node_name, batch: HeteroData):
    assert node_name in batch.node_types, f"Node type {node_name} not found in batch"
    subgraph = batch[node_name]
    
    workshop_features = []
    output = []
    n_nodes = subgraph.num_nodes
    device = subgraph.batch.device
    for f in features:
        if isinstance(f, str):
            f: str = f.lower().strip()
            if f.startswith("random"):
                size = int(f.split("_")[1])
                f = dict(type="random", size=size)
            elif f.startswith("zero"):
                size = int(f.split("_")[1])
                f = dict(type="zero", size=size)
            else:
                workshop_features.append(f)
                continue
       
        assert isinstance(f, Mapping), "Feature must be a string or a mapping"
        assert "type" in f, "Feature type must be specified"
        if f["type"] == "navg":
            assert "basis" in f, "Basis node type must be specified"
            assert f["basis"] in batch.node_types, f"Basis node type {f['basis']} must be present in the data"
            basis = f["basis"]
            assert hasattr(batch[basis], "pos") and batch[basis].pos is not None, f"Basis node positions must be present in the data"
            basis_pos = batch[basis].pos
            assert hasattr(batch[basis], "x") and batch[basis].x is not None, f"Basis node features must be present in the data"
            basis_x = batch[basis].x
            k = f.get("k", 16)
            dist_p = f.get("dist_p", 2.0)
            weight_p = f.get("weight_p", 0.0)
            output.append(neighbour_average_feature(subgraph.pos, basis_pos, basis_x, k, dist_p, weight_p))
        elif f["type"] == "random":
            assert "size" in f.keys(), "Size of random feature must be specified"    
            output.append(generate_random_features(n_nodes, int(f['size']), device, torch.float))
        elif f["type"] == "zero":
            assert "size" in f.keys(), "Size of zero feature must be specified"    
            output.append(generate_zero_features(n_nodes, int(f['size']), device, torch.float))
        else:
            raise ValueError(f"Feature type {f['type']} not recognised.")
    
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
