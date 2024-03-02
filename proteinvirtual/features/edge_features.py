# This file contains the functions to add edge features to the graph
from typing import *
from jaxtyping import jaxtyped
from beartype import beartype as typechecker

import torch
from torch_geometric.data import HeteroData, Data, Batch
from torch_geometric.data.storage import NodeStorage, EdgeStorage
from proteinworkshop.features.edge_features import pos_emb
from proteinworkshop.features.utils import _normalize
from proteinvirtual.features.utils import *



def compute_edge_distance_hetero(
    pos_from: torch.Tensor,
    edge_index: torch.Tensor,
    pos_to: torch.Tensor):
    
    return torch.pairwise_distance(
        pos_from[edge_index[0, :]],  pos_to[edge_index[1, :]]
    ).unsqueeze(-1)


def add_scalar_edge_features(features, n_from, edge_name, n_to, batch: HeteroData):
    assert (n_from, edge_name, n_to) in batch.edge_types, f"Edge type {n_from} -[{edge_name}]-> {n_to} not found in batch"
    subgraph = batch[n_from, edge_name, n_to]
    node_from = batch[n_from]
    node_to = batch[n_to]
    
    output = []
    n_edges = subgraph.num_edges
    device = subgraph.edge_index.device
    for f in features:
        f: str = f.lower().strip()
        if f.startswith("random"):
            size = int(f.split("_")[1])
            output.append(generate_random_features(n_edges, size, device, torch.float))
        elif f.startswith("zero"):
            size = int(f.split("_")[1])
            output.append(generate_zero_features(n_edges, size, device, torch.float))
        elif f == "edge_distance":
            output.append(compute_edge_distance_hetero(node_from.pos, subgraph.edge_index, node_to.pos))
        elif f == "node_features":
            n1, n2 = node_from.x[subgraph.edge_index[0]], node_to.x[subgraph.edge_index[1]]
            output.append(torch.cat([n1, n2], dim=1))
        elif f == "edge_type":
            output.append(subgraph.edge_type.T)
        elif f == "orientation":
            raise NotImplementedError
        elif f == "sequence_distance":
            output.append((subgraph.edge_index[1] - subgraph.edge_index[0]).unsqueeze(-1))
        elif f == "pos_emb":
            output.append(pos_emb(subgraph.edge_index)) 
        else:
            raise ValueError(f"Unknown edge feature {f}")
    
    scalar_edge_features = torch.cat(output, dim=1)
    batch[n_from, edge_name, n_to].edge_attr = scalar_edge_features
    return batch
    

def add_vector_edge_features(features, n_from, edge_name, n_to, batch: HeteroData):
    assert (n_from, edge_name, n_to) in batch.edge_types, f"Edge type {n_from} -[{edge_name}]-> {n_to} not found in batch"
    subgraph = batch[n_from, edge_name, n_to]
    node_from = batch[n_from]
    node_to = batch[n_to]
    
    output = []
    for f in features:
        if f == "edge_vectors":
            E_vectors = node_from.pos[subgraph.edge_index[0]] - node_to.pos[subgraph.edge_index[1]]
            output.append(_normalize(E_vectors).unsqueeze(-2))
        else:
            raise ValueError(f"Unknown vector feature {f}")
    
    batch[n_from, edge_name, n_to].edge_vector_attr = torch.cat(output, dim=0)
    return batch