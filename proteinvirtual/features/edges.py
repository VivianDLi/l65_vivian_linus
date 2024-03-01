from typing import *

import logging
import torch
from torch_geometric.nn.pool import knn, radius
from torch_geometric.data import HeteroData
from proteinworkshop.features.edges import sequence_edges



def compute_fully_connect_by_node_type(
        from_type: str,
        to_type: str,
        data: HeteroData,
):
    logging.debug(f"compute_fully_connect_by_node_type: {from_type} -> {to_type}")
    from_length = len(data[from_type].pos)
    to_length = len(data[to_type].pos)

    logging.debug(f"compute_fully_connect_by_node_type: Lengths: {from_length} -> {to_length}")

    return torch.cartesian_prod(
        torch.arange(0, from_length, dtype=torch.long),
        torch.arange(0, to_length, dtype=torch.long)
    )

def compute_fully_connected_by_indices(
    from_indices: torch.Tensor,
    to_indices: torch.Tensor,
):
    logging.debug(f"compute_fully_connected_by_indices: Lengths: {len(from_indices)} -> {len(to_indices)}")
    return torch.cartesian_prod(
        from_indices.to(torch.long),
        to_indices.to(torch.long)
    )


def compute_knn(
    from_pos: torch.Tensor,
    to_pos: torch.Tensor,
    batch_x: torch.Tensor = None,
    batch_y: torch.Tensor = None,
    k: int = 5,
    **kwargs
):
    assert from_pos.shape[1]==3 and to_pos.shape[1]==3, "Expecting pos tensor of shape (*, 3)"
    logging.debug(f"compute_knn: For {from_pos.shape[0]} -> {to_pos.shape[0]} nodes, k={k}, kwargs={kwargs}")

    return knn(x=to_pos, 
               y=from_pos, 
               k=k,
               batch_x=batch_x, 
               batch_y=batch_y, 
               **kwargs)


def compute_radius(
    from_pos: torch.Tensor,
    to_pos: torch.Tensor,
    radius: float = 1.0,
    batch_x: torch.Tensor = None,
    batch_y: torch.Tensor = None,
    **kwargs
):
    assert from_pos.shape[1]==3 and to_pos.shape[1]==3, "Expecting pos tensor of shape (*, 3)"
    logging.debug(f"compute_radius: For {from_pos.shape[0]} -> {to_pos.shape[0]} nodes, radius={radius}, kwargs={kwargs}")

    return radius(x=to_pos, 
                  y=from_pos, 
                  r=radius,
                  batch_x=batch_x, 
                  batch_y=batch_y, 
                  **kwargs)
    

def add_edges(strategies: List[str], n_from, edge_name, n_to, data: HeteroData):
    edges = []
    for strat in strategies:
        strat: str = strat.lower().strip()
        if strat.startswith("knn"):
            val = strat.split("_")[1]
            new_edges = compute_knn(data[n_from].pos, data[n_to].pos, k=int(val))
        elif strat.startswith("eps"):
            val = strat.split("_")[1]
            new_edges = compute_radius(data[n_from].pos, data[n_to].pos, radius=float(val))
        elif strat in ["full", "fc", "fully_connected"]:
            new_edges = compute_fully_connect_by_node_type(n_from, n_to, data)
        elif strat in ["seq_forward", "seq_backward"]:
            direction = strat.split("_")[1]
            assert n_from == "real" and edge_name == "r_to_r" and n_to == "real", "Only real to real sequence edges are supported"
            new_edges = sequence_edges(data["real"], chains=data["real"].chains, direction=direction)
        else:
            raise ValueError(f"Edge strategy {strat} not recognised")
    
    indxs = torch.cat(
        [
            torch.ones_like(e_idx[0, :]) * idx
            for idx, e_idx in enumerate(edges)
        ],
        dim=0,
    ).unsqueeze(0)
    
    data[n_from, edge_name, n_to].edge_index = torch.cat(edges, dim=1)
    data[n_from, edge_name, n_to].edge_type = indxs