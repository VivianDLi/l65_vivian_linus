import logging
import torch
from torch_geometric.nn.pool import knn
from torch_geometric.data import HeteroData
import numpy as np



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
    assert from_pos.shape[1]==3 and to_pos.shape[1]==3, "Expecting poss tensor of shape (*, 3)"
    logging.debug(f"compute_knn: For {from_pos.shape[0]} -> {to_pos.shape[0]} nodes, k={k}, kwargs={kwargs}")

    return knn(to_pos, from_pos, k=k,
               batch_x=batch_x, batch_y=batch_y, **kwargs)


def add_edges(strategies, n_from, edge_name, n_to, data: HeteroData):
    # TODO: Implement
    pass