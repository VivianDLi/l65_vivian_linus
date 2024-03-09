from typing import *

import logging
import torch
from torch_geometric.nn.pool import knn, radius
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.batch import Batch
from proteinworkshop.features.edges import sequence_edges
from proteinvirtual.features.utils import fool_typechecker_node


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
        torch.arange(0, from_length, dtype=torch.long, device=data[from_type].pos.device),
        torch.arange(0, to_length, dtype=torch.long, device=data[from_type].pos.device)
    ).T
    
def compute_fc_by_node_type_batch(
    from_type: str,
    to_type: str,
    batch: Batch):
    assert isinstance(batch, HeteroData), "Batch must be a Batch(HeteroData)"
    
    logging.debug(f"compute_fc_by_node_type_batch: {from_type} -> {to_type}")
    bsz = batch.num_graphs
    output = []
    for i in range(bsz):
        from_idx = (batch[from_type].batch==i).nonzero().long().squeeze(-1)
        to_idx = (batch[to_type].batch==i).nonzero().long().squeeze(-1)
        # logging.debug(f"compute_fc_by_node_type_batch: Lengths: {len(from_idx)} -> {len(to_idx)}")
        output.append(torch.cartesian_prod(
            from_idx,
            to_idx
        ).T)
    return torch.cat(output, dim=1)

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
    from_batch: torch.Tensor = None,
    to_batch: torch.Tensor = None,
    k: int = 5,
    **kwargs
):
    assert from_pos.shape[1]==3 and to_pos.shape[1]==3, "Expecting pos tensor of shape (*, 3)"
    logging.debug(f"compute_knn: For {from_pos.shape[0]} -> {to_pos.shape[0]} nodes, k={k}, kwargs={kwargs}")

    return knn(x=to_pos, 
               y=from_pos, 
               k=k,
               batch_x=to_batch, 
               batch_y=from_batch, 
               **kwargs)


def compute_radius(
    from_pos: torch.Tensor,
    to_pos: torch.Tensor,
    r: float = 1.0,
    from_batch: torch.Tensor = None,
    to_batch: torch.Tensor = None,
    **kwargs
):
    assert from_pos.shape[1]==3 and to_pos.shape[1]==3, "Expecting pos tensor of shape (*, 3)"
    logging.debug(f"compute_radius: For {from_pos.shape[0]} -> {to_pos.shape[0]} nodes, radius={r}, kwargs={kwargs}")

    return radius(x=to_pos, 
                  y=from_pos, 
                  r=r,
                  batch_x=to_batch, 
                  batch_y=from_batch,
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
            wrapped = fool_typechecker_node(data[n_from])
            new_edges = sequence_edges(wrapped, chains=wrapped.chains, direction=direction)
        else:
            raise ValueError(f"Edge strategy {strat} not recognised")

        edges.append(new_edges)
    
    indxs = torch.cat(
        [
            torch.ones_like(e_idx[0, :]) * idx
            for idx, e_idx in enumerate(edges)
        ],
        dim=0,
    ).unsqueeze(0)
    
    data[n_from, edge_name, n_to].edge_index = torch.cat(edges, dim=1)
    data[n_from, edge_name, n_to].edge_type = indxs
    return data


def add_edge_batch(strategies: List[str], n_from, edge_name, n_to, batch: Batch, bidirectional: bool = False):
    assert isinstance(batch, HeteroData), "Batch must be a Batch(HeteroData)"
    assert n_from in batch.node_types, f"Node type {n_from} not found in batch"
    assert n_to in batch.node_types, f"Node type {n_to} not found in batch"
    
    edges = []
    for strat in strategies:
        strat: str = strat.lower().strip()
        if strat.startswith("knn"):
            val = strat.split("_")[1]
            new_edges = compute_knn(
                from_pos=batch[n_from].pos, 
                to_pos=batch[n_to].pos,
                from_batch=batch[n_from].batch,
                to_batch=batch[n_to].batch,
                k=int(val))
        elif strat.startswith("eps"):
            val = strat.split("_")[1]
            new_edges = compute_radius(
                from_pos=batch[n_from].pos, 
                to_pos=batch[n_to].pos,
                from_batch=batch[n_from].batch,
                to_batch=batch[n_to].batch,
                r=float(val))
        elif strat in ["full", "fc", "fully_connected"]:
            new_edges = compute_fc_by_node_type_batch(n_from, n_to, batch)
        elif strat in ["seq_forward", "seq_backward"]:
            direction = strat.split("_")[1]
            assert n_from == "real" and edge_name == "r_to_r" and n_to == "real", "Only real to real sequence edges are supported"
            
            wrapped_data = Data().update(batch[n_from].to_dict())
            new_edges = sequence_edges(wrapped_data, chains=wrapped_data.chains, direction=direction)
        else:
            raise ValueError(f"Edge strategy {strat} not recognised")

        edges.append(new_edges)
    
    indxs = torch.cat(
        [
            torch.ones_like(e_idx[0, :]) * idx
            for idx, e_idx in enumerate(edges)
        ],
        dim=0,
    ).unsqueeze(0)

    edge_index = torch.cat(edges, dim=1)
    batch[n_from, edge_name, n_to].edge_index = edge_index
    batch[n_from, edge_name, n_to].edge_type = indxs
    if bidirectional:
        # swap rows of edge_index to swap edge direction
        batch[n_to, edge_name, n_from].edge_index = edge_index.index_select(0, torch.LongTensor([1, 0]).to(device=edge_index.device))
        batch[n_to, edge_name, n_from].edge_type = indxs
    return batch
    