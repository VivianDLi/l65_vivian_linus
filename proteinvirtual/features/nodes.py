import math
from typing import Mapping
import torch
from torch_geometric.data import HeteroData, Data


def fps_torch(pos: torch.tensor, k: int = 1, select: int = 0) -> torch.tensor:
    """_summary_

    Args:
        pos (torch.tensor): (N, 3) tensor of positions
        k (int): number of samples to select
        select (int, optional): Optional initial position to be sampled

    Return:
        torch.tensor: (k, 3) tensor of sampled positions
    
    """
    
    n = pos.shape[0]
    assert pos.shape[1] == 3, "Position tensor must be (N, 3)"
    
    selected = torch.zeros(n, dtype=torch.bool)
    selected[select] = True
    
    distances = torch.pairwise_distance(pos, pos[selected])
    distances[selected] = - math.inf
    
    for i in range(k-1):
        new_sample = torch.argmax(distances)
        selected[new_sample] = True
        distances = torch.min(distances, torch.pairwise_distance(pos, pos[new_sample]))
        distances[selected] = - math.inf

    return pos[selected]
    

def compute_random_uniform(num_nodes, basis_pos):
    max = torch.max(basis_pos, dim=0)
    min = torch.min(basis_pos, dim=0)
    return torch.rand(num_nodes, 3, dtype=basis_pos.dtype, device=basis_pos.device) * (max - min) + min

def compute_random_normal(num_nodes, basis_pos):
    mean = torch.mean(basis_pos, dim=0)
    std = torch.std(basis_pos, dim=0)
    return torch.randn(num_nodes, 3, dtype=basis_pos.dtype, device=basis_pos.device) * std + mean

def compute_fps(num_nodes, basis_pos):
    return fps_torch(basis_pos, k=num_nodes)

   
def add_vnode_positions(position, n_nodes, node_name, data: HeteroData):
    if position in ['random', 'random_normal', 'random_uniform', 'fps']:
        position = {
            "type": position, 
            "basis": "real"
        }
    
    assert "type" in position, "Position strategy must be specified"
    assert "basis" in position, "Basis node type must be specified"
    assert position["basis"] in data.node_types, f"Basis node type {position['basis']} must be present in the data"
    assert hasattr(data["basis"], "pos") and data["basis"].pos is not None, f"Basis node positions must be present in the data"
    
    if position["type"] in ["random", "random_uniform"]:
        new_pos = compute_random_uniform(n_nodes, data[position["basis"]].pos)
    elif position["type"] == "random_normal":
        new_pos = compute_random_normal(n_nodes, data[position["basis"]].pos)
    elif position["type"] == "fps":
        new_pos = compute_fps(n_nodes, data[position["basis"]].pos)
    else:
        raise NotImplemented(f"Position strategy {position['type']} not implemented")
  
    data[node_name].pos = new_pos