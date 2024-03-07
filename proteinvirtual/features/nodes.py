import math
from typing import Mapping
import torch
from torch_scatter import scatter
from torch_geometric.data import HeteroData, Data
from torch_geometric.data.batch import Batch, DynamicInheritanceGetter


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
    assert k <= n, "Position tensor must have enough items to select"

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
    

def compute_random_uniform(num_nodes, basis_pos, basis_batch=None):
    if basis_batch is None:
        max = torch.max(basis_pos, dim=0)
        min = torch.min(basis_pos, dim=0)
        return torch.rand(num_nodes, 3, dtype=basis_pos.dtype, device=basis_pos.device) * (max - min) + min
    else:
        max = scatter(src=basis_pos, index=basis_batch, dim=0, reduce='max')  # (B, 3)
        min = scatter(src=basis_pos, index=basis_batch, dim=0, reduce='min')  # (B, 3)
        bsz = max.shape[0]
        max = torch.repeat_interleave(max, num_nodes, dim=0)
        min = torch.repeat_interleave(min, num_nodes, dim=0)
        rand = torch.rand(bsz * num_nodes, 3, dtype=basis_pos.dtype, device=basis_pos.device)
        return rand * (max - min) + min
    
    
def scatter_std(x, index, mean=None):
    if mean is None:
        mean = scatter(x, index, dim=0, reduce='mean')
    mean = torch.gather(mean, index.unsqueeze(-1).repeat(3, dim=1))
    return torch.sqrt(scatter((x - mean) ** 2, index, dim=0, reduce='mean'))


def compute_random_normal(num_nodes, basis_pos, basis_batch=None):
    if basis_batch is None:
        mean = torch.mean(basis_pos, dim=0)
        std = torch.std(basis_pos, dim=0)
        return torch.randn(num_nodes, 3, dtype=basis_pos.dtype, device=basis_pos.device) * std + mean
    else:
        mean = scatter(src=basis_pos, index=basis_batch, dim=0, reduce='mean')  # (B, 3)
        std = scatter_std(basis_pos, basis_batch, mean)  # (B, 3)
        bsz = mean.shape[0]
        mean = torch.repeat_interleave(mean, bsz, dim=0)
        std = torch.repeat_interleave(std, bsz, dim=0)
        rand = torch.randn(bsz * num_nodes, 3, dtype=basis_pos.dtype, device=basis_pos.device)
        return rand * std + mean


def compute_fps(num_nodes, basis_pos, basis_batch=None):
    if basis_batch is None:
        return fps_torch(basis_pos, k=num_nodes)
    else:
        bsz = torch.max(basis_batch).item() + 1
        out = torch.empty((bsz, num_nodes, 3), dtype=basis_pos.dtype, device=basis_pos.device)
        # TODO: look into parallelising this
        for i in range(bsz):
            out[i] = fps_torch(basis_pos[basis_batch == i], k=num_nodes)
        return out.view(-1, 3)

   
def add_vnode_positions(position, n_nodes, node_name, data: HeteroData):
    if position in ['random', 'random_normal', 'random_uniform', 'fps']:
        position = {
            "type": position, 
            "basis": "real"
        }
    
    assert "type" in position, "Position strategy must be specified"
    assert "basis" in position, "Basis node type must be specified"
    assert position["basis"] in data.node_types, f"Basis node type {position['basis']} must be present in the data"
    basis = position["basis"]
    assert hasattr(data[basis], "pos") and data[basis].pos is not None, f"Basis node positions must be present in the data"
    
    if position["type"] in ["random", "random_uniform"]:
        new_pos = compute_random_uniform(n_nodes, data[basis].pos)
    elif position["type"] == "random_normal":
        new_pos = compute_random_normal(n_nodes, data[basis].pos)
    elif position["type"] == "fps":
        new_pos = compute_fps(n_nodes, data[basis].pos)
    else:
        raise NotImplemented(f"Position strategy {position['type']} not implemented")
  
    data[node_name].pos = new_pos
    return data


def add_vnode_positions_batch(position,
                              n_nodes,
                              node_name,
                              batch: Batch) -> Batch:
    assert isinstance(batch, HeteroData), "Batch must be a Batch(HeteroData)"
    if position in ['random', 'random_normal', 'random_uniform', 'fps']:
        position = {
            "type": position, 
            "basis": "real"
        }
    assert "type" in position, "Position strategy must be specified"
    assert "basis" in position, "Basis node type must be specified"
    assert position["basis"] in batch.node_types, f"Basis node type {position['basis']} must be present in the data"
    basis = position["basis"]
    assert hasattr(batch[basis], "pos") and batch[basis].pos is not None, f"Basis node positions must be present in the data"
    assert hasattr(batch[basis], "batch") and batch[basis].batch is not None, f"Missing batch information for node type {basis}"
     
    if position["type"] in ["random", "random_uniform"]:
        new_pos = compute_random_uniform(n_nodes, batch[basis].pos, batch[basis].batch)
    elif position["type"] == "random_normal":
        new_pos = compute_random_normal(n_nodes, batch[basis].pos, batch[basis].batch)
    elif position["type"] == "fps":
        new_pos = compute_fps(n_nodes, batch[basis].pos, batch[basis].batch)
    else:
        raise NotImplemented(f"Position strategy {position['type']} not implemented")
    
    new_batch = torch.arange(batch.num_graphs).repeat_interleave(n_nodes).to(dtype=torch.long, device=new_pos.device)
    batch[node_name].pos = new_pos
    batch[node_name].batch = new_batch
    return batch