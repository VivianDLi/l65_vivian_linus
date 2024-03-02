import torch
from torch_geometric.data import Data
from torch_geometric.data.storage import NodeStorage

def fool_typechecker_node(data: NodeStorage) -> Data:
    return Data().update(data.to_dict())


def generate_random_features(n_nodes, n_dim, device, dtype):
    return torch.rand(n_nodes, n_dim, dtype=dtype, device=device) * 2 - 1  # Random values between -1 and 1


def generate_zero_features(n_nodes, n_dim, device, dtype):
    return torch.zeros(n_nodes, n_dim, dtype=dtype, device=device)