import torch
from torch_geometric.data import HeteroData, Data


def generate_node_position(
    strategy,
    node_type: str,
    data: HeteroData,
) -> torch.Tensor:
    if strategy == "ifps":
        raise NotImplementedError("IFPS not implemented! We lazy")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def generate_node_features(
    strategy,
    node_type: str,
    data: HeteroData,
) -> torch.Tensor:
    pass


def compute_random_uniform(num_nodes, num_features, min=0.0, max=1.0):
    return torch.rand(num_nodes, num_features) * (max - min) + min


def add_vnode_positions(position, n_nodes, node_name, data: HeteroData):
    # TODO: Implement
    pass
