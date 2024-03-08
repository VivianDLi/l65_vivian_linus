import random
import torch
import pytest
import numpy as np
from proteinvirtual.features.node_features import neighbour_average_feature


def test_neighbour_average_feature():
    def nnavg_np(node_pos, basis_pos, basis_x, k, dist_p, weight_p):
        # Calculate pairwise distances between node_pos and basis_pos
        dist = np.linalg.norm(node_pos[:, np.newaxis, :] - basis_pos, ord=dist_p, axis=2)
        # Find the k nearest neighbors for each node
        knn_indices = np.argpartition(dist, k, axis=1)[:, :k]
        # Raise distances to the power of dist_p
        dist_sel = np.zeros((dist.shape[0], k))
        for i, topk in enumerate(knn_indices):
            dist_sel[i] = dist[i, topk]

        # Calculate the weights as the inverse of weight_p
        weights = 1 / (dist_sel + 1e-06) ** weight_p
        weights /= weights.sum(axis=1, keepdims=True)
        
        # Calculate the weighted node_x
        weighted_node_x = np.sum(basis_x[knn_indices] * weights[..., np.newaxis], axis=1)
        
        return weighted_node_x
    
    node_pos = torch.rand(10, 3)
    basis_pos = torch.rand(50, 3)
    basis_x = torch.rand(50, 7)
    k = random.randint(1, 5)
    weight_p = random.random() * 6 - 3
    dist_p = random.random() * 5
    out = neighbour_average_feature(node_pos, basis_pos, basis_x, k, dist_p, weight_p)
    
    np_out = nnavg_np(node_pos.numpy(), basis_pos.numpy(), basis_x.numpy(), k, dist_p, weight_p)
    out_np = out.detach().numpy()
    
    print(np_out)
    print("  == ")
    print(out_np)
    
    assert np.allclose(out_np, np_out, atol=1e-6)    
    
        