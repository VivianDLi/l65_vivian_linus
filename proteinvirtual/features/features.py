# This file contains the functions to add node and edge features to the graph
from torch_geometric.data import HeteroData


def add_scalar_node_features(features, node_name, data: HeteroData):
    # TODO: Implement
    raise NotImplementedError("add_scalar_node_features not implemented! We lazy")

def add_vector_node_features(features, node_name, data: HeteroData):
    # TODO: Implement
    raise NotImplementedError("add_vector_node_features not implemented! We lazy")

def add_scalar_edge_features(features, n_from, edge_name, n_to, data: HeteroData):
    # TODO: Implement
    raise NotImplementedError("add_scalar_edge_features not implemented! We lazy")

def add_vector_edge_features(features, n_from, edge_name, n_to, data: HeteroData):
    # TODO: Implement
    raise NotImplementedError("add_vector_edge_features not implemented! We lazy")