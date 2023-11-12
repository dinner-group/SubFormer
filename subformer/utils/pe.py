import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos

def comp_pe(dim: int,
            num_nodes: int,
            edge_attr: torch.Tensor or None,
            edge_index: torch.Tensor,
            normalization: str = 'rw',
            use_edge_attr: bool = False,
            correct_sign: bool = True,
            ):
    r"""Computes the local positional encoding of a graph."""
    if normalization not in ['rw', 'sym']:
        raise ValueError('Normalization must be either "rw" or "sym".')
    if use_edge_attr:
        edge_attr = edge_attr
    else:
        edge_attrs = None

    ## get graph laplacian
    edge_index, edge_attr = get_laplacian(edge_index,
                                          edge_attr,
                                          normalization=normalization,
                                          num_nodes=num_nodes)
    ## convert to scipy sparse matrix
    L = to_scipy_sparse_matrix(edge_index, edge_attr).tocsc()

    ## compute eigenvectors
    eig_val, eig_vec = np.linalg.eigh(L.todense())
    eig_vec = eig_vec[:, :dim]
    eig_vec = torch.from_numpy(eig_vec).to(float)
    eig_vec = F.normalize(eig_vec, p=2, dim=-1, eps=1e-12)

    if num_nodes < dim:
        eig_vec = F.pad(eig_vec, (0, dim - num_nodes), value=float('nan'))
    else:
        eig_vec = eig_vec

    if correct_sign:
        ## correct signs

        sign = torch.rand(eig_vec.shape[-1])
        sign[sign < 0.5] = -1
        sign[sign >= 0.5] = 1

        eig_vec = eig_vec * sign.unsqueeze(0)

        return eig_vec

    else:

        return eig_vec


def comp_deg(
        edge_index: torch.Tensor,
        num_nodes: int,
):
    r"""Computes the degree of each node in a graph."""

    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    degree = adj.long().sum(dim=1).view(-1)

    return degree


def comp_spa(
        dim: int,
        num_nodes: int,
        edge_index: torch.Tensor,
):
    r"""Computes the spatial distance of each node in a graph."""

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True
    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    spatial_pos = torch.from_numpy(shortest_path_result)

    _, eig_vec = np.linalg.eigh(spatial_pos)
    eig_vec = eig_vec[:, :dim]
    eig_vec = torch.from_numpy(eig_vec).to(float)
    eig_vec = F.normalize(eig_vec, p=2, dim=-1, eps=1e-12)

    if num_nodes < dim:
        eig_vec = F.pad(eig_vec, (0, dim - num_nodes), value=float('nan'))
    else:
        eig_vec = eig_vec

    return eig_vec


