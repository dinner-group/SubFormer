# data class from https://github.com/rusty1s/himp-gnn

import rdkit.Chem
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, VirtualNode
from torch_geometric.utils.tree_decomposition import tree_decomposition
from subformer.utils.pe import comp_pe, comp_deg, comp_spa
from collections import defaultdict

bonds = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]


def mol_from_data(data):
    mol = Chem.RWMol()
    x = data.x if data.x.dim() == 1 else data.x[:, 0]
    for z in x.tolist():
        mol.AddAtom(Chem.Atom(z))

    row, col = data.edge_index
    mask = row < col
    row, col = row[mask].tolist(), col[mask].tolist()

    bond_type = data.edge_attr
    bond_type = bond_type if bond_type.dim() == 1 else bond_type[:, 0]
    bond_type = bond_type[mask].tolist()

    for i, j, bond in zip(row, col, bond_type):
        # print(bond)
        # # for lrgb we plus 1
        bond = int(bond)  # + 1
        assert 1 <= bond <= 4
        mol.AddBond(i, j, bonds[bond - 1])

    return mol.GetMol()


class JunctionTreeData(Data):
    def __inc__(self, key, item, *args):
        if key == 'tree_edge_index':
            return self.x_clique.size(0)
        elif key == 'atom2clique_index':
            return torch.tensor([[self.x.size(0)], [self.x_clique.size(0)]])
        else:
            return super(JunctionTreeData, self).__inc__(key, item, *args)


class OGBTransform(object):
    # OGB saves atom and bond types zero-index based. We need to revert that.
    def __call__(self, data: Data) -> Data:
        data.x[:, 0] += 1
        data.edge_attr[:, 0] += 1
        return data


class GEOM_OGB(object):
    def __call__(self, data: Data):
        mol = data.mol
        tree = tree_decomposition(mol, return_vocab=True)

        tree_edge_index, atom2clique_index, num_cliques, x_clique = tree
        data = JunctionTreeData(**{k: v for k, v in data})

        data.tree_edge_index = tree_edge_index
        data.atom2clique_index = atom2clique_index
        data.num_cliques = num_cliques
        data.x_clique = x_clique
        data.edge_attr_graph = data.edge_attr
        data.edge_index_graph = data.edge_index
        data.tree_degree = comp_deg(tree_edge_index, num_cliques)

        pos = data.pos_non_h
        data.tree_pos = compute_group_centers_of_mass(atom2clique_index, pos)

        data.tree_lpe = comp_pe(
            dim=10,
            edge_attr=None,
            edge_index=tree_edge_index,
            num_nodes=num_cliques,
            normalization='rw',
            use_edge_attr=False,
            correct_sign=True
        ).to(float)
        if num_cliques < 1:
            print('this datapoint should be skipped')
            data.tree_spa = torch.zeros(num_cliques, 10)
            return data

        data.tree_spa = comp_spa(
            dim=10,
            num_nodes=num_cliques,
            edge_index=tree_edge_index,
        ).to(float)

        return data


class JunctionTree(object):
    def __call__(self, data: Data):
        mol = mol_from_data(data)
        tree = tree_decomposition(mol, return_vocab=True)
        tree_edge_index, atom2clique_index, num_cliques, x_clique = tree
        data = JunctionTreeData(**{k: v for k, v in data})

        data.tree_edge_index = tree_edge_index
        data.atom2clique_index = atom2clique_index
        data.num_cliques = num_cliques
        data.x_clique = x_clique
        data.edge_attr_graph = data.edge_attr
        data.edge_index_graph = data.edge_index

        ## graph level
        data.graph_degree = comp_deg(data.edge_index_graph, data.num_nodes)
        data.graph_lpe,data.graph_product = comp_pe(
            dim=10,
            edge_attr=None,
            edge_index=data.edge_index_graph,
            num_nodes=data.num_nodes,
            normalization='none',
            use_edge_attr=False,
            correct_sign=True
        ).to(float)
        data.graph_spa,data.graph_sp_product = comp_spa(
            dim=10,
            num_nodes=data.num_nodes,
            edge_index=data.edge_index_graph,
        ).to(float)
        ## tree level
        data.tree_degree = comp_deg(tree_edge_index, num_cliques)
        data.tree_lpe,data.tree_product = comp_pe(
            dim=10,
            edge_attr=None,
            edge_index=tree_edge_index,
            num_nodes=num_cliques,
            normalization='rw',
            use_edge_attr=False,
            correct_sign=True
        ).to(float)
        if num_cliques < 1:
            print('this datapoint should be skipped')
            data.tree_spa = torch.zeros(num_cliques, 10)
            data.tree_sp_product = torch.zeros(num_cliques, 1)
            return data

        data.tree_spa,data.tree_sp_product = comp_spa(
            dim=10,
            num_nodes=num_cliques,
            edge_index=tree_edge_index,
        ).to(float)

        return data


class JunctionTreeORG(object):
    def __call__(self, data: Data):
        mol = data.mol
        # mol = rdkit.Chem.AddHs(mol)
        tree = tree_decomposition(mol, return_vocab=True)
        tree_edge_index, atom2clique_index, num_cliques, x_clique = tree
        data = JunctionTreeData(**{k: v for k, v in data})

        data.tree_edge_index = tree_edge_index
        data.atom2clique_index = atom2clique_index
        data.num_cliques = num_cliques
        data.x_clique = x_clique
        data.edge_attr_graph = data.edge_attr
        data.edge_index_graph = data.edge_index
        data.tree_degree = comp_deg(tree_edge_index, num_cliques)
        data.tree_lpe = comp_pe(
            dim=10,
            edge_attr=None,
            edge_index=tree_edge_index,
            num_nodes=num_cliques,
            normalization='rw',
            use_edge_attr=False,
            correct_sign=True
        ).to(float)
        if num_cliques < 1:
            print('this datapoint should be skipped')
            data.tree_spa = torch.zeros(num_cliques, 10)
            return data
        pos = data.pos
        data.tree_pos = compute_group_centers_of_mass(atom2clique_index, pos)

        assert data.tree_pos.shape[0] == num_cliques
        data.tree_spa = comp_spa(
            dim=10,
            num_nodes=num_cliques,
            edge_index=tree_edge_index,
        ).to(float)

        return data


def get_transform(add_virtual_node=True):
    if add_virtual_node:
        return Compose([OGBTransform(), JunctionTree(), VirtualNode()])
    else:
        return Compose([OGBTransform(), JunctionTree()])


def get_transform_zinc(add_virtual_node=True):
    if add_virtual_node:
        return Compose([JunctionTree(), VirtualNode()])
    else:
        return Compose([JunctionTree()])


def compute_group_centers_of_mass(atom2clique_index, pos):
    # Create a dictionary to store positions for each group
    group_positions = defaultdict(list)

    for idx, group in zip(atom2clique_index[0], atom2clique_index[1]):
        group_positions[group.item()].append(idx.item())

    centers_of_mass = []
    for group, indices in group_positions.items():
        group_indices = torch.tensor(indices, dtype=torch.long)

        # Debugging information:
        if group_indices.max() >= len(pos):
            print(f"Error with group: {group}")
            print(f"Group indices: {group_indices}")
            print(f"Max index: {group_indices.max()}")
            print(f"Position tensor length: {len(pos)}")
            continue
        group_center_of_mass = pos[group_indices].mean(dim=0)
        centers_of_mass.append(group_center_of_mass)

    return torch.stack(centers_of_mass)


def get_geom_ogb():
    return Compose([OGBTransform(), GEOM_OGB()])
