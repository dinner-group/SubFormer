import torch
from subformer.modules.emb import AtomEncoder, BondEncoder
from subformer.modules.agat import AGATConv
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, LeakyReLU
from torch_geometric.nn import GINEConv, PNAConv
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
from torch_scatter import scatter


class LocalMP(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 num_layers: int = 3,
                 dropout: float = 0,
                 dropout_edge: float = 0,
                 local_mp: str = 'gine',
                 learn_gating: bool = True,
                 activation: str = 'relu',
                 back_activation: str = 'leaky_relu',
                 aggregation: str = 'sum',
                 pe_fea: bool = False,
                 pe_dim: int = 10,
                 deg = None,
                 ):
        super(LocalMP, self).__init__()

        self.atom_encoder = AtomEncoder(hidden_channels)
        self.clique_encoder = torch.nn.Embedding(4, hidden_channels)

        self.learn_gating = learn_gating
        self.num_layers = num_layers

        self.aggregation = aggregation

        if activation == 'relu':
            self.activation = ReLU()
        elif activation == 'leaky_relu':
            self.activation = LeakyReLU()
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()

        if back_activation == 'leaky_relu':
            self.back_activation = LeakyReLU()
        elif back_activation == 'relu':
            self.back_activation = ReLU()
        elif back_activation == 'gelu':
            self.back_activation = torch.nn.GELU()


        if learn_gating:
            self.weight_g2t = torch.nn.Parameter(torch.tensor(1.0))
            self.weight_t2g = torch.nn.Parameter(torch.tensor(1.0))

        elif not learn_gating:
            self.weight_g2t = 1.0
            self.weight_t2g = 1.0

        self.bond_encoders = torch.nn.ModuleList()
        self.graph_convs = torch.nn.ModuleList()
        self.graph_norms = torch.nn.ModuleList()
        self.sub_norms = torch.nn.ModuleList()

        if local_mp == 'gine':
            for _ in range(num_layers):
                self.bond_encoders.append(BondEncoder(hidden_channels))
                nn = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    BatchNorm1d(2 * hidden_channels),
                    ReLU(),
                    Linear(2 * hidden_channels, hidden_channels),
                )
                conv = GINEConv(nn=nn,
                                train_eps=True,
                                )
                self.graph_convs.append(conv)
                self.graph_norms.append(BatchNorm1d(hidden_channels))
                self.sub_norms.append(BatchNorm1d(hidden_channels))

        elif local_mp == 'agat':

            for _ in range(num_layers):
                self.bond_encoders.append(BondEncoder(hidden_channels))
                conv = AGATConv(hidden_channels)
                self.graph_convs.append(conv)
                self.graph_norms.append(BatchNorm1d(hidden_channels))
                self.sub_norms.append(BatchNorm1d(hidden_channels))

        else:
            print('local_mp must be gine or aGAT for now')
            raise NotImplementedError

        self.atom2clique_lins = torch.nn.ModuleList()
        self.clique2atom_lins = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.atom2clique_lins.append(
                Linear(hidden_channels, hidden_channels))
            self.clique2atom_lins.append(
                Linear(hidden_channels, hidden_channels))

        self.clique = Linear(hidden_channels, out_channels)

        self.dropout_edge = dropout_edge
        self.dropout = dropout
        self.pe_fea = pe_fea
        if pe_fea:
            self.pe_lin = Linear(pe_dim, hidden_channels)
            self.cat_lin = Linear(2 * hidden_channels, hidden_channels)

            self.degree_emb = torch.nn.Embedding(50, hidden_channels//8)
            self.degree_lin = torch.nn.Linear(hidden_channels + hidden_channels//8, hidden_channels)

            self.product_lin = Linear(1, hidden_channels//8)
            self.product_merge = Linear(hidden_channels + hidden_channels//8, hidden_channels)

            self.pe_lin = Linear(pe_dim, hidden_channels)
            self.pe_merge = Linear(2*hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        self.clique_encoder.reset_parameters()

        for emb, conv, batch_norm, sub_norm in zip(self.bond_encoders, self.graph_convs,
                                                   self.graph_norms, self.sub_norms):
            emb.reset_parameters()
            conv.reset_parameters()
            batch_norm.reset_parameters()
            sub_norm.reset_parameters()

        for lin1, lin2 in zip(self.atom2clique_lins, self.clique2atom_lins):
            lin1.reset_parameters()
            lin2.reset_parameters()

        if self.pe_fea:
            self.pe_lin.reset_parameters()
            self.cat_lin.reset_parameters()

    def init_pe(self, data: Data, x:torch.Tensor,pe_type:str):
        
        graph_deg = data.graph_degree
        graph_lpe = data.graph_lpe
        graph_product = data.graph_product
        graph_spa = data.graph_spa
        graph_sp_product = data.graph_sp_product

        if pe_type == 'lpe':
            graph_pe = graph_lpe
            graph_product = graph_product
        elif pe_type == 'spde':
            graph_pe = graph_spa
            graph_product = graph_sp_product


        graph_deg = self.degree_emb(graph_deg.long())
        x = torch.cat([x, graph_deg], dim=-1)
        x = self.degree_lin(graph_deg)
        
        graph_product = graph_product.unsqueeze(-1)
        graph_product = self.product_lin(graph_product)
        x = torch.cat([x, graph_product], dim=-1)
        x = self.product_merge(x)
        
        graph_pe = self.pe_lin(graph_pe)
        x = torch.cat([x, graph_pe], dim=-1)
        x = self.pe_merge(x)

        return x

    def forward(self, data: Data):

        x = self.atom_encoder(data.x.squeeze())
        graph_emb = x
        x_clique = self.clique_encoder(data.x_clique.squeeze())
        x_clique = self.clique(x_clique)

        if self.pe_fea:
            x = self.init_pe(data, x,self.pe_type)

        for i in range(self.num_layers):
            edge_attr = self.bond_encoders[i](data.edge_attr_graph)
            edge_index = data.edge_index_graph
            edge_index, edge_mask = dropout_edge(edge_index, p=self.dropout_edge)
            edge_attr = edge_attr[edge_mask]

            x = self.graph_convs[i](x, edge_index, edge_attr)
            x = self.graph_norms[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout)

            row, col = data.atom2clique_index
            x_clique = x_clique + self.weight_g2t * (self.activation(self.atom2clique_lins[i](scatter(
                x[row], col, dim=0, dim_size=x_clique.size(0),
                reduce=self.aggregation))))
            x_clique = self.sub_norms[i](x_clique)
            x = x + self.weight_t2g * (self.back_activation(self.clique2atom_lins[i](scatter(
                x_clique[col], row, dim=0, dim_size=x.size(0),
                reduce=self.aggregation))))

        graph_readout = scatter(x, data.batch, reduce='sum', dim=0)
        
        return x_clique, graph_emb, graph_readout
