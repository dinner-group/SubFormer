import torch
from torch.nn import Linear, BatchNorm1d
from torch_geometric.data import Data


class PositionalEncoding(torch.nn.Module):
    def __init__(self,
                 pe_dim: int = 16,
                 hidden_channels: int = 64,
                 activation: str = 'relu',
                 concat_pe: bool = False,
                 use_deg: bool = True,
                 use_lpe: bool = True,
                 use_spa: bool = False,
                 ):
        super(PositionalEncoding, self).__init__()

        self.use_deg = use_deg
        self.use_lpe = use_lpe
        self.use_spa = use_spa
        self.concat_pe = concat_pe
        # if concat_pe:
        #     self.pre_norm = BatchNorm1d(hidden_channels * 2 )
        # else:
        #     self.pre_norm = BatchNorm1d(hidden_channels)

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()

        if self.use_deg == False and self.use_lpe == False and self.use_spa == False:
            raise ValueError('At least one of use_deg, use_lpe, use_spa must be True')

        if use_deg:
            self.deg_emb = torch.nn.Embedding(50, hidden_channels)
            self.deg_lin = Linear(hidden_channels, hidden_channels)
            self.deg_merge = Linear(hidden_channels, hidden_channels)

        if use_lpe:
            self.lpe_lin = Linear(pe_dim, hidden_channels)
            if not concat_pe:
                self.lpe_merge = Linear(hidden_channels, hidden_channels)

        if use_spa:
            self.spa_lin = Linear(pe_dim, hidden_channels)
            if not concat_pe:
                self.spa_merge = Linear(hidden_channels, hidden_channels)


            print('Concatenating positional encodings,please remember to change the input dimension of the encoder')
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_deg:
            self.deg_emb.reset_parameters()
            self.deg_lin.reset_parameters()
            self.deg_merge.reset_parameters()

        if self.use_lpe and not self.concat_pe:
            self.lpe_lin.reset_parameters()
            self.lpe_merge.reset_parameters()

        if self.use_spa:
            self.spa_lin.reset_parameters()
            if not self.concat_pe:
                self.spa_merge.reset_parameters()

    def forward(self, data: Data, x_clique: torch.Tensor) -> torch.Tensor:

        if self.use_deg:
            deg = self.deg_emb(data.tree_degree)
            deg = self.deg_lin(deg)
            deg = self.activation(deg)

            x_clique = x_clique + deg
            x_clique = self.deg_merge(x_clique)

        if self.use_lpe:
            lpe = data.tree_lpe.to(torch.float32)
            lpe_mask = torch.isnan(lpe)
            lpe[lpe_mask] = 0
            # lpe = self.activation(lpe)
            lpe = self.lpe_lin(lpe)
            lpe = self.activation(lpe)

            if self.concat_pe:
                x_clique = torch.cat([x_clique, lpe], dim=-1)

            else:
                x_clique = x_clique + lpe
                x_clique = self.lpe_merge(x_clique)

        if self.use_spa:
            spa = data.tree_spa.to(torch.float32)
            spa_mask = torch.isnan(spa)
            spa[spa_mask] = 0
            # spa = self.activation(spa)
            spa = self.spa_lin(spa)
            spa = self.activation(spa)

            if self.concat_pe:
                x_clique = torch.cat([x_clique, spa], dim=-1)

            else:
                x_clique = x_clique + spa

        return x_clique
