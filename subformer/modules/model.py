import torch
from torch_geometric.data import Data
from subformer.modules.local_mp import LocalMP
from subformer.modules.pe import PositionalEncoding
from subformer.modules.encoder import Encoder


class SubFormer(torch.nn.Module):
    '''
    Main SubFormer model.
    '''

    def __init__(self,
                 hidden_channels: int = 64,
                 readout_channels: int = 128,
                 out_channels: int = 1,
                 num_mp_layers: int = 4,
                 num_enc_layers: int = 4,
                 mp_dropout: float = 0,
                 mp_dropout_edge: float = 0,
                 enc_dropout: float = 0,
                 local_mp: str = 'gine',
                 learn_gating: bool = True,
                 activation: str = 'relu',
                 back_activation: str = 'leaky_relu',
                 enc_activation: str = 'relu',
                 aggregation: str = 'sum',
                 pe_fea: bool = False,
                 pe_dim: int = 10,
                 n_head: int = 8,
                 padding_length: int = 50,
                 d_model: int = 128,
                 dim_feedforward: int = 512,
                 binary_readout: bool = False,
                 concat_pe: bool = False,
                 use_deg: bool = True,
                 use_lpe: bool = True,
                 use_spa: bool = False,
                 return_raw: bool = False,
                 ):
        super(SubFormer, self).__init__()

        self.local_mp = LocalMP(hidden_channels=hidden_channels,
                                out_channels=hidden_channels,
                                num_layers=num_mp_layers,
                                dropout=mp_dropout,
                                dropout_edge=mp_dropout_edge,
                                local_mp=local_mp,
                                learn_gating=learn_gating,
                                activation=activation,
                                back_activation=back_activation,
                                aggregation=aggregation,
                                pe_fea=pe_fea,
                                pe_dim=pe_dim,
                                )

        self.pe = PositionalEncoding(pe_dim=pe_dim,
                                     hidden_channels=hidden_channels,
                                     activation=activation,
                                     concat_pe=concat_pe,
                                     use_deg=use_deg,
                                     use_lpe=use_lpe,
                                     use_spa=use_spa,
                                     )

        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               num_encoder_layers=num_enc_layers,
                               dim_feedforward=dim_feedforward,
                               dropout=enc_dropout,
                               activation=enc_activation,
                               batch_first=True,
                               padding_length=padding_length,
                               return_raw=return_raw
                               )

        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = torch.nn.LeakyReLU()

        self.readout = torch.nn.Sequential(
            torch.nn.Linear(readout_channels, d_model),
            self.activation,
            torch.nn.Linear(d_model, d_model),
            self.activation,
            torch.nn.Linear(d_model, out_channels)
        )

        self.reset_parameters()

        self.return_raw = return_raw
        self.binary_readout = binary_readout

    def reset_parameters(self):
        self.local_mp.reset_parameters()
        self.pe.reset_parameters()
        self.encoder.reset_parameters()
        for layer in self.readout:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, data: Data):
        x_clique,graph_emb, graph_readout = self.local_mp(data)
        x_clique = self.pe(x_clique=x_clique, data=data)
        if self.return_raw:
            out,raw,attn = self.encoder(x_clique=x_clique, data=data)
            return out,raw, graph_emb,attn
        tree_out = self.encoder(x_clique=x_clique, data=data)
        if self.binary_readout:
            out = torch.concat((tree_out, graph_readout), dim=1)
        else:
            out = tree_out

        out = self.readout(out)
        return out
