from typing import Tuple, Any

import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, ReLU
from torch.nn import TransformerEncoder as PyTorchTransformerEncoder
from torch.nn import TransformerEncoderLayer as PyTorchTransformerEncoderLayer
from torch.nn import MultiheadAttention, ReLU
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reset_model_weights(layer):
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()
    else:
        if hasattr(layer, 'children'):
            for child in layer.children():
                reset_model_weights(child)



class Encoder(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 padding_length: int,
                 num_encoder_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 activation: str,
                 batch_first: bool,
                 return_raw: bool = False,
                 ):
        super().__init__()

        self.d_model = d_model
        self.nhead = n_head
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = ReLU()
        self.batch_first = batch_first
        self.padding_length = padding_length

        self.encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, activation,
                                                     batch_first=batch_first)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.cls_token = torch.nn.Parameter(torch.randn(1, d_model))


        self.reset_parameters()

        self.return_raw = return_raw

    def reset_parameters(self):
        for layer in self.encoder.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        reset_model_weights(self.encoder)

    def forward(self, x_clique: torch.Tensor, data: Data) -> torch.Tensor:

        tree_batch = torch.repeat_interleave(data.batch, data.num_cliques.tolist(), dim=0)
        src, mask = to_dense_batch(x_clique, batch=tree_batch)
        cls_token = self.cls_token.expand(src.shape[0], -1, -1)
        src = torch.cat([cls_token, src], dim=1)
        mask = torch.cat([torch.ones(src.shape[0], 1).bool().to(self.device), mask], dim=1)
        output = self.encoder(src, src_key_padding_mask=~mask)
        output = output[:, 0, :]

        return output
