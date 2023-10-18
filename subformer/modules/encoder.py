from typing import Tuple, Any

import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer, ReLU
from torch.nn import TransformerEncoder as PyTorchTransformerEncoder
from torch.nn import TransformerEncoderLayer as PyTorchTransformerEncoderLayer
from torch.nn import MultiheadAttention, ReLU
from torch_geometric.data import Data

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

    def forward(self, x_clique: torch.Tensor, data: Data) -> tuple[Any, Any]:

        tree_list = torch.split(x_clique, data.num_cliques.tolist())
        max_length = self.padding_length
        src_padded = []
        masks = []

        for src in tree_list:

            src = torch.cat([self.cls_token, src], dim=0)
            pad_length = max_length - len(src)
            mask = torch.zeros((1, max_length), dtype=torch.bool).to(device)
            mask[:, :] = True
            mask[:, 0:len(src)] = False
            padded_src = torch.cat([src, torch.zeros([pad_length, self.d_model], dtype=torch.float).to(device)],
                                   dim=0).to(device)
            src_padded.append(padded_src)
            masks.append(mask)

        src_padded = torch.stack(src_padded, dim=0).to(device)
        masks_padded = torch.stack(masks, dim=0).to(device).squeeze(1)

        src = src_padded
        mask = masks_padded

        output = self.encoder(src, src_key_padding_mask=mask)
        raw = output
        output = output[:, 0, :]

        return output
