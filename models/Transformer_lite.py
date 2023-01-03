import torch
import torch.nn as nn
from layers.TransformerBlocks import Encoder
from layers.Embedding import DataEmbedding_wo_temp
from layers.Invertible import RevIN     

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, out_dim) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, out_dim)
    
    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, configs, sampling) :
        super().__init__()

        assert sampling <= 8
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            Encoder(
            configs.e_layers, configs.n_heads, configs.d_model, configs.d_ff, 
            configs.dropout, configs.activation, configs.output_attention,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
            ) for _ in range(sampling)
        ])

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, idx::self.sampling, :] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            feat, attn = samp(x[:, idx::self.sampling, :])
            x_samp.append(feat)

        x = self.merge(x.shape, x_samp)

        return x, None


class Model(nn.Module):
    """
    Transformer-lite w/o decoder
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.dropout)

        self.encoder = FactorizedTemporalMixing(configs, configs.sampling)

        # self.encoder = Encoder(
        #     configs.e_layers, configs.n_heads, configs.d_model, configs.d_ff, 
        #     configs.dropout, configs.activation, configs.output_attention,
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )

        self.channel_mixing = MLPBlock(configs.d_model, 64, configs.enc_in)
        # self.projection_channel = nn.Linear(configs.d_model, configs.enc_in)
        self.projection_temporal = nn.Linear(configs.seq_len, configs.pred_len)
        self.rev = RevIN(configs.enc_in) if configs.rev else None

    def forward(self, x):
        x = self.rev(x, 'norm') if self.rev else x

        x = self.enc_embedding(x, None)
        x, attns = self.encoder(x)
        x = self.projection_channel(x)
        x = self.projection_temporal(x.transpose(1, 2)).transpose(1, 2)

        x = self.rev(x, 'denorm') if self.rev else x

        if self.output_attention:
            pass

        return x
