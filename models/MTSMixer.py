import torch
import torch.nn as nn
from layers.Invertible import RevIN

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)
    
    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim) :
        super().__init__()

        self.temporal_even = MLPBlock(input_dim, mlp_dim)
        self.temporal_odd = MLPBlock(input_dim, mlp_dim)
    
    def merge(self, even, odd):
        assert even.shape[1] == odd.shape[1]

        even = even.transpose(0, 1)
        odd = odd.transpose(0, 1)
        merge = []

        for i in range(even.shape[0]):
            merge.append(even[i].unsqueeze(0))
            merge.append(odd[i].unsqueeze(0))

        return torch.cat(merge, 0).transpose(0, 1)

    def forward(self, x):
        x_even = self.temporal_even(x[:, 0::2, :])
        x_odd = self.temporal_odd(x[:, 1::2, :])
        x = self.merge(x_odd, x_even)

        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim) :
        super().__init__()

        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):

        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim, fac_T, fac_C, norm_flag):
        super().__init__()
        self.tokens_mixing = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim) if fac_T else MLPBlock(tokens_dim, tokens_hidden_dim)
        self.channels_mixing = FactorizedChannelMixing(channels_dim, channels_hidden_dim) if fac_C else None
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self,x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        # channel-mixing [B, #tokens, D]
        if self.channels_mixing:
            y += x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(configs.seq_len, configs.enc_in, configs.d_model, configs.d_ff, configs.fac_T, configs.fac_C, configs.norm) for _ in range(configs.e_layers)
        ])
        self.norm = nn.LayerNorm(configs.enc_in) if configs.norm else None
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        self.rev = RevIN(configs.enc_in) if configs.rev else None

    def forward(self, x):
        x = self.rev(x, 'norm') if self.rev else x
        for block in self.mlp_blocks:
            x = block(x)
        
        x = self.norm(x) if self.norm else x
        x = self.projection(x.transpose(1, 2)).transpose(1, 2)
        x = self.rev(x, 'denorm') if self.rev else x

        return x
