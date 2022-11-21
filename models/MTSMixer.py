import torch
import torch.nn as nn
from layers.Invertible import RevIN

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)
    
    def forward(self,x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim):
        super().__init__()
        self.tokens_mixing = MLPBlock(tokens_dim, mlp_dim=tokens_hidden_dim)
        self.channels_mixing = MLPBlock(channels_dim, mlp_dim=channels_hidden_dim)
        self.norm = nn.LayerNorm(channels_dim)

    def forward(self,x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x).transpose(1, 2)
        y = self.tokens_mixing(y)

        # channel-mixing [B, #tokens, D]
        y = y.transpose(1, 2) + x
        res = y
        y = self.norm(y)
        y = res + self.channels_mixing(y)

        return y


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mlp_blocks = nn.ModuleList([
            MixerBlock(configs.seq_len, configs.enc_in, configs.d_model, configs.d_ff) for _ in range(configs.e_layers)
        ])
        self.norm = nn.LayerNorm(configs.enc_in)
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        self.rev = RevIN(configs.enc_in)

    def forward(self, x):
        # options: tokenize [B, L, D]
        # x = self.embed(x)
        x = self.rev(x, 'norm')
        for block in self.mlp_blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.projection(x.transpose(1, 2)).transpose(1, 2)
        x = self.rev(x, 'denorm')

        return x
