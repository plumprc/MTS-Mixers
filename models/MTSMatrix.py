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


class Model(nn.Module):
    def __init__(self, configs) :
        super().__init__()
        self.temporal = nn.Parameter(torch.rand(configs.seq_len, configs.seq_len))
        # self.channels = nn.Parameter(torch.rand(configs.enc_in, configs.enc_in))
        self.channels_mixing = MLPBlock(configs.enc_in, mlp_dim=configs.d_model)
        self.norm = nn.LayerNorm(configs.enc_in)
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        self.rev = RevIN(configs.enc_in)
    
    def forward(self, x):
        # x = self.rev(x, 'norm')
        x = torch.matmul(self.temporal, x)
        # x = torch.matmul(x, self.channels)
        x = self.channels_mixing(x)
        x = self.norm(x)
        x = self.projection(x.transpose(1, 2)).transpose(1, 2)
        # x = self.rev(x, 'denorm')

        return x
