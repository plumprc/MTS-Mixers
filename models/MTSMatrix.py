import torch
import torch.nn as nn
from layers.Invertible import RevIN

class Model(nn.Module):
    def __init__(self, configs) :
        super().__init__()
        self.temporal = nn.Parameter(torch.eye(configs.seq_len))
        self.channels = nn.Parameter(torch.rand(configs.enc_in, configs.enc_in))
        self.norm = nn.LayerNorm(configs.enc_in)
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        self.acti = nn.GELU()
        self.rev = RevIN(configs.enc_in) if configs.rev else None
    
    def forward(self, x):
        x = self.rev(x, 'norm') if self.rev else x
        x = x + self.acti(torch.matmul(self.temporal, x))
        x = x + self.norm(torch.matmul(x, self.channels))

        x = self.projection(x.transpose(1, 2)).transpose(1, 2)
        x = self.rev(x, 'denorm') if self.rev else x

        return x
