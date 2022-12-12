import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.tokens_mixing = MLPBlock(configs.enc_in, mlp_dim=configs.d_model)
        self.norm = nn.LayerNorm(configs.enc_in)
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]

        # x = self.norm(x).transpose(1, 2)
        x = self.tokens_mixing(x.transpose(1, 2)).transpose(1, 2)
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)

        return x # [Batch, Output length, Channel]
        