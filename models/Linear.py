import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import PositionalEmbedding, TokenEmbedding

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.token = TokenEmbedding(c_in=configs.enc_in, d_model=configs.enc_in)
        self.pe = PositionalEmbedding(d_model=configs.enc_in)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.pe(x) + self.token(x)
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x # [Batch, Output length, Channel]
        