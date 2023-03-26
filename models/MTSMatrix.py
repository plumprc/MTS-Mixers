import torch
import torch.nn as nn
from layers.Invertible import RevIN

class Matrix(nn.Module):
    def __init__(self, seq_len, enc_in, mat, norm):
        super().__init__()
        self.temporal = nn.Parameter(torch.rand(seq_len, seq_len)) if mat == 0 else nn.Parameter(torch.eye(seq_len))
        self.channels = nn.Parameter(torch.rand(enc_in, enc_in)) if mat == 0 else nn.Parameter(torch.eye(enc_in, enc_in))
        self.norm = nn.LayerNorm(enc_in)
        self.acti = nn.GELU()
        self.is_norm = norm
    
    def forward(self, x):
        x = x + self.acti(torch.matmul(self.temporal, x))
        x = x + self.norm(torch.matmul(x, self.channels)) if self.is_norm else x + torch.matmul(x, self.channels)

        return x


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, seq_len, enc_in, sampling, mat, norm) :
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            Matrix(seq_len // sampling, enc_in, mat, norm) for _ in range(sampling)
        ])

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, idx::self.sampling, :] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, idx::self.sampling, :]))

        x = self.merge(x.shape, x_samp)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.matrics = nn.ModuleList([
            FactorizedTemporalMixing(configs.seq_len, configs.enc_in, configs.sampling, configs.mat, configs.norm) for _ in range(configs.e_layers)
        ])
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        self.rev = RevIN(configs.enc_in) if configs.rev else None
    
    def forward(self, x):
        x = self.rev(x, 'norm') if self.rev else x
        for mat in self.matrics:
            x = mat(x)
        
        x = self.projection(x.transpose(1, 2)).transpose(1, 2)
        x = self.rev(x, 'denorm') if self.rev else x

        return x
