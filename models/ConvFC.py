import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.conv = nn.Conv1d(in_channels=configs.seq_len, out_channels=configs.pred_len, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)

        return x
