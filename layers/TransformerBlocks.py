import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.masking import TriangularCausalMask
import numpy as np
from math import sqrt

class Attention(nn.Module):
    def __init__(self, n_heads, mask_flag=True, attention_dropout=0.1, output_attention=False):
        super(Attention, self).__init__()

        self.n_heads = n_heads
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)

        scale = 1. / sqrt(queries.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                # attn_mask = TriangularCausalMask(B, L, device=queries.device)
                pass

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values).contiguous()

        if self.output_attention:
            return V.view(B, L, -1), A
        else:
            return V.view(B, L, -1), None


class PointwiseFeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.1, activation="relu"):
        super(PointwiseFeedForward, self).__init__()

        self.conv_1 = nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1)
        self.conv_2 = nn.Conv1d(in_channels=hidden_dim, out_channels=in_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.conv_1(x.transpose(-1, 1))
        x = self.dropout(self.activation(x))
        x = self.conv_2(x).transpose(-1, 1)

        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff=None, dropout=0.1, activation="relu", output_attention=False):
        super(EncoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.attention = Attention(n_heads, mask_flag=False, attention_dropout=dropout, output_attention=output_attention)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ffn = PointwiseFeedForward(d_model, d_ff, dropout, activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm_1(x)
        y = self.ffn(y)

        return self.norm_2(x + y), attn


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff=None, dropout=0.1, activation="relu", output_attention=False):
        super(DecoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.self_attention = Attention(n_heads, mask_flag=True, attention_dropout=dropout, output_attention=output_attention)
        self.cross_attention = Attention(n_heads, mask_flag=False, attention_dropout=dropout, output_attention=output_attention)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.ffn = PointwiseFeedForward(d_model, d_ff, dropout, activation)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm_1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm_2(x)
        y = self.ffn(y)

        return self.norm_3(x + y)


class Encoder(nn.Module):
    def __init__(self, e_layers, n_heads, d_model, d_ff, dropout, activation, output_attention=True, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(
            EncoderLayer(n_heads, d_model, d_ff, dropout, activation, output_attention) for _ in range(e_layers)
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Decoder(nn.Module):
    def __init__(self, d_layers, n_heads, d_model, d_ff, dropout, activation, output_attention, norm_layer=None):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            DecoderLayer(n_heads, d_model, d_ff, dropout, activation, output_attention) for _ in range(d_layers)
        )
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
