import torch
import torch.nn as nn
from layers.TransformerBlocks import Encoder, Decoder
from layers.Embedding import DataEmbedding, DataEmbedding_wo_temp
from layers.Invertible import RevIN

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            configs.e_layers, configs.n_heads, configs.d_model, configs.d_ff, 
            configs.dropout, configs.activation, configs.output_attention,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            configs.d_layers, configs.n_heads, configs.d_model, configs.d_ff,
            configs.dropout, configs.activation, configs.output_attention,
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, configs.c_out)
        self.rev = RevIN(configs.c_out) if configs.rev else None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc = self.rev(x_enc, 'norm') if self.rev else x_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        dec_out = self.rev(dec_out, 'denorm') if self.rev else dec_out

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
