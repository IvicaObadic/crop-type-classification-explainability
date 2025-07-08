"""
Lightweight Temporal Attention Encoder module

Credits:
The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
the Transformer served as code base for the present script.

paper: https://arxiv.org/abs/1706.03762
code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
import copy
from torchtext.nn.modules.multiheadattention import InProjContainer
from .TransformerEncoder import PositionalEncoding, PositionwiseFeedForward

POS_ENC_OBS_AQ_DATE = "obs_aq_date"
POS_ENC_SEQ_ORDER = "seq_order"
POSITIONAL_ENCODING_OPTIONS = [POS_ENC_OBS_AQ_DATE, POS_ENC_SEQ_ORDER]

class LightEncoderLayer(nn.Module):

    def __init__(self, seq_len, num_heads=4, emb_dim=128, d_inner=512, dropout=0.2, d_k=8):
        super(LightEncoderLayer, self).__init__()

        emb_prime = emb_dim // num_heads

        self.attention_layer = MultiHeadAttention(
                            n_heads=num_heads, 
                            d_k=d_k, 
                            d_in=emb_dim, 
                            seq_len=seq_len)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(emb_prime)

        self.pos_ffn = PositionwiseFeedForward(emb_prime, d_inner, dropout=dropout)

    def forward(self, x, attn_mask, non_padding_mask):
        
        attn_output, attn_weight = self.attention_layer(x, x, x, attn_mask=attn_mask)
        output = self.dropout(attn_output)

        output = self.layer_norm(output) # remove + x) TODO Q: keep x but downsize to B,T,E'?
        output *= non_padding_mask

        output = self.pos_ffn(output, non_padding_mask)
        return output, attn_weight


class LightTransformerEncoder(nn.Module):
    def __init__(self, pos_enc_opt, sequence_length, d_model, num_layers, num_heads, d_inner):

        super(LightTransformerEncoder, self).__init__()
        self.pos_enc_opt = pos_enc_opt
        self.num_heads = num_heads
        self.d_model = d_model
        self.encoder_layers = nn.ModuleList(
            [LightEncoderLayer(seq_len=sequence_length, num_heads=num_heads, emb_dim=self.d_model, d_inner=d_inner) for _ in range(num_layers)])
        
        self.positional_encoding = PositionalEncoding(pos_enc_opt, self.d_model)

    def create_attention_mask(self, x, non_padding_mask):
        """
        Creates the attention masks boolean tensor where
        True represent the padded elements which need to be ignored during the attention
        and 0 represent
        :param x:
        :param non_padding_mask:
        :return:
        """
        batch_size = x.shape[0]
        total_sequence_length = x.shape[1]

        attn_mask = torch.zeros(
            (batch_size*self.num_heads, 1, total_sequence_length),
            dtype=torch.int8)
        if torch.cuda.is_available():
            attn_mask = attn_mask.cuda()

        sequences_to_mask = (1 - non_padding_mask.squeeze(dim=-1)).bool()

        for sample_idx in range(0, batch_size):
            start_batch_idx = sample_idx * self.num_heads
            end_batch_idx = sample_idx * self.num_heads + self.num_heads
            sequence_elements_to_mask = sequences_to_mask[sample_idx]
            #prevent attention to the padded sequence elements
            attn_mask[start_batch_idx:end_batch_idx, :, sequence_elements_to_mask] = 1

        return attn_mask.bool()

    def forward(self, x, positions, non_padding_mask):

        sz_b, seq_len, d = x.shape

        attn_weights_by_layer = {}
   
        output = self.positional_encoding(x, positions)
        #vcreate the attention mask
        attn_mask = self.create_attention_mask(x, non_padding_mask)

        for i, encoder_layer in enumerate(self.encoder_layers):
            output, attn_weights = encoder_layer(output, attn_mask, non_padding_mask)
            attn_weights_by_layer["layer_{}".format(i)] = attn_weights
            
        return output, attn_weights_by_layer

    def get_label(self):
        return os.path.join(
            self.positional_encoding.pos_enc_opt,
            "layers={},heads={},emb_dim={}".format(len(self.encoder_layers), self.num_heads, self.d_model))


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_heads, d_k, d_in, seq_len):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_heads, d_k))).requires_grad_(True) 
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_heads * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v, attn_mask=None):
        
        
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_heads
        sz_b, seq_len, _ = q.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        v = v.view(n_head, sz_b, seq_len, -1).permute(1, 2, 0, 3)
        
        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        attn = attn.squeeze(dim=2).permute(0, 2, 1)  
        # print(attn)   

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    # Insert attention mask here (optional; later on)
    def forward(self, q, k, v, attn_mask):
        # sz_b = v.shape[0] // self.n_heads
        sz_b, seq_len, n_heads, emb_prime = v.shape
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            attn.masked_fill_(
                    attn_mask,
                    -1e8,
                )

        attn = attn.view(sz_b, n_heads, seq_len)#.permute(0, 3, 2, 1)
        attn = attn.transpose(1, 2).unsqueeze(dim=2)

        attn = self.softmax(attn)  
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v).squeeze(dim=2)

        return output, attn


if __name__=='__main__':
    x = torch.tensor([[[1,2]], [[1,2]]])
    y = torch.tensor([[[1,1], [1,1]], [[1,1], [1,1]]])
    print(torch.matmul(x,y).shape)
