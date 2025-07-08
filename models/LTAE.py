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

POS_ENC_OBS_AQ_DATE = "obs_aq_date"
POS_ENC_SEQ_ORDER = "seq_order"
POSITIONAL_ENCODING_OPTIONS = [POS_ENC_OBS_AQ_DATE, POS_ENC_SEQ_ORDER]

class LightEncoderLayer(nn.Module):

    def __init__(self, seq_len, num_heads, emb_dim, d_inner, concatenate_heads, dropout=0.2, d_k=8):
        super(LightEncoderLayer, self).__init__()

        self.attention_layer = MultiHeadAttention(
                            n_head=num_heads, 
                            d_k=d_k, 
                            d_in=emb_dim)

        self.concatenate_heads = concatenate_heads # If true, use LTAE source code
        
        if concatenate_heads:
            n_neurons = [emb_dim, d_inner]
            activation = nn.ReLU()
            print('Using MLP:', n_neurons)

            layers = []
            for i in range(len(n_neurons) - 1):
                layers.extend([nn.Linear(n_neurons[i], n_neurons[i + 1]),
                               nn.BatchNorm1d(n_neurons[i + 1]),
                               activation])
                
            self.mlp = nn.Sequential(*layers)
            self.outlayernorm = nn.LayerNorm(n_neurons[-1])

        else:
            assert d_inner == emb_dim // num_heads, "d_inner must be equal to the scaled embedded dimension E' (emb_dim // num_heads)"
            self.fc1 = nn.Linear(d_inner, d_inner)  # MLP per head
            self.batchnorm = nn.BatchNorm1d(d_inner)
            self.relu = nn.ReLU()

            self.fc2 = nn.Linear(d_inner, 1)  # Maxpool here instead?

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, attn_mask, non_padding_mask):

        sz_b, seq_len, d = x.shape
        attn_output, attn_weight = self.attention_layer(x, x, x, attn_mask=attn_mask)

        if self.concatenate_heads:
            output = attn_output.permute(1, 0, 2).contiguous().view(sz_b, -1)                   # Shape: BxE
            output = self.outlayernorm(self.dropout(self.mlp(output)))
        else: 
            # Our new implementation
            output = attn_output.permute(1, 0, 2).contiguous()                                  # Shape: BxHxE'
            output = self.relu(self.batchnorm(self.fc1(output).transpose(1,2))).transpose(1,2)
            output = self.fc2(self.dropout(output)).squeeze(-1)                                 # Shape: BxH

        return output, attn_weight


class LightTransformerEncoder(nn.Module):
    def __init__(self, pos_enc_opt, sequence_length, d_model, num_layers, num_heads, d_inner, concatenate_heads):

        super(LightTransformerEncoder, self).__init__()
        self.pos_enc_opt = pos_enc_opt
        self.num_heads = num_heads
        self.d_model = d_model
        self.concatenate_heads = concatenate_heads
        self.encoder_layers = nn.ModuleList(
            [LightEncoderLayer(seq_len=sequence_length, num_heads=num_heads, emb_dim=self.d_model, d_inner=d_inner, concatenate_heads=concatenate_heads) for _ in range(num_layers)])
        
        self.positional_encoding = PositionalEncoding(pos_enc_opt, self.d_model, self.num_heads)

        if concatenate_heads:
            self.scale_dim = d_model
        else:
            self.scale_dim = d_model // num_heads

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

        attn_weights_by_layer = {}

        output = self.positional_encoding(x, positions)
        # create the attention mask
        attn_mask = self.create_attention_mask(x, non_padding_mask)
        for i, encoder_layer in enumerate(self.encoder_layers):
            output, attn_weights = encoder_layer(output, attn_mask, non_padding_mask)
            attn_weights_by_layer["layer_{}".format(i)] = attn_weights
            
        return output, attn_weights_by_layer

    def get_label(self):
        return os.path.join(
            self.positional_encoding.pos_enc_opt,
            "concatenate_heads={}".format(self.concatenate_heads),
            "layers={},heads={},emb_dim={},scale_dim={}".format(len(self.encoder_layers), self.num_heads, self.d_model, self.scale_dim))

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        output, attn = self.attention(q, k, v, attn_mask)
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, attn_mask):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            attn.masked_fill_(attn_mask,
                    -1e8,
                )
        
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn
    

class PositionalEncoding(nn.Module):

    def __init__(self, pos_enc_opt, d_model, num_heads, max_sequence_length=365, T=1000, dropout=0.1):
        '''
        Positional encoding dependent on the observation acquisition date.

        :param pos_enc_opt:
        :param d_model:
        :param sequence_length:
        :param dropout:
        '''
        assert pos_enc_opt in POSITIONAL_ENCODING_OPTIONS, \
            "The positional encoding must be one of {}".format(",".join(POSITIONAL_ENCODING_OPTIONS))
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        self.pos_enc_opt = pos_enc_opt
        self.temperature = T
        self.max_sequence_length = max_sequence_length
        self.d_hid = d_model // num_heads

        sin_tab = self.get_sinusoid_encoding_table()
        self.positional_encoding = nn.Embedding.from_pretrained(torch.cat([sin_tab for _ in range(num_heads)], dim=1),
                                                         freeze=True)
        

    def get_sinusoid_encoding_table(self):
        ''' Sinusoid position encoding table
        positions: int or list of integer, if int range(positions)'''

        positions = list(range(self.max_sequence_length))

        def cal_angle(position, hid_idx):
            return position / np.power(self.temperature, 2 * (hid_idx // 2) / self.d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(self.d_hid)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        if torch.cuda.is_available():
            return torch.FloatTensor(sinusoid_table).cuda()
        else:
            return torch.FloatTensor(sinusoid_table)


    def get_sinusoid_encoding_table_var(self, clip=4, offset=3):
        ''' Sinusoid position encoding table
        positions: int or list of integer, if int range(positions)'''

        positions = list(range(self.max_sequence_length))

        x = np.array(positions)

        def cal_angle(position, hid_idx):
            return position / np.power(self.temperature, 2 * (hid_idx + offset // 2) / self.d_hid)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(self.d_hid)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

        sinusoid_table = np.sin(sinusoid_table)  # dim 2i
        sinusoid_table[:, clip:] = torch.zeros(sinusoid_table[:, clip:].shape)

        if torch.cuda.is_available():
            return torch.FloatTensor(sinusoid_table).cuda()
        else:
            return torch.FloatTensor(sinusoid_table)


    def forward(self, x, positions):

        for batch_elem_idx in range(x.shape[0]):
            non_padded_indices = positions[batch_elem_idx] != -1
            if self.pos_enc_opt == POS_ENC_OBS_AQ_DATE:
                src_pos = positions[batch_elem_idx, non_padded_indices].to(x.device)
            else:
                num_non_padded_obs = torch.sum(non_padded_indices)
                src_pos = torch.arange(0, num_non_padded_obs, dtype=torch.long).to(x.device)
            pos_enc = self.positional_encoding(src_pos)
            x[batch_elem_idx, non_padded_indices] += pos_enc

        return x
    

def get_decoder(n_neurons):
    """Returns an MLP with the layer widths specified in n_neurons.
    Every linear layer but the last one is followed by BatchNorm + ReLu

    args:
        n_neurons (list): List of int that specifies the width and length of the MLP.
    """
    layers = []
    for i in range(len(n_neurons)-1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        if i < (len(n_neurons) - 2):
            layers.extend([
                nn.BatchNorm1d(n_neurons[i + 1]),
                nn.ReLU()
            ])
    m = nn.Sequential(*layers)
    return m



if __name__=='__main__':
    x = torch.tensor([[[1,2]], [[1,2]]])
    y = torch.tensor([[[1,1], [1,1]], [[1,1], [1,1]]])
    print(torch.matmul(x,y).shape)
