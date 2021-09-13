import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.nn.modules.multiheadattention import *

POS_ENC_OBS_AQ_DATE = "obs_aq_date"
POS_ENC_SEQ_ORDER = "seq_order"
POSITIONAL_ENCODING_OPTIONS = [POS_ENC_OBS_AQ_DATE, POS_ENC_SEQ_ORDER]


class EncoderLayer(nn.Module):

    def __init__(self, num_heads=4, emb_dim=128, d_inner=512, dropout=0.2):
        super(EncoderLayer, self).__init__()

        self.inp_projection_layer = InProjContainer(
            nn.Linear(emb_dim, emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.Linear(emb_dim, emb_dim))

        self.attention_layer = MultiheadAttentionContainer(
            num_heads,
            self.inp_projection_layer,
            ScaledDotProduct(),
            nn.Linear(emb_dim, emb_dim),
            batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)

        self.pos_ffn = PositionwiseFeedForward(emb_dim, d_inner, dropout=dropout)

    def forward(self, x, attn_mask, non_padding_mask):
        attn_output, attn_weight = self.attention_layer(x, x, x, attn_mask=attn_mask)

        output = self.dropout(attn_output)
        output = self.layer_norm(output + x)
        output *= non_padding_mask

        output = self.pos_ffn(output, non_padding_mask)
        return output, attn_weight


class TransformerEncoder(nn.Module):

    def __init__(self, pos_enc_opt, d_model, num_layers, num_heads, d_inner):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(num_heads=num_heads, emb_dim=self.d_model, d_inner=d_inner) for _ in range(num_layers)])

        self.positional_encoding = PositionalEncoding(self.d_model, pos_enc_opt)

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
            (batch_size*self.num_heads, total_sequence_length, total_sequence_length),
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
        #create the attention mask
        attn_mask = self.create_attention_mask(x, non_padding_mask)
        for i, encoder_layer in enumerate(self.encoder_layers):
            output, attn_weights = encoder_layer(output, attn_mask, non_padding_mask)
            attn_weights_by_layer["layer_{}".format(i)] = attn_weights

        return output, attn_weights_by_layer

    def get_label(self):
        return os.path.join(
            self.positional_encoding.pos_enc_opt,
            "layers={},heads={},emb_dim={}".format(len(self.encoder_layers), self.num_heads, self.d_model))


class PositionalEncoding(nn.Module):

    def __init__(self, pos_enc_opt, d_model, max_sequence_length=365, dropout=0.1):
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
        self.dropout = nn.Dropout(p=dropout)
        self.pos_enc_opt = pos_enc_opt

        pe = torch.zeros(max_sequence_length, d_model)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, positions):

        for batch_elem_idx in range(x.shape[0]):
            non_padded_indices = positions[batch_elem_idx] != -1
            if self.pos_enc_opt == POS_ENC_OBS_AQ_DATE:
                pos_enc = torch.index_select(self.pe, 0, positions[batch_elem_idx, non_padded_indices])
            else:
                num_non_padded_obs = torch.sum(non_padded_indices)
                pos_enc = self.pe[:num_non_padded_obs]
            x[batch_elem_idx, non_padded_indices] += pos_enc

        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, non_padding_mask):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        output *= non_padding_mask
        return output