import os

import numpy as np
import torch
import torch.nn as nn
from .TransformerEncoder import TransformerEncoder


class CropTypeClassifier(nn.Module):

    def __init__(
            self,
            input_channels,
            sequence_length,
            pos_enc_opt="obs_aq_date",
            d_model=256,
            num_layers=4,
            num_heads=8,
            d_inner=1024,
            num_classes=23):

        super(CropTypeClassifier, self).__init__()

        self.raw_input_norm = nn.LayerNorm(input_channels)
        self.inconv = torch.nn.Conv1d(input_channels, d_model, 1)
        self.convlayernorm = nn.LayerNorm(d_model)
        self.d_model = d_model

        self.transformer_encoder = TransformerEncoder(
            pos_enc_opt,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_inner=d_inner)

        self.max_pool_over_time = nn.MaxPool1d(int(sequence_length))

        self.outlinear = nn.Linear(d_model, num_classes, bias=False)

        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def create_non_padding_mask(self, x, positions):
        batch_size = x.shape[0]
        total_sequence_length = x.shape[1]

        non_padding_mask = torch.ones((batch_size, total_sequence_length), dtype=torch.float)
        for batch_elem_idx in range(batch_size):
            padded_indices_for_sample = positions[batch_elem_idx] == -1
            non_padding_mask[batch_elem_idx, padded_indices_for_sample] = 0

        non_padding_mask = torch.unsqueeze(non_padding_mask, -1)
        if torch.cuda.is_available():
            non_padding_mask = non_padding_mask.cuda()
        return non_padding_mask

    def forward(self, x, positions):
        """
        Normalizes the input dimension, increases the embedding dimension of the input tensor
        and forwards the input tensor to the transformer encoder.

        :param x: a tensor of shape (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIMENSION)
        :param positions: a vector specifying the number of days since the earliest observation in the dataset for
         every observation in the sequence
        :param padded_indices: a boolean tensor of shape (BATCH_SIZE, sequence_length) that indicates
               the padded elements of every sequence in the batch
        :return: tuple of log probabilities and attention weights for each layer and head of the transformer encoder
        """
        x = self.raw_input_norm(x)
        x = self.inconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.convlayernorm(x)

        non_padding_mask = self.create_non_padding_mask(x, positions)
        x *= non_padding_mask

        enc_output, attn_weights = self.transformer_encoder(x, positions, non_padding_mask)

        classifier_features = self.max_pool_over_time(enc_output.transpose(1, 2)).squeeze(-1)

        logits = self.outlinear(classifier_features)

        log_probabilities = self.logsoftmax(logits)

        return log_probabilities, attn_weights

    def predict(self, logprobabilities):
        return logprobabilities.argmax(-1)

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to "+path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state,**kwargs),path)

    def load(self, path):
        print("loading model from "+path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot

    def get_label(self):
        return self.transformer_encoder.get_label()


def init_model_with_hyper_params(
        sequence_length,
        num_classes,
        pos_enc_opt,
        d_model,
        num_layers,
        num_heads):

    d_inner = d_model * 4
    crop_type_classifier = CropTypeClassifier(
        input_channels=13,
        sequence_length=sequence_length,
        pos_enc_opt=pos_enc_opt,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_inner=d_inner,
        num_classes=num_classes)

    if torch.cuda.is_available():
        crop_type_classifier = crop_type_classifier.cuda()
    return crop_type_classifier