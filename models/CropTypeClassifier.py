import os

import numpy as np
import torch
import torch.nn as nn
from .TransformerEncoder import TransformerEncoder, NdviDecoder
from .LTAE import LightTransformerEncoder, get_decoder
# from .LTAEcopy import LightTransformerEncoder


class CropTypeClassifier(nn.Module):

    def __init__(
            self,
            input_channels,
            sequence_length,
            pos_enc_opt="obs_aq_date",
            d_model=256,
            num_layers=4,
            num_heads=8,
            num_classes=23, 
            d_inner_ndvi=64,
            use_lightweight=False,
            concatenate_heads=False,
            use_bias=True):

        super(CropTypeClassifier, self).__init__()

        self.raw_input_norm = nn.LayerNorm(input_channels)
        # self.inconv = torch.nn.Conv1d(input_channels, d_model, 1, bias=use_bias)
        self.inconv1 = torch.nn.Conv1d(input_channels, d_model // 2, 1, bias=use_bias)
        self.inconv2 = torch.nn.Conv1d(d_model // 2, d_model, 1, bias=use_bias)
        self.convlayernorm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.light = use_lightweight
        self.concatenate_heads = concatenate_heads

        if use_lightweight: 
            ltae_out_string = 'original' if concatenate_heads else 'newly implemented'
            print(f'Run training on {ltae_out_string} Lightweight Temporal Attention Encoder (LTAE)')
            
            if concatenate_heads:
                d_inner = d_model // 2
                decoder_neurons = [d_inner, 64, 32, num_classes]
            else:
                d_inner = d_model // num_heads
                decoder_neurons = [num_heads, num_heads, num_classes]

            print('decoder_neurons', decoder_neurons)

            self.transformer_encoder = LightTransformerEncoder(
                pos_enc_opt,
                sequence_length=sequence_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_inner=d_inner,
                concatenate_heads=concatenate_heads)

            self.decoder = get_decoder(decoder_neurons)
            # self.outlinear = nn.Linear(num_heads, num_classes, bias=use_bias)
            # self.ndvipredict = NdviDecoder(num_heads, d_inner_ndvi, seq_len, use_bias=use_bias)
            self.ndvipredict = NdviDecoder(num_heads, d_inner_ndvi, 1, use_bias=use_bias) # when predicting on att_weights

        else:
            print('Run training on standard Temporal Attention Encoder (TAE)')
            self.concatenate_heads = False
            d_inner = d_model*4 # TAE input

            self.transformer_encoder = TransformerEncoder(
                pos_enc_opt,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_inner=d_inner)

            self.max_pool_over_time = nn.MaxPool1d(int(sequence_length))
            self.decoder = nn.Linear(d_model, num_classes, bias=use_bias)
            self.ndvipredict = NdviDecoder(d_model, d_inner_ndvi, 1, use_bias=use_bias)

        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, positions, non_padding_mask):
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
        # x = self.inconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.inconv2(self.inconv1(x.transpose(1, 2))).transpose(1, 2)
        x = self.convlayernorm(x)

        x *= non_padding_mask

        enc_output, attn_weights = self.transformer_encoder(x, positions, non_padding_mask)

        if not self.light: # TAE
            classifier_features = self.max_pool_over_time(enc_output.transpose(1, 2)).squeeze(-1)
            ndvi_output = enc_output # Predict NDVI on the encoded output 
                                    # Q here: should i process the attention weights & predict on those instead?

        else:   #LTAE
            classifier_features = enc_output
            # ndvi_output = enc_output.unsqueeze(1) # Predict NDVI on the encoded output - makes no sense since output looses temporal information
            ndvi_output = attn_weights["layer_0"].permute(1, 2, 0) # Enforce NDVI prediction on the attention weights

        ndvi_pred = self.ndvipredict(ndvi_output, non_padding_mask) 
        
        logits = self.decoder(classifier_features)
                 
        log_probabilities = self.logsoftmax(logits)

        return log_probabilities, attn_weights, ndvi_pred


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
        input_channels,
        sequence_length,
        num_classes,
        pos_enc_opt,
        d_model,
        num_layers,
        num_heads,
        with_gpu=True,
        use_lightweight=False,
        concatenate_heads=False,
        use_bias=True):

    crop_type_classifier = CropTypeClassifier(
        input_channels=input_channels,
        sequence_length=sequence_length,
        pos_enc_opt=pos_enc_opt,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        num_classes=num_classes,
        use_lightweight=use_lightweight,
        concatenate_heads=concatenate_heads,
        use_bias=use_bias)

    if with_gpu and torch.cuda.is_available():
        crop_type_classifier = crop_type_classifier.cuda()

    print("Initialized the transformer encoder with the following parameters: {}"
          .format(crop_type_classifier.get_label()))
    return crop_type_classifier