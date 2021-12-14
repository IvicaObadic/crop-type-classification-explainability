import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.sampler import RandomSampler
import pickle
from datasets import dataset_utils
from datasets import sequence_aggregator


timeframe_columns = ['2018-01-01', '2018-01-08', '2018-01-15', '2018-01-22', '2018-01-29',
       '2018-02-05', '2018-02-12', '2018-02-19', '2018-02-26', '2018-03-05',
       '2018-03-12', '2018-03-19', '2018-03-26', '2018-04-02', '2018-04-09',
       '2018-04-16', '2018-04-23', '2018-04-30', '2018-05-07', '2018-05-14',
       '2018-05-21', '2018-05-28', '2018-06-04', '2018-06-11', '2018-06-18',
       '2018-06-25', '2018-07-02', '2018-07-09', '2018-07-16', '2018-07-23',
       '2018-07-30', '2018-08-06', '2018-08-13', '2018-08-20', '2018-08-27',
       '2018-09-03', '2018-09-10', '2018-09-17', '2018-09-24', '2018-10-01',
       '2018-10-08', '2018-10-15', '2018-10-22', '2018-10-29', '2018-11-05',
       '2018-11-12', '2018-11-19', '2018-11-26', '2018-12-03', '2018-12-10',
       '2018-12-17', '2018-12-24']

def get_labels_for_parcel_ids(parcel_ids, results_path, return_names=True):
    predictions_path = os.path.join(results_path, "predicted_vs_true.csv")
    confusion_matrix_path = os.path.join(results_path, "confusion_matrix.csv")

    predictions = np.loadtxt(predictions_path, skiprows=1, delimiter=",", dtype=np.uint)

    confusion_matrix = pd.read_csv(confusion_matrix_path)
    class_names = list(confusion_matrix.columns.values)
    parcels_labels = []
    for parcel_id in parcel_ids:
        predictions_parcel_ids = predictions[:, 0]

        label = predictions[predictions_parcel_ids == int(parcel_id), 1]
        if return_names:
            label = class_names[int(label)]

        parcels_labels.append(label)

    return parcels_labels


def get_parcel_ids_for_class_name(predictions_data, class_name, n_parcels = 10):
    predictions_for_a_class = predictions_data[predictions_data["CLASS_NAME"] == class_name].sample(n_parcels)
    return predictions_for_a_class["PARCEL_ID"]


def convert_attn_weight_dict_to_df(attn_weights_per_class_dict):
    attn_weights_per_class_result = dict()

    for class_name in attn_weights_per_class_dict.keys():
        attn_weights_per_class = attn_weights_per_class_dict[class_name]
        attn_weights_per_class_and_layer_dict = dict()
        for layer_idx in range(0, attn_weights_per_class.shape[0]):
            attn_weights_per_class_and_layer = attn_weights_per_class[layer_idx]
            for head_idx in range(0, attn_weights_per_class_and_layer.shape[0]):
                layer_and_head_key = "{}_head_{}".format(layer_idx, head_idx)
                attn_weights_per_class_and_layer_dict[layer_and_head_key] = attn_weights_per_class_and_layer[head_idx]

        attn_weights_per_class_and_layer_df = pd.DataFrame.from_dict(
            attn_weights_per_class_and_layer_dict,
            orient="index",
            columns=timeframe_columns)

        attn_weights_per_class_result[class_name] = attn_weights_per_class_and_layer_df

    return attn_weights_per_class_result