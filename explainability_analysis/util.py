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


def extract_attn_weights_to_non_padded_indices(attn_weights_dir, post_processed_attn_weights_dir=None):

    if post_processed_attn_weights_dir is None:
        post_processed_attn_weights_dir = os.path.join(attn_weights_dir, "postprocessed")

    if not os.path.exists(post_processed_attn_weights_dir):
        os.makedirs(post_processed_attn_weights_dir)

    observations_positions_files = [attn_weight_file
                         for attn_weight_file in os.listdir(attn_weights_dir)
                         if attn_weight_file.split(".")[-1] == "csv"]

    for dataset_sample_idx, observation_positions_file in enumerate(observations_positions_files):

        if dataset_sample_idx % 500 == 0:
            print('Post-processing the attention weights for the {}-th sample'.format(dataset_sample_idx))

        parcel_id = observation_positions_file.split("_")[0]
        positions = np.loadtxt(os.path.join(attn_weights_dir, observation_positions_file), dtype=int)
        non_padded_indices = positions != -1

        with open(os.path.join(attn_weights_dir, '{}_attn_weights.pickle'.format(parcel_id)), 'rb') as handle:
            parcel_attn_weights = pickle.load(handle)

        non_padded_attn_weights_tensors = dict()
        non_padded_attn_weights_dfs = dict()
        for layer_id in parcel_attn_weights.keys():
            attn_weights_per_layer = parcel_attn_weights[layer_id].cpu()
            #filter out the attention to the padded columns
            attn_weights_per_layer = attn_weights_per_layer[:, :, non_padded_indices]
            #filter out the attention to the padded rows
            attn_weights_per_layer = attn_weights_per_layer[:, non_padded_indices, :]

            non_padded_attn_weights_tensors[layer_id] = attn_weights_per_layer

            observation_acqusition_day = positions[non_padded_indices]
            observation_acqusition_dates = pd.to_datetime(observation_acqusition_day, unit="D", origin="2018").\
                map(lambda x: "{:02d}-{:02d}".format(x.day, x.month))

            attn_weights_per_layer_df = pd.DataFrame(
                data=attn_weights_per_layer.squeeze().detach().numpy(),
                index=observation_acqusition_dates,
                columns=observation_acqusition_dates)

            non_padded_attn_weights_dfs[layer_id] = attn_weights_per_layer_df

        with open(os.path.join(post_processed_attn_weights_dir, "{}_attn_weights.pickle".format(parcel_id)),
                  "wb") as handle:
            pickle.dump(non_padded_attn_weights_tensors, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(post_processed_attn_weights_dir, "{}_attn_weights_df.pickle".format(parcel_id)),
                  "wb") as handle:
            pickle.dump(non_padded_attn_weights_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    num_classes = 12
    dataset_folder = "C:/Users/datasets/BavarianCrops/"
    class_mapping = os.path.join(dataset_folder, "classmapping{}.csv".format(num_classes))
    _, _, test_set = dataset_utils.get_partitioned_dataset(
        dataset_folder,
        class_mapping,
        sequence_aggregator.SequencePadder(),
        None,
        shuffle_sequences=False)

    attn_weights_dir = "C:/Users/results/{}_classes/shuffled_sequences/right_padding/obs_aq_date/layers=1,heads=1,emb_dim=128/predictions/attn_weights".format(test_set.nclasses)
    extract_attn_weights_to_non_padded_indices(attn_weights_dir)