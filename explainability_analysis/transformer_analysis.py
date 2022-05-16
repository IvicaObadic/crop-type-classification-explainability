import os

import numpy as np
import scipy.stats as stats
import pandas as pd
import pickle
import torch
from torch.utils.data.sampler import RandomSampler
import collections
import argparse


from sklearn.cluster import KMeans
from sklearn.metrics import *
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from explainability_analysis.util import timeframe_columns
from models.CropTypeClassifier import *

from datasets import dataset_utils
from datasets import sequence_aggregator
from datasets.util_functions import *

from explainability_analysis.visualization_functions import *
from explainability_analysis.crop_spectral_signature_analysis import *


def summarize_attention_weights_as_feature_embeddings(
        attn_weights_root_dir,
        target_layer,
        target_head_idx=-1,
        summary_fn="sum"):

    feature_embeddings = []

    attention_weights_files = [attn_weight_file
                               for attn_weight_file in os.listdir(attn_weights_root_dir)
                               if attn_weight_file.split("_")[-1] == "weights.pickle"]

    for i, attn_weight_file in enumerate(attention_weights_files):
        parcel_id = attn_weight_file.split("_")[0]

        if i % 1000 == 0:
            print("Reading the attention weights for the {}-th test example".format(i))

        with open(os.path.join(attn_weights_root_dir, attn_weight_file), 'rb') as handle:
            attn_weights_by_layer = pickle.load(handle)

        with open(os.path.join(attn_weights_root_dir, '{}_attn_weights_df.pickle'.format(parcel_id)), 'rb') as handle:
            attn_weights_df = pickle.load(handle)[target_layer]

        #first resolve the target layers
        if target_layer != "all":
            relevant_attention_weights = attn_weights_by_layer[target_layer]
            num_heads = relevant_attention_weights.shape[0]
        else:
            relevant_attention_weights = None
            num_heads = None
            for layer_idx, layer_key in enumerate(sorted(attn_weights_by_layer.keys())):
                weights_for_a_layer = attn_weights_by_layer[layer_key]
                if relevant_attention_weights is None:
                    relevant_attention_weights = weights_for_a_layer
                    num_heads = weights_for_a_layer.shape[0]
                else:
                    relevant_attention_weights = torch.cat((relevant_attention_weights, weights_for_a_layer))


        #resolve the target heads
        if target_head_idx != -1:
            relevant_indices = np.arange(start=target_head_idx, stop=relevant_attention_weights.shape[0], step=num_heads)
        else:
            relevant_indices = np.arange(0, relevant_attention_weights.shape[0], step=1)

        if summary_fn == "sum":
            feature_embeddings_for_parcel = relevant_attention_weights[relevant_indices].sum(dim=1)
        else:
            feature_embeddings_for_parcel = relevant_attention_weights[relevant_indices].mean(dim=1)


        feature_embeddings_for_parcel = feature_embeddings_for_parcel.cpu().detach().numpy()
        feature_embeddings_for_parcel = pd.DataFrame(index=[parcel_id],
                                                     data=feature_embeddings_for_parcel,
                                                     columns=attn_weights_df.columns)
        feature_embeddings_for_parcel = pd.melt(feature_embeddings_for_parcel, var_name="Date", value_name="Attention", ignore_index=False)
        feature_embeddings.append(feature_embeddings_for_parcel)

    print("Concatenating the attention weights of the different parcels into a single dataframe")
    return pd.concat(feature_embeddings)


def get_temporal_attn_weights(root_results_path, classes_to_exclude=None, with_spectral_diff_as_input=False):
    model_classes = 12
    if classes_to_exclude is not None:
        classes_to_exclude = [class_to_exclude for class_to_exclude in classes_to_exclude.split(',')]
        model_classes = model_classes - len(classes_to_exclude)

    model_conf_path = "{}/{}_classes/".format(root_results_path, model_classes)
    model_conf_path = append_occluded_classes_label(model_conf_path, classes_to_exclude)
    model_conf_path = append_spectral_diff_label(model_conf_path, with_spectral_diff_as_input)
    model_conf_path = os.path.join(model_conf_path, "right_padding/obs_aq_date/layers=1,heads=1,emb_dim=128/all_dates/")
    model_path = os.path.join(model_conf_path, os.listdir(model_conf_path)[0])

    predictions_path = os.path.join(model_path, "predictions")
    attn_weights_path = os.path.join(predictions_path, "attn_weights", "postprocessed")
    total_temporal_attention_per_parcel_file = os.path.join(attn_weights_path, "parcel_temporal_attention.csv")
    if os.path.exists(total_temporal_attention_per_parcel_file):
        print("Reading the precomputed attention weights from {}".format(total_temporal_attention_per_parcel_file))
        return pd.read_csv(total_temporal_attention_per_parcel_file, index_col=0)

    predicted_vs_true_results = pd.read_csv(os.path.join(predictions_path, "predicted_vs_true.csv"), index_col=0)
    predicted_vs_true_results.index = predicted_vs_true_results.index.map(str)
    classmapping = pd.read_csv(os.path.join(predictions_path, "confusion_matrix.csv")).columns
    predicted_vs_true_results["TRUE_CROP_TYPE"] = predicted_vs_true_results["LABEL"].apply(lambda x: classmapping[x])
    predicted_vs_true_results["PREDICTED_CROP_TYPE"] = predicted_vs_true_results["PREDICTION"].apply(lambda x: classmapping[x])

    total_temporal_attention_per_parcel = summarize_attention_weights_as_feature_embeddings(attn_weights_path,
                                                                                            "layer_0",
                                                                                             summary_fn="sum")
    total_temporal_attention_per_parcel = total_temporal_attention_per_parcel.join(predicted_vs_true_results, how="inner")
    total_temporal_attention_per_parcel["Date"] = pd.to_datetime(
        total_temporal_attention_per_parcel["Date"].apply(lambda x: "{}-2018".format(x)), format="%d-%m-%Y")
    total_temporal_attention_per_parcel.drop(["LABEL", "PREDICTION"], axis=1, inplace=True)

    total_temporal_attention_per_parcel.to_csv(total_temporal_attention_per_parcel_file)
    return total_temporal_attention_per_parcel

def calc_and_save_weekly_average_attn_weights(attn_weights_feature_embeddings_dict, root_dir_path=None):
    weekly_average_attn_weights_per_parcel = dict()

    for parcel_id in attn_weights_feature_embeddings_dict.keys():
        feature_embeddings_for_parcel = attn_weights_feature_embeddings_dict[parcel_id]
        feature_embeddings_for_parcel = pd.melt(
            feature_embeddings_for_parcel,
            var_name="OBS_AQ_DATE",
            value_name="TOTAL_ATTENTION")
        feature_embeddings_for_parcel["OBS_AQ_DATE"] = pd.to_datetime(
            feature_embeddings_for_parcel["OBS_AQ_DATE"].map(
            lambda obs_aq_date: "2018/{}/{}".format(obs_aq_date.split("-")[1], obs_aq_date.split("-")[0])))
        feature_embeddings_for_parcel["WEEK"] = feature_embeddings_for_parcel["OBS_AQ_DATE"].dt.isocalendar().week.map(
            lambda week: timeframe_columns[week - 1])
        average_total_weekly_attention = feature_embeddings_for_parcel.groupby("WEEK").mean()
        average_total_weekly_attention.index = average_total_weekly_attention.index.astype('str')
        weekly_average_attn_weights_per_parcel[parcel_id] = average_total_weekly_attention
        if root_dir_path is not None:
            average_total_weekly_attention.to_csv(os.path.join(root_dir_path, "{}.csv".format(parcel_id)))

    return weekly_average_attn_weights_per_parcel


def calc_average_attention_weights_per_class(attn_weights_root_dir, predictions, target_class_label=-1, reduction="head"):

    assert reduction in ["head", "head_time_point", "time_point"], 'Invalid reduction parameter'

    if target_class_label != -1:
        predictions = predictions[predictions[:, 1] == target_class_label, :]

    relevant_parcel_ids = predictions[:,0].tolist()

    attn_weight_files = [attn_weight_file
                               for attn_weight_file in os.listdir(attn_weights_root_dir)
                               if attn_weight_file.split("_")[-1] == "weights.pickle"]


    relevant_attn_weight_files = [attn_weight_file
                                        for attn_weight_file in attn_weight_files
                                        if int(attn_weight_file.split("_")[0]) in relevant_parcel_ids]

    summarized_attn_weights_per_parcel = None
    for parcel_idx, relevant_attn_weight_file in enumerate(relevant_attn_weight_files):
        with open(os.path.join(attn_weights_root_dir, relevant_attn_weight_file), 'rb') as handle:
            attn_weights_by_layer = pickle.load(handle)

        attn_weights_per_parcel_layer = None
        for layer_id in attn_weights_by_layer.keys():
            attn_weights = attn_weights_by_layer[layer_id].detach().cpu()
            if reduction in ["head_time_point", "time_point"]:
                summarized_attn_weights_per_layer = attn_weights.sum(dim=1)
            else:
                summarized_attn_weights_per_layer = attn_weights

            summarized_attn_weights_per_layer = summarized_attn_weights_per_layer.unsqueeze(0)

            if attn_weights_per_parcel_layer is None:
                attn_weights_per_parcel_layer = summarized_attn_weights_per_layer
            else:
                attn_weights_per_parcel_layer = torch.cat(
                    (attn_weights_per_parcel_layer,
                    summarized_attn_weights_per_layer),
                    dim=0)

        attn_weights_per_parcel_layer = attn_weights_per_parcel_layer.unsqueeze(0)
        if summarized_attn_weights_per_parcel is None:
            summarized_attn_weights_per_parcel = attn_weights_per_parcel_layer
        else:
            summarized_attn_weights_per_parcel = torch.cat(
                (summarized_attn_weights_per_parcel,
                 attn_weights_per_parcel_layer),
                dim=0)

    if reduction == "time_point":
        summarized_attn_weights_per_parcel = summarized_attn_weights_per_parcel.sum(dim=2).sum(dim=1)

    mean_attention_weights = summarized_attn_weights_per_parcel.mean(dim=0).numpy()
    sd_attention_weights = summarized_attn_weights_per_parcel.std(dim=0, unbiased=True).numpy()
    return mean_attention_weights, sd_attention_weights


def get_avg_attn_weights_and_sd(attn_weights_root_dir, predictions, class_names, reduction="head", norm_fn="min_max"):

    avg_attn_weights_per_class = {}
    sd_attn_weights_per_class = {}
    for class_idx in range(len(class_names)):
        class_name = class_names[class_idx]
        avg_attn_weights, sd_attn_weights = calc_average_attention_weights_per_class(
            attn_weights_root_dir,
            predictions,
            class_idx,
            reduction)

        if norm_fn == "min_max":
            avg_attn_weights = avg_attn_weights.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaler.fit(avg_attn_weights)
            avg_attn_weights = scaler.transform(avg_attn_weights).reshape(-1)
            print(avg_attn_weights.reshape(-1))

        avg_attn_weights_per_class[class_name] = avg_attn_weights
        sd_attn_weights_per_class[class_name] = sd_attn_weights

    return avg_attn_weights_per_class, sd_attn_weights_per_class


def calc_attn_weight_corr_per_crop_type(spectral_indices, attn_weights_feature_embeddings):
    crop_type_attn_weights_spectral_bands = dict()
    for parcel_id in spectral_indices.keys():
        parcel_spectral_index = spectral_indices[parcel_id]
        target_class = parcel_spectral_index["CLASS"][0]
        spectral_indices_for_parcel_corr = parcel_spectral_index.copy().drop(["YEAR", "MONTH", "DATE", "CLASS"],
                                                                             axis=1)
        spectral_indices_for_parcel_corr.rename(
            {"B1": "Coastal Aerosol", "B2": "BLUE", "B3": "GREEN", "B4": "RED", "B8": "Near Infrared"},
            inplace=True,
            axis=1)
        attn_weights_for_parcel = attn_weights_feature_embeddings[str(parcel_id)].to_numpy().T
        spectral_indices_for_parcel_corr[["ATTN_WEIGHT"]] = attn_weights_for_parcel

        if target_class not in crop_type_attn_weights_spectral_bands:
            crop_type_attn_weights_spectral_bands[target_class] = spectral_indices_for_parcel_corr
        else:
            crop_type_attn_weights_spectral_bands[target_class] = pd.concat(
                [crop_type_attn_weights_spectral_bands[target_class], spectral_indices_for_parcel_corr])

    crop_type_corr = dict()
    for target_class in crop_type_attn_weights_spectral_bands.keys():
        attn_spectral_bands_corr = crop_type_attn_weights_spectral_bands[target_class].corr()
        crop_type_corr[target_class] = attn_spectral_bands_corr

    return crop_type_corr








def calculate_weights_gradients_correlations(predictions_path, predictions):
    parcel_ids = []
    layer = []
    head = []
    correlation = []
    for parcel_id in predictions[:, 0]:
        with open(os.path.join(predictions_path, "attn_weights", '{}_attn_weights.pickle'.format(parcel_id)), 'rb') as handle:
            attn_weights_by_layer = pickle.load(handle)

        with open(os.path.join(predictions_path, "attn_weights_gradients", '{}_attn_weights_gradient.pickle'.format(parcel_id)),
                  'rb') as handle:
            attn_weights_gradient_by_layer = pickle.load(handle)

        for layer_id in attn_weights_by_layer.keys():
            attn_weights = attn_weights_by_layer[layer_id].detach().cpu().numpy()
            attn_weights_gradients = attn_weights_gradient_by_layer[layer_id].detach().cpu().numpy()

            for i in range(attn_weights.shape[0]):
                attn_weights_head = attn_weights[i]
                attn_gradients_head = attn_weights_gradients[i]
                ken_corr = stats.kendalltau(attn_weights_head, attn_gradients_head)[0]

                layer.append(layer_id)
                head.append(i)
                correlation.append(ken_corr)
                parcel_ids.append(parcel_id)

    result = pd.DataFrame(
        {'parcel_id': parcel_ids,
         'layer': layer,
         'head': head,
         'correlation': correlation
         })

    return result
