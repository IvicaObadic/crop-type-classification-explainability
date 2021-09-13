import os

import numpy as np
import scipy.stats as stats
import pandas as pd
import pickle
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import *
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


def summarize_attention_weights_as_feature_embeddings(attn_weights_root_dir, target_layer, target_head_idx=-1):

    parcel_ids = []
    feature_embeddings = []

    attention_weights_files = [attn_weight_file
                               for attn_weight_file in os.listdir(attn_weights_root_dir)
                               if attn_weight_file.split("_")[-1] == "weights.pickle"]

    for i, attn_weight_file in enumerate(attention_weights_files):

        if i % 5000 == 0:
            print("Reading the attention weights for the {}-th test example".format(i))

        with open(os.path.join(attn_weights_root_dir, attn_weight_file), 'rb') as handle:
            attn_weights_by_layer = pickle.load(handle)

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

        feature_embeddings_for_parcel = relevant_attention_weights[relevant_indices].sum(dim=1).flatten().tolist()
        feature_embeddings.append(feature_embeddings_for_parcel)
        parcel_ids.append(attn_weight_file.split("_")[0])

    return parcel_ids, np.array(feature_embeddings)


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


def cluster_and_evaluate_attention_features(
        attention_features,
        label_names_for_parcels,
        label_ids_for_parcels,
        num_classes,
        attention_features_label,
        experiments_root_dir):

    experiments_output_path = os.path.join(experiments_root_dir, attention_features_label)
    if not os.path.isdir(experiments_output_path):
        os.makedirs(experiments_output_path)

    print("Clustering the attention features")
    clustering_results = KMeans(num_classes).fit_predict(attention_features)

    true_class_vs_cluster_cm = confusion_matrix(y_true=label_ids_for_parcels, y_pred=clustering_results)

    np.savetxt(os.path.join(experiments_output_path, "confusion_matrix.csv"),
               true_class_vs_cluster_cm,
               fmt='%i',
               delimiter=",",
               comments="labels/cluster_assignments")

    row_ind, col_ind = linear_sum_assignment(true_class_vs_cluster_cm, maximize=True)

    accuracy = true_class_vs_cluster_cm[row_ind, col_ind].sum() / true_class_vs_cluster_cm.sum()
    per_class_accuracy = true_class_vs_cluster_cm[row_ind, col_ind] / true_class_vs_cluster_cm.sum(axis=1)

    print("Reducing the data to 2D for visualization")
    plot_data = TSNE(2).fit_transform(attention_features)
    plot_data = pd.DataFrame(plot_data, columns=["TSNE_DIM_1", "TSNE_DIM_2"])
    plot_data["CLUSTER"] = clustering_results
    plot_data["CLASS_NAME"] = label_names_for_parcels
    return accuracy, per_class_accuracy, plot_data


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


def extract_attn_weights_in_time_frame(attn_weights_per_class, target_classes, start_idx, end_idx):
    class_labels = []
    target_attn_weights = dict()
    for class_label in attn_weights_per_class.keys():
        class_attn_weights = attn_weights_per_class[class_label]
        if class_label in target_classes:
            for layer in class_attn_weights.keys():
                class_attn_weights_per_layer = class_attn_weights[layer]
                relevant_attn_weights = class_attn_weights_per_layer[:, :, start_idx:end_idx].detach().cpu().numpy()
                if layer not in target_attn_weights:
                    target_attn_weights[layer] = relevant_attn_weights
                else:
                    target_attn_weights[layer] = np.hstack((target_attn_weights[layer], relevant_attn_weights))

            class_labels.append(class_label)

    return target_attn_weights, class_labels

