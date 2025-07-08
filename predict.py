import os
import argparse
import pickle

import numpy as np

from datasets.dataset_utils import *
from models.CropTypeClassifier import *
from utils.classmetric import *
from datasets.sequence_aggregator import *

import torch
from torch.utils.data.sampler import RandomSampler
import torch.nn.functional as F
from models.LossFunctions import *
from sklearn.manifold import TSNE
import collections

from sklearn.cluster import KMeans

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', help='the chosen dataset', default="DENETHOR", choices=["BavarianCrops", "DENETHOR", "TimeSen2Crop"])
    parser.add_argument(
        '--dataset_folder', help='the root folder of the dataset', default="/home/luca/luca_docker/datasets/")
    parser.add_argument(
        '--classes_to_exclude', type=str, default=None, help='the classes to exclude during model training/testing')
    parser.add_argument(
        '--num_classes', type=int, default=9, help='the classmaping is selected based on the number of classes')
    parser.add_argument(
        # '--model_dir', help='the directory where the trained model is stored', default="/home/luca/luca_docker/results/crop-type-classification-explainability/paper/9_classes/ltae/right_padding/obs_aq_date/layers=1,heads=16,emb_dim=128/focal_loss_ratio=100/all_dates/1746306746") #1746219370
        '--model_dir', help='the directory where the trained model is stored', default="/home/luca/luca_docker/results/crop-type-classification-explainability/paper/9_classes/ltae/right_padding/obs_aq_date/concatenate_heads=False/layers=1,heads=32,emb_dim=1024,scale_dim=32/WCE,gamma=0.0/focal_loss_ratio=100/all_dates/1748852964") #1746219370
    parser.add_argument(
        '--seq_aggr', help='sequence aggregation method', default="right_padding",
        choices=["random_sampling", "fixed_sampling", "weekly_average", "right_padding"])
    parser.add_argument(
        '--pos_enc_opt', type=str, default="obs_aq_date", help='positional encoding method')
    parser.add_argument(
        '--time_points_to_sample', type=int, default=70,
        help='number of points to sample for the random and fixed sampling procedures')
    parser.add_argument(
        '--focal_loss_weights', type=str, default="1.0", help='Weighting choice alpha in [0,1] for the loss function: total_loss = alpha*focal_loss + (1-alpha)*rsme_loss. Multiple values should be separated with ,')
    parser.add_argument(
        '--num_layers', type=int, default=1, help='the number of layers for the model')
    parser.add_argument(
        '--num_heads', type=int, default=32, help='the number of heads in each layer of the model')
    parser.add_argument('--model_dim', type=int, default=1024, help='embedding dimension of the model')
    parser.add_argument('--use_lightweight', action="store_false", help='Choose encoder type: Transformer Encoder (default) or Lightweight Attention Encoder Transformer')
    parser.add_argument('--save_weights_and_gradients', action="store_false",
                        help='store the weights and gradients during test time')
    parser.add_argument('--save_key_queries_embeddings', action="store_false",
                        help='store the weights and gradients during test time')
    parser.add_argument('--shuffle_sequences', action="store_true", help='whether to shuffle sequences during training and test time')
    parser.add_argument('--most_important_dates_file', type=str, default=None, help='file which contains the most important days in the calendar year') # "key_attention_dates.csv"
    parser.add_argument('--fraction_of_important_dates_to_keep', type=float, default=0.2, help='fraction of the most important days to use for every parcel')

    args, _ = parser.parse_known_args()
    return args


def create_non_padding_mask(x, positions):
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


def extract_attn_weights_to_non_padded_indices(attn_weights_dir, use_lightweight=False, post_processed_attn_weights_dir=None, test_year="2018"):

    if post_processed_attn_weights_dir is None:
        post_processed_attn_weights_dir = os.path.join(attn_weights_dir, "postprocessed")

    if not os.path.exists(post_processed_attn_weights_dir):
        os.makedirs(post_processed_attn_weights_dir, exist_ok=True)

    observations_positions_files = [attn_weight_file
                         for attn_weight_file in os.listdir(attn_weights_dir)
                         if attn_weight_file.split(".")[-1] == "csv"]


    for dataset_sample_idx, observation_positions_file in enumerate(observations_positions_files):

        if dataset_sample_idx % 500 == 0:
            print('Post-processing the attention weights for the {}-th sample'.format(dataset_sample_idx))

        parcel_id = observation_positions_file.split("_")[0]
        positions = np.loadtxt(os.path.join(attn_weights_dir, observation_positions_file))
        non_padded_indices = positions != -1

        with open(os.path.join(attn_weights_dir, '{}_attn_weights.pickle'.format(parcel_id)), 'rb') as handle:
            parcel_attn_weights = pickle.load(handle)

        non_padded_attn_weights_tensors = dict()
        non_padded_attn_weights_dfs = dict()

        for layer_id in parcel_attn_weights.keys():
            attn_weights_per_layer = parcel_attn_weights[layer_id].cpu()
            
            observation_acqusition_day = positions[non_padded_indices]
            observation_acqusition_dates = pd.to_datetime(observation_acqusition_day, unit="D", origin=test_year).\
                map(lambda x: "{:02d}-{:02d}".format(x.day, x.month))            
            
            #filter out the attention to the padded columns
            attn_weights_per_layer = attn_weights_per_layer[:, :, non_padded_indices]
            if use_lightweight == False:
                #filter out the attention to the padded rows
                attn_weights_per_layer = attn_weights_per_layer[:, non_padded_indices, :]

                attn_weights_per_layer_df = pd.DataFrame(
                    data=attn_weights_per_layer.squeeze().detach().numpy(),
                    index=observation_acqusition_dates,
                    columns=observation_acqusition_dates)
            else:
                attn_weights_per_layer_df = pd.DataFrame(
                    data=attn_weights_per_layer.squeeze(dim=1).detach().numpy(),    # dim=0 for new LTAE, dim=1 for old LTAE
                    columns=observation_acqusition_dates)

            non_padded_attn_weights_tensors[layer_id] = attn_weights_per_layer
            non_padded_attn_weights_dfs[layer_id] = attn_weights_per_layer_df

        with open(os.path.join(post_processed_attn_weights_dir, "{}_attn_weights.pickle".format(parcel_id)),
                  "wb") as handle:
            pickle.dump(non_padded_attn_weights_tensors, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(post_processed_attn_weights_dir, "{}_attn_weights_df.pickle".format(parcel_id)),
                  "wb") as handle:
            pickle.dump(non_padded_attn_weights_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def track_attn_weights_gradient(attn_weights_by_layer):
    for attn_weights in attn_weights_by_layer.values():
        attn_weights.retain_grad()


def get_attn_weights_gradient(attn_weights_by_layer):
    attn_weights_grad_by_layer = {}
    for layer, attn_weights in attn_weights_by_layer.items():
        attn_weights_grad_by_layer[layer] = attn_weights.grad

    return attn_weights_grad_by_layer


def predict(
        test_dataset,
        crop_type_classifier_model,
        results_dir,
        loss_fn,
        use_lightweight,
        save_weights_and_gradients,
        save_key_queries_embeddings,
        test_year):

    #function so store keys and values
    def summarize_keys_and_queries(mod, inp, multi_head_attn_layer_output):

        def map_types(label):
            if label[0] == "Q":
                return "QUERY"
            else:
                return "KEY"

        parcel_positions = positions.squeeze()
        valid_positions = parcel_positions[parcel_positions != -1].tolist()

        queries = multi_head_attn_layer_output[0]
        keys = multi_head_attn_layer_output[1]
        queries = queries.squeeze()[parcel_positions != -1].cpu().detach().numpy()
        keys = keys.squeeze()[parcel_positions != -1].cpu().detach().numpy()

        keys_clusters = KMeans(n_clusters=2).fit_predict(keys)
        clusters = [-1] * len(valid_positions)
        clusters.extend(keys_clusters)

        observation_acqusition_dates = pd.to_datetime(valid_positions, unit="D", origin=test_year). \
            map(lambda x: "{:02d}-{:02d}".format(x.day, x.month))

        keys_and_queries = np.vstack((queries, keys))

        attn_before_softmax = np.matmul(queries, keys.T)
        max_key_indices = np.argsort(attn_before_softmax)[:, -5]
        most_common_keys_indices = [key_idx for key_idx, key_count in
                                    collections.Counter(max_key_indices).most_common(5)]

        query_labels = ["Q_{}".format(observation_acqusition_dates[i]) for i in range(len(valid_positions))]
        key_labels = ["K_{}".format(observation_acqusition_dates[i]) for i in range(len(valid_positions))]
        query_labels.extend(key_labels)

        keys_and_queries = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(keys_and_queries)
        keys_and_queries = pd.DataFrame(data=keys_and_queries, index=query_labels, columns=["emb_dim_1", "emb_dim_2"])
        keys_and_queries["TYPE"] = keys_and_queries.index.map(map_types)
        keys_and_queries["CLUSTER"] = clusters

        show_marker_text = [False] * keys_and_queries.shape[0]
        for key_idx in most_common_keys_indices:
            show_marker_text[len(valid_positions) + key_idx] = True
        keys_and_queries["SHOW_MARKER_TEXT"] = show_marker_text

        key_query_parcel_data[parcel_id.item()] = keys_and_queries

    print("Predicting on a test set...")
    predictions_path = os.path.join(results_dir, "predictions")

    model_path = os.path.join(results_dir, "best_model.pth")
    assert os.path.exists(model_path), 'The provided resulting directory does not contain the learned model'
    crop_type_classifier_model.load(model_path)
    crop_type_classifier_model.eval()

    #register the hook for storing self-attention query and key embeddings
    if save_key_queries_embeddings:
        for name, module in crop_type_classifier_model.named_modules():
            if name == "transformer_encoder.encoder_layers.0.inp_projection_layer":
                module.register_forward_hook(summarize_keys_and_queries)

    # os.mkdir(predictions_path)
    os.makedirs(predictions_path, exist_ok=True)
    attn_weights_dir = os.path.join(predictions_path, "attn_weights")
    # os.mkdir(attn_weights_dir)
    os.makedirs(attn_weights_dir, exist_ok=True)

    attn_weights_gradients_dir = os.path.join(predictions_path, "attn_weights_gradients")
    # os.mkdir(attn_weights_gradients_dir)
    os.makedirs(attn_weights_gradients_dir, exist_ok=True)
    
    classification_metric = ClassMetric()
    ndvi_metric = NDVIMetric()
    key_query_parcel_data = dict()


    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   sampler=RandomSampler(test_dataset),
                                                   batch_size=1,
                                                   num_workers=4)
    for dataset_sample_idx, batch_sample in enumerate(test_data_loader):

        crop_type_classifier_model.zero_grad()

        if dataset_sample_idx % 500 == 0:
            # if dataset_sample_idx == 1: break
            print('Crop type prediction for sample {}'.format(dataset_sample_idx))

        x, positions, y, y_ndvi, parcel_id = batch_sample
        non_padding_mask = create_non_padding_mask(x, positions)

        if torch.cuda.is_available():
            x = x.cuda() 
            y = y.cuda()
            y_ndvi = y_ndvi.cuda()
            positions = positions.cuda()

        log_probabilities, attn_weights_by_layer, ndvi_pred = crop_type_classifier_model(x, positions, non_padding_mask)

        track_attn_weights_gradient(attn_weights_by_layer)
        prediction = log_probabilities.exp().max()
        prediction.backward()
        attn_weights_gradient = get_attn_weights_gradient(attn_weights_by_layer)
        for layer_id in attn_weights_by_layer.keys():
            attn_weights_by_layer[layer_id] = attn_weights_by_layer[layer_id].detach().cpu()
            # print(attn_weights_by_layer[layer_id].shape)
            # print(attn_weights_by_layer[layer_id])
            # print(attn_weights_by_layer[layer_id].sum())
            # for hdz in range(attn_weights_by_layer[layer_id].shape[1]):
            #     print('Head', hdz)
            #     print(attn_weights_by_layer[layer_id][0,hdz,:].sum())

        if save_weights_and_gradients:
            observation_positions = positions.detach().cpu().numpy()
            np.savetxt(os.path.join(attn_weights_dir,"{}_prediction_positions.csv".format(parcel_id.item())),
                       observation_positions,
                       fmt="%s")
            with open(os.path.join(attn_weights_dir, "{}_attn_weights.pickle".format(parcel_id.item())), "wb") as handle:
                pickle.dump(attn_weights_by_layer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(attn_weights_gradients_dir, "{}_attn_weights_gradient.pickle".format(parcel_id.item())), "wb") as handle:
                pickle.dump(attn_weights_gradient, handle, protocol=pickle.HIGHEST_PROTOCOL)

        loss = loss_fn(log_probabilities, y[:, 0], ndvi_pred, y_ndvi).item()
        label = y.mode(1)[0].detach().cpu()
        prediction = crop_type_classifier_model.predict(log_probabilities).detach().cpu()

        classification_metric.add_batch_stats(parcel_id, loss, label, prediction)
        ndvi_metric.add_batch_stats(parcel_id, label, y_ndvi, ndvi_pred)

    classification_metric.save_results(predictions_path, test_dataset.get_class_names())
    ndvi_metric.save_results(predictions_path, test_dataset.get_class_names())

    if save_weights_and_gradients:
        extract_attn_weights_to_non_padded_indices(attn_weights_dir, use_lightweight=use_lightweight, test_year=test_year)
    if save_key_queries_embeddings:
        with open(os.path.join(attn_weights_dir, "keys_and_queries.pickle"), "wb") as handle:
            print("Saving keys and queries data for dataset parcels")
            pickle.dump(key_query_parcel_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = parse_args()

    dataset_folder = args.dataset_folder+args.dataset
    class_mapping = os.path.join(dataset_folder, "classmapping{}.csv".format(args.num_classes))
    sequence_aggregator = resolve_sequence_aggregator(args.seq_aggr, args.time_points_to_sample)

    if args.classes_to_exclude is not None:
        classes_to_exclude = [class_to_exclude for class_to_exclude in args.classes_to_exclude.split(',')]
    else:
        classes_to_exclude = None

    if args.dataset == 'BavarianCrops':
        _, _, test_dataset = get_partitioned_dataset_BavarianCrops(
        dataset_folder,
        class_mapping,
        sequence_aggregator,
        classes_to_exclude,
        args.most_important_dates_file,
        args.fraction_of_important_dates_to_keep,
        args.shuffle_sequences)
        input_channels = 13
        test_year = "2018"
    elif args.dataset == 'DENETHOR':
        train_dataset, _, test_dataset = get_partitioned_dataset_DenethorS2(
        dataset_folder,
        class_mapping,
        sequence_aggregator,
        classes_to_exclude,
        args.most_important_dates_file,
        args.fraction_of_important_dates_to_keep,
        args.shuffle_sequences)
        input_channels = 12
        test_year = "2019"
    else:
        _, _, test_dataset = get_partitioned_dataset_TimeSen2Crop(
        dataset_folder,
        class_mapping,
        sequence_aggregator,
        classes_to_exclude,
        args.most_important_dates_file,
        args.fraction_of_important_dates_to_keep,
        args.shuffle_sequences)
        input_channels = 9
        test_year = "2018"
    
    # if args.with_spectral_diff_as_input:
    #     input_channels = 12
    crop_type_classifier_model = init_model_with_hyper_params(
        input_channels,
        train_dataset[0][0].shape[0],
        train_dataset.nclasses,
        args.pos_enc_opt,
        args.model_dim,
        args.num_layers,
        args.num_heads,
        use_lightweight=args.use_lightweight,
        concatenate_heads=False,
        use_bias=True)

    predict(
        train_dataset,
        crop_type_classifier_model,
        args.model_dir,
        loss_fn=CombinedLoss(gamma=1.0, weight_focal=float(args.focal_loss_weights)),
        use_lightweight=args.use_lightweight,
        save_weights_and_gradients=args.save_weights_and_gradients,
        save_key_queries_embeddings=args.save_key_queries_embeddings,
        test_year="2018")




