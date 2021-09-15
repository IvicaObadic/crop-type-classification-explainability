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

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_folder', help='the root folder of the dataset')
    parser.add_argument(
        '--num_classes', type=int, default=12, help='the classmaping is selected based on the number of classes')
    parser.add_argument(
        '--model_dir', help='the directory where the trained model is stored')
    parser.add_argument(
        '--seq_aggr', help='sequence aggregation method', default="weekly_average",
        choices=["random_sampling", "fixed_sampling", "weekly_average", "right_padding"])
    parser.add_argument(
        '--pos_enc_opt', type=str, default="obs_aq_date", help='positional encoding method')
    parser.add_argument(
        '--time_points_to_sample', type=int, default=70,
        help='number of points to sample for the random and fixed sampling procedures')
    parser.add_argument(
        '--num_layers', type=int, default=3, help='the number of layers for the model')
    parser.add_argument(
        '--num_heads', type=int, default=4, help='the number of heads in each layer of the model')
    parser.add_argument('--model_dim', type=int, default=128, help='embedding dimension of the model')
    parser.add_argument('--save_weights_and_gradients', action="store_true",
                        help='store the weights and gradients during test time')

    args, _ = parser.parse_known_args()
    return args


def track_attn_weights_gradient(attn_weights_by_layer):
    for attn_weights in attn_weights_by_layer.values():
        attn_weights.retain_grad()


def get_attn_weights_gradient(attn_weights_by_layer):
    attn_weights_grad_by_layer = {}
    for layer, attn_weights in attn_weights_by_layer.items():
        attn_weights_grad_by_layer[layer] = attn_weights.grad

    return attn_weights_grad_by_layer


def predict(test_dataset, crop_type_classifier_model, results_dir, loss_fn, save_weights_and_gradients=True):

    print("Predicting on a test set...")
    model_path = os.path.join(results_dir, "best_model.pth")
    assert os.path.exists(model_path), 'The provided results_dir does not contain the learned model'

    crop_type_classifier_model.load(model_path)
    crop_type_classifier_model.eval()

    predictions_path = os.path.join(results_dir, "predictions")
    if os.path.exists(predictions_path):
        return

    os.mkdir(predictions_path)

    attn_weights_dir = os.path.join(predictions_path, "attn_weights")
    os.mkdir(attn_weights_dir)

    attn_weights_gradients_dir = os.path.join(predictions_path, "attn_weights_gradients")
    os.mkdir(attn_weights_gradients_dir)

    classification_metric = ClassMetric()

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        sampler=RandomSampler(test_dataset),
        batch_size=1,
        num_workers=4)

    for dataset_sample_idx, batch_sample in enumerate(test_data_loader):

        crop_type_classifier_model.zero_grad()

        if dataset_sample_idx % 500 == 0:
            print('Crop type prediction for sample {}'.format(dataset_sample_idx))

        x, positions, y, parcel_id = batch_sample
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            positions = positions.cuda()

        log_probabilities, attn_weights_by_layer = crop_type_classifier_model(x, positions)
        track_attn_weights_gradient(attn_weights_by_layer)
        prediction = log_probabilities.exp().max()
        prediction.backward()
        attn_weights_gradient = get_attn_weights_gradient(attn_weights_by_layer)

        if save_weights_and_gradients:
            with open(os.path.join(attn_weights_dir, "{}_attn_weights.pickle".format(parcel_id.item())), "wb") as handle:
                pickle.dump(attn_weights_by_layer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(attn_weights_gradients_dir, "{}_attn_weights_gradient.pickle".format(parcel_id.item())), "wb") as handle:
                pickle.dump(attn_weights_gradient, handle, protocol=pickle.HIGHEST_PROTOCOL)

        loss = loss_fn(log_probabilities, y[:, 0]).item()
        label = y.mode(1)[0].detach().cpu()
        prediction = crop_type_classifier_model.predict(log_probabilities).detach().cpu()

        classification_metric.add_batch_stats(parcel_id, loss, label, prediction)

    classification_metric.save_results(predictions_path, test_dataset.get_class_names())


if __name__ == "__main__":
    args = parse_args()

    class_mapping = os.path.join(args.dataset_folder, "classmapping{}.csv".format(args.num_classes))
    sequence_aggregator = resolve_sequence_aggregator(args.seq_aggr, args.time_points_to_sample)
    _,_, test_dataset = get_partitioned_dataset(args.dataset_folder, class_mapping, sequence_aggregator)

    crop_type_classifier_model = init_model_with_hyper_params(
        test_dataset[0][0].shape[0],
        args.num_classes,
        args.pos_enc_opt,
        args.model_dim,
        args.num_layers,
        args.num_heads)

    predict(
        test_dataset,
        crop_type_classifier_model,
        args.model_dir,
        loss_fn=FocalLoss(gamma=1.0),
        save_weights_and_gradients=False)




