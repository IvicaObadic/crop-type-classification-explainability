import os
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


def track_attn_weights_gradient(attn_weights_by_layer):
    for attn_weights in attn_weights_by_layer.values():
        attn_weights.retain_grad()


def get_attn_weights_gradient(attn_weights_by_layer):
    attn_weights_grad_by_layer = {}
    for layer, attn_weights in attn_weights_by_layer.items():
        attn_weights_grad_by_layer[layer] = attn_weights.grad

    return attn_weights_grad_by_layer


def predict(test_dataset, crop_type_classifier_model, results_dir, loss_fn, save_weights_and_gradients = True):

    print("Predicting on a test set...")
    print(results_dir)
    model_path = os.path.join(results_dir, "best_model.pth")
    assert os.path.exists(model_path), 'The provided results_dir does not contain the learned model'

    predictions_path = os.path.join(results_dir, "predictions")
    if os.path.exists(predictions_path):
        return

    os.mkdir(predictions_path)

    attn_weights_dir = os.path.join(predictions_path, "attn_weights")
    os.mkdir(attn_weights_dir)

    attn_weights_gradients_dir = os.path.join(predictions_path, "attn_weights_gradients")
    os.mkdir(attn_weights_gradients_dir)

    classification_metric = ClassMetric()

    crop_type_classifier_model.load(model_path)
    crop_type_classifier_model.eval()

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        sampler=RandomSampler(test_dataset),
        batch_size=1,
        num_workers=4)

    for dataset_sample_idx, batch_sample in enumerate(test_data_loader):

        crop_type_classifier_model.zero_grad()

        if dataset_sample_idx % 500 == 0:
            print('Crop type prediction for sample {}'.format(dataset_sample_idx))

        x, positions, padded_indices, y, parcel_id = batch_sample
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            padded_indices = padded_indices.cuda()
            positions = positions.cuda()

        log_probabilities, attn_weights_by_layer = crop_type_classifier_model(x, positions, padded_indices)
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
    root = "C:/Users/Ivica Obadic/Desktop/Explainable Machine Learning in Earth Observations/Projects/EO_explainability_survey/Datasets/BavarianCrops/"
    num_classes = 12
    class_mapping = os.path.join(root, "classmapping{}.csv".format(num_classes))
    _,_, test_dataset = get_partitioned_dataset(root, class_mapping, SequencePadder())

    crop_type_classifier_model = init_model_with_default_hyper_params(
        test_dataset[0][0].shape[0],
        num_classes=num_classes)
    results_dir = "C:/Users/Ivica Obadic/EO_explainability_survey/training_results/bavarian_crops/pos_enc_obs_date/{}_classes/sampling/fixed_70_obs_1626980215/".format(num_classes)

    predict(
        test_dataset,
        crop_type_classifier_model,
        results_dir,
        loss_fn=FocalLoss(gamma=1.0),
        save_weights_and_gradients=False)




