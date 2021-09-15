import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from datasets.dataset_utils import *
from datasets.sequence_aggregator import *
from models.CropTypeClassifier import *
from models.LossFunctions import *

from utils.scheduled_optimizer import ScheduledOptim
from utils.logger import Logger
from utils.trainer import Trainer
from predict import *

import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_folder', help='the root folder of the dataset')
    parser.add_argument(
        '--num_classes', type=int, default=12, help='the classmaping is selected based on the number of classes')
    parser.add_argument(
        '--results_root_dir', help='the directory where the results are stored')
    parser.add_argument(
        '--seq_aggr', help='sequence aggregation method', default="weekly_average",
        choices=["random_sampling", "fixed_sampling", "weekly_average", "right_padding"])
    parser.add_argument(
        '--pos_enc_opt', type=str, default="obs_aq_date", help='positional encoding method')
    parser.add_argument(
        '--time_points_to_sample', type=int, default=70, help='number of points to sample for the random and fixed sampling procedures')
    parser.add_argument(
        '--num_layers', type=str, default="3", help='the number of layers for the model. Multiple values should be separated with ,')
    parser.add_argument(
        '--num_heads', type=str, default="4", help='the number of heads in each layer of the model. Multiple values should be separated with ,')
    parser.add_argument('--model_dim', type=str, default="128", help='embedding dimension of the model. Multiple values should be separated with ,')
    parser.add_argument('--save_weights_and_gradients', action="store_true", help='store the weights and gradients during test time')

    args, _ = parser.parse_known_args()
    return args

def create_data_loader(dataset, batch_size=128, num_workers=4):
    return(torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=RandomSampler(dataset),
        batch_size=batch_size,
        num_workers=num_workers))

def train_and_evaluate_crop_classifier(args):

    dataset_folder = args.dataset_folder
    class_mapping = os.path.join(args.dataset_folder, "classmapping{}.csv".format(args.num_classes))

    sequence_aggregator = resolve_sequence_aggregator(args.seq_aggr, args.time_points_to_sample)

    num_layers_opts = [int(layer) for layer in args.num_layers.split(',')]
    num_heads_opts = [int(head) for head in args.num_heads.split(',')]
    model_dims_opts = [int(model_dim) for model_dim in args.model_dim.split(',')]

    for num_layers in num_layers_opts:
        for num_heads in num_heads_opts:
            for model_dim in model_dims_opts:
                for training_time in range(sequence_aggregator.get_num_training_times()):

                    train_dataset, valid_dataset, test_dataset = get_partitioned_dataset(
                        dataset_folder,
                        class_mapping,
                        sequence_aggregator)

                    sequence_aggregator.set_timestamp(int(time.time()))

                    #all pixels contain sequences of same length
                    sequence_length = train_dataset[0][0].shape[0]
                    crop_type_classifier = init_model_with_hyper_params(
                        sequence_length,
                        args.num_classes,
                        args.pos_enc_opt,
                        model_dim,
                        num_layers,
                        num_heads)

                    optimizer = ScheduledOptim(
                        torch.optim.Adam(
                            filter(lambda x: x.requires_grad, crop_type_classifier.parameters()),
                            betas=(0.9, 0.98), eps=1e-09, weight_decay=0.000413),
                        crop_type_classifier.d_model, 4000)

                    training_directory = os.path.join(
                        args.results_root_dir,
                        "{}_classes".format(args.num_classes),
                        sequence_aggregator.get_label(),
                        crop_type_classifier.get_label())

                    logger = Logger(modes=["train", "test"], rootpath=training_directory)
                    config = dict(
                        epochs=100,
                        store=training_directory,
                        test_every_n_epochs=5,
                        logger=logger,
                        optimizer=optimizer)

                    loss_fn = FocalLoss(gamma=1.0)
                    trainer = Trainer(crop_type_classifier,
                                      create_data_loader(train_dataset),
                                      create_data_loader(valid_dataset),
                                      loss_fn=loss_fn,
                                      **config)
                    logger = trainer.fit()

                    # stores all stored values in the rootpath of the logger
                    logger.save()

                    # evaluate on the test set
                    predict(test_dataset, crop_type_classifier, training_directory, loss_fn, args.save_weights_and_gradients)


if __name__ == '__main__':
    args = parse_args()
    train_and_evaluate_crop_classifier(args)
