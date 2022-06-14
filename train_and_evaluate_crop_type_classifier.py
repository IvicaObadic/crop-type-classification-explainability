import os
import numpy
import random
import torch
from torch.utils.data import DataLoader

from datasets.dataset_utils import *
from datasets.util_functions import *
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
        '--dataset_folder', help='the root folder of the dataset', default="/home/datasets/BavarianCrops")
    parser.add_argument(
        '--classes_to_exclude', type=str, default=None, help='the classes to exclude during model training/testing')
    parser.add_argument(
        '--num_classes', type=int, default=12, help='the classmaping is selected based on the number of classes')
    parser.add_argument(
        '--results_root_dir', help='the directory where the results are stored', default="/home/results/crop-type-classification-explainability/")
    parser.add_argument(
        '--seq_aggr', help='sequence aggregation method', default="right_padding",
        choices=["random_sampling", "fixed_sampling", "weekly_average", "right_padding"])
    parser.add_argument(
        '--pos_enc_opt', type=str, default="obs_aq_date", help='positional encoding method')
    parser.add_argument(
        '--time_points_to_sample', type=int, default=70, help='number of points to sample for the random and fixed sampling procedures')
    parser.add_argument(
        '--num_layers', type=str, default="1", help='the number of layers for the model. Multiple values should be separated with ,')
    parser.add_argument(
        '--num_heads', type=str, default="1", help='the number of heads in each layer of the model. Multiple values should be separated with ,')
    parser.add_argument('--model_dim', type=str, default="128", help='embedding dimension of the model. Multiple values should be separated with ,')
    parser.add_argument('--save_weights_and_gradients', action="store_true", help='store the weights and gradients during test time')
    parser.add_argument('--save_key_queries_embeddings', action="store_true",
                        help='store the weights and gradients during test time')
    parser.add_argument('--most_important_dates_file', type=str, default=None, help='file which contains the most important days in the calendar year')
    parser.add_argument('--num_important_dates_to_keep', type=int, default=10, help='fraction of the most important days to use for every parcel')
    parser.add_argument('--with_spectral_diff_as_input', action="store_true", help='train the model with the spectral difference between the consecutive channels as input')
    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        #source: https://discuss.pytorch.org/t/torch-deterministic-algorithms-error/125200/6
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def create_data_loader(dataset, batch_size=128, num_workers=4):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)

def train_and_evaluate_crop_classifier(args):

    dataset_folder = args.dataset_folder

    class_mapping = os.path.join(dataset_folder, "classmapping{}.csv".format(args.num_classes))

    most_important_dates_file = args.most_important_dates_file
    num_important_dates_to_keep = args.num_important_dates_to_keep
    sequence_aggregator = resolve_sequence_aggregator(args.seq_aggr,
                                                      args.time_points_to_sample,
                                                      most_important_dates_file is not None)
    num_layers_opts = [int(layer) for layer in args.num_layers.split(',')]
    num_heads_opts = [int(head) for head in args.num_heads.split(',')]
    model_dims_opts = [int(model_dim) for model_dim in args.model_dim.split(',')]
    if args.classes_to_exclude is not None:
        classes_to_exclude = [class_to_exclude for class_to_exclude in args.classes_to_exclude.split(',')]
    else:
        classes_to_exclude = None

    for num_layers in num_layers_opts:
        for num_heads in num_heads_opts:
            for model_dim in model_dims_opts:
                for training_time in range(sequence_aggregator.get_num_training_times()):

                    train_dataset, valid_dataset, test_dataset = get_partitioned_dataset(
                        dataset_folder,
                        class_mapping,
                        sequence_aggregator,
                        classes_to_exclude,
                        most_important_dates_file,
                        num_important_dates_to_keep,
                        args.with_spectral_diff_as_input)

                    #all observation contain sequences of same length
                    sequence_length = train_dataset[0][0].shape[0]
                    num_classes = train_dataset.nclasses

                    set_seed()
                    crop_type_classifier = init_model_with_hyper_params(
                        sequence_length,
                        num_classes,
                        args.pos_enc_opt,
                        model_dim,
                        num_layers,
                        num_heads)

                    optimizer = ScheduledOptim(torch.optim.Adam(
                            filter(lambda x: x.requires_grad, crop_type_classifier.parameters()),
                            betas=(0.9, 0.98), eps=1e-09, weight_decay=0.000413),
                        crop_type_classifier.d_model, 4000)

                    with_most_important_dates = "all_dates"
                    if most_important_dates_file is not None:
                        with_most_important_dates = "{}_num_dates".format(num_important_dates_to_keep)

                    training_directory = os.path.join(args.results_root_dir, "{}_classes".format(num_classes))
                    training_directory = append_occluded_classes_label(training_directory, classes_to_exclude)
                    training_directory = append_spectral_diff_label(training_directory, args.with_spectral_diff_as_input)
                    training_directory = os.path.join(
                        training_directory,
                        sequence_aggregator.get_label(),
                        crop_type_classifier.get_label(),
                        with_most_important_dates,
                        str(int(time.time())))

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
                    predict(
                        test_dataset,
                        crop_type_classifier,
                        training_directory,
                        loss_fn,
                        args.save_weights_and_gradients,
                        args.save_key_queries_embeddings)


if __name__ == '__main__':
    args = parse_args()
    train_and_evaluate_crop_classifier(args)
