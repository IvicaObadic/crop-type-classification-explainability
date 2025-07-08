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


# torch.autograd.set_detect_anomaly(True)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', help='the chosen dataset', default="BavarianCrops", choices=["BavarianCrops", "DENETHOR", "TimeSen2Crop"])
    parser.add_argument(
        '--dataset_folder', help='the root folder of the dataset', default="/home/luca/luca_docker/datasets/")
    parser.add_argument(
        '--classes_to_exclude', type=str, default=None, help='the classes to exclude during model training/testing')
    parser.add_argument(
        '--num_classes', type=int, default=12, help='the classmaping is selected based on the number of classes')
    parser.add_argument(
        '--focal_loss_weights', type=str, default="1.0", help='Weighting choice alpha in [0,1] for the loss function: total_loss = alpha*focal_loss + (1-alpha)*rsme_loss. Multiple values should be separated with ,')
    parser.add_argument(
        '--loss_choice', type=str, default="WCE", help='Choice of loss-function', choices=['WCE', 'FL'])
    parser.add_argument(
        '--results_root_dir', help='the directory where the results are stored', default="/home/luca/luca_docker/results/crop-type-classification-explainability/paper/")
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
    parser.add_argument('--use_lightweight', action="store_false", help='Choose encoder type: Transformer Encoder (default) or Lightweight Attention Encoder Transformer')
    parser.add_argument('--concatenate_heads', action="store_true", help='Choose LTAE type: new implementation (default) or original')
    parser.add_argument('--use_biases', action="store_false", help='set the biases for the linear operations in the layers. Default: true')
    parser.add_argument('--save_weights_and_gradients', action="store_true", help='store the weights and gradients during test time')
    parser.add_argument('--save_key_queries_embeddings', action="store_true",
                        help='store the weights and gradients during test time')
    parser.add_argument('--most_important_dates_file', type=str, default=None, help='file which contains the most important days in the calendar year')
    parser.add_argument('--num_dates_to_consider', type=int, default=10, help='number of dates to consider use for every parcel')
    parser.add_argument('--dates_removal', action="store_true", help='whether to remove the considered dates or only keep them in the dataset')
    parser.add_argument('--with_spectral_diff_as_input', action="store_true", help='train the model with the spectral difference between the consecutive channels as input')
    parser.add_argument('--use_fixed_seed', action="store_true", help='whether to use fixed seed to initialize the model')
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

def create_data_loader(dataset, use_fixed_seed, batch_size=32, num_workers=4):
    if use_fixed_seed:
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    else:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

def train_and_evaluate_crop_classifier(args):

    dataset_folder = args.dataset_folder + args.dataset

    class_mapping = os.path.join(dataset_folder, "classmapping{}.csv".format(args.num_classes))  # Change this for TimeSen2Crop - num_classes = 16

    most_important_dates_file = args.most_important_dates_file
    num_dates_to_consider = args.num_dates_to_consider
    dates_removal = args.dates_removal
    sequence_aggregator = resolve_sequence_aggregator(args.seq_aggr,
                                                      args.time_points_to_sample,
                                                      most_important_dates_file is not None)
    num_layers_opts = [int(layer) for layer in args.num_layers.split(',')]
    num_heads_opts = [int(head) for head in args.num_heads.split(',')]
    model_dims_opts = [int(model_dim) for model_dim in args.model_dim.split(',')]
    focal_loss_weights = [float(weight) for weight in args.focal_loss_weights.split(',')]
    if args.classes_to_exclude is not None:
        classes_to_exclude = [class_to_exclude for class_to_exclude in args.classes_to_exclude.split(',')]
    else:
        classes_to_exclude = None

    for num_layers in num_layers_opts:
        for num_heads in num_heads_opts:
            for model_dim in model_dims_opts:
                for training_time in range(sequence_aggregator.get_num_training_times()):
                    for focal_loss_weight in focal_loss_weights:
                        
                        # Load requested dataset
                        if args.dataset == 'BavarianCrops':
                            train_dataset, valid_dataset, test_dataset = get_partitioned_dataset_BavarianCrops(
                                dataset_folder,
                                class_mapping,
                                sequence_aggregator,
                                classes_to_exclude,
                                most_important_dates_file,
                                num_dates_to_consider,
                                dates_removal,
                                args.with_spectral_diff_as_input)
                            input_channels = 13
                            test_year = "2018"
                        elif args.dataset == 'DENETHOR':
                            train_dataset, valid_dataset, test_dataset = get_partitioned_dataset_DenethorS2(
                                dataset_folder,
                                class_mapping,
                                sequence_aggregator,
                                classes_to_exclude,
                                most_important_dates_file,
                                num_dates_to_consider,
                                dates_removal,
                                args.with_spectral_diff_as_input)
                            input_channels = 12
                            test_year = "2019"
                        else:
                            train_dataset, valid_dataset, test_dataset = get_partitioned_dataset_TimeSen2Crop(
                                dataset_folder,
                                class_mapping,
                                sequence_aggregator,
                                classes_to_exclude,
                                most_important_dates_file,
                                num_dates_to_consider,
                                dates_removal,
                                args.with_spectral_diff_as_input)
                            input_channels = 9
                            test_year = "2018"

                        if args.with_spectral_diff_as_input:
                            input_channels -= 1

                        class_weights = torch.from_numpy(train_dataset.classweights).float()
                        if torch.cuda.is_available():
                            class_weights = class_weights.cuda()                            
                        
                        #all observation contain sequences of same length
                        sequence_length = train_dataset[0][0].shape[0]
                        num_classes = train_dataset.nclasses

                        if args.use_fixed_seed:
                            set_seed()
                        crop_type_classifier = init_model_with_hyper_params(
                            input_channels,
                            sequence_length,
                            num_classes,
                            args.pos_enc_opt,
                            model_dim,
                            num_layers,
                            num_heads,
                            use_lightweight=args.use_lightweight,
                            concatenate_heads=args.concatenate_heads,
                            use_bias=args.use_biases)

                        optimizer = ScheduledOptim(torch.optim.Adam(
                                filter(lambda x: x.requires_grad, crop_type_classifier.parameters()),
                                lr=1e-3,                            # TODO eventually change lr here
                                betas=(0.9, 0.98), eps=1e-09, weight_decay=0.1), # weight_decay=0.000413, 0.05
                            crop_type_classifier.d_model, 4000)

                        with_most_important_dates = "all_dates"
                        if most_important_dates_file is not None:
                            key_dates_usage = "kept"
                            if dates_removal:
                                key_dates_usage = "removed"
                            with_most_important_dates = "{}_dates_{}".format(num_dates_to_consider, key_dates_usage)

                        training_directory = os.path.join(args.results_root_dir, "{}_classes".format(num_classes))
                        training_directory = append_occluded_classes_label(training_directory, classes_to_exclude)
                        training_directory = append_spectral_diff_label(training_directory, args.with_spectral_diff_as_input)

                        print(f'\nStarting training for FL-ratio = {focal_loss_weight}')

                        encoder_dir = 'ltae' if args.use_lightweight else 'tae'
                        gamma = 0.0

                        training_directory = os.path.join(
                            training_directory,
                            encoder_dir,
                            sequence_aggregator.get_label(),
                            crop_type_classifier.get_label(),
                            f'{args.loss_choice},gamma={gamma}',
                            f'focal_loss_ratio={int(focal_loss_weight*100)}',
                            with_most_important_dates,
                            str(int(time.time())))

                        print('Saving training data in: {}'.format(training_directory))
                        logger = Logger(modes=["train", "test"], rootpath=training_directory)
                    
                        config = dict(
                            epochs=100,
                            store=training_directory,
                            loss_fn_weighting=focal_loss_weight,
                            test_every_n_epochs=5,
                            logger=logger,
                            optimizer=optimizer)

                        # total_loss_fn = FocalLoss(gamma=1.0)
                        total_loss_fn = CombinedLoss(fn=args.loss_choice, class_weights=class_weights,  gamma=gamma, weight_focal=focal_loss_weight)
                        trainer = Trainer(crop_type_classifier,
                                        create_data_loader(train_dataset, args.use_fixed_seed),
                                        create_data_loader(valid_dataset, args.use_fixed_seed),
                                        loss_fn=total_loss_fn,
                                        class_names=train_dataset.get_class_names(),
                                        **config)
                        logger = trainer.fit()

                        # stores all stored values in the rootpath of the logger
                        logger.save()

                        # evaluate on the test set
                        predict(
                            test_dataset,
                            crop_type_classifier,
                            training_directory,
                            total_loss_fn,
                            args.use_lightweight,
                            args.save_weights_and_gradients,
                            args.save_key_queries_embeddings,
                            test_year)


if __name__ == '__main__':

    args = parse_args()
    train_and_evaluate_crop_classifier(args)
