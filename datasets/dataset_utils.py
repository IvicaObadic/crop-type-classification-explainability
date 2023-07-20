import os
import numpy as np
import time
import pandas as pd

from datasets.BavarianCrops_Dataset import BavarianCropsDataset
from datasets.ConcatDataset import ConcatDataset


AVAILABLE_REGIONS = ["holl", "krum", "nowa"]

def get_partitioned_dataset(
        root,
        class_mapping,
        sequence_aggregator,
        classes_to_exclude,
        most_important_dates_file=None,
        num_dates_to_consider=1,
        dates_removal=False,
        with_spectral_diff_as_input=False,
        target_regions=AVAILABLE_REGIONS):
    assert len(target_regions) > 0, 'At least one region must be supplied'

    train_datasets = []
    valid_datasets = []
    test_datasets = []


    raw_sequence_lengths = np.array([], dtype=np.int8)
    for region in target_regions:
        assert region in AVAILABLE_REGIONS, 'The region must be in one of the predefined regions'

        train_dataset = BavarianCropsDataset(root=root,
                                             partition="train",
                                             classmapping=class_mapping,
                                             region=region,
                                             sequence_aggregator=sequence_aggregator,
                                             classes_to_exclude=classes_to_exclude,
                                             scheme="blocks",
                                             most_important_dates_file = most_important_dates_file,
                                             num_dates_to_consider=num_dates_to_consider,
                                             dates_removal=dates_removal,
                                             with_spectral_diff_as_input=with_spectral_diff_as_input)
        valid_dataset = BavarianCropsDataset(root=root,
                                             partition="valid",
                                             classmapping=class_mapping,
                                             region=region,
                                             sequence_aggregator=sequence_aggregator,
                                             classes_to_exclude=classes_to_exclude,
                                             scheme="blocks",
                                             most_important_dates_file=most_important_dates_file,
                                             num_dates_to_consider=num_dates_to_consider,
                                             dates_removal=dates_removal,
                                             with_spectral_diff_as_input=with_spectral_diff_as_input)
        test_dataset = BavarianCropsDataset(root=root,
                                            partition="test",
                                            classmapping=class_mapping,
                                            region=region,
                                            sequence_aggregator=sequence_aggregator,
                                            classes_to_exclude=classes_to_exclude,
                                            scheme="blocks",
                                            most_important_dates_file=most_important_dates_file,
                                            num_dates_to_consider=num_dates_to_consider,
                                            dates_removal=dates_removal,
                                            with_spectral_diff_as_input=with_spectral_diff_as_input)

        raw_sequence_lengths = np.append(raw_sequence_lengths, train_dataset.sequencelengths)
        raw_sequence_lengths = np.append(raw_sequence_lengths, valid_dataset.sequencelengths)
        raw_sequence_lengths = np.append(raw_sequence_lengths, test_dataset.sequencelengths)

        train_datasets.append(train_dataset)
        valid_datasets.append(valid_dataset)
        test_datasets.append(test_dataset)


    np.save(os.path.join(root, "sequence_lengths.npy"), raw_sequence_lengths)
    max_sequence_length = np.amax(raw_sequence_lengths)

    print("Max sequence length is {}".format(max_sequence_length))

    train_dataset = ConcatDataset(train_datasets, max_sequence_length)
    valid_dataset = ConcatDataset(valid_datasets, max_sequence_length)
    test_dataset = ConcatDataset(test_datasets, max_sequence_length)

    return train_dataset, valid_dataset, test_dataset



def get_sequence_lengths_per_region(concat_datasets):
    """
    Gets the sequence lengths from array consisting of ConcatDataset elements
    """
    sequence_lenghts_per_region = dict()
    for concat_dataset in concat_datasets:
        for dataset in concat_dataset.datasets:
            sequence_lenghts = dataset.sequencelengths.tolist()
            if dataset.region not in sequence_lenghts_per_region.keys():
                sequence_lenghts_per_region[dataset.region] = list()
            sequence_lenghts_per_region[dataset.region].extend(sequence_lenghts)

    return sequence_lenghts_per_region
