import os
import numpy as np
import time
import pandas as pd
import torch
from torch.utils.data import Subset
import matplotlib.pyplot as plt

from datasets.sequence_aggregator import *
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
from datasets.TimeSen2Crop_Dataset import TimeSen2CropDataset
from datasets.s2dataset import DENETHOR_S2Dataset
from datasets.ConcatDataset import ConcatDataset

# from sequence_aggregator import *
# # from BavarianCrops_Dataset import BavarianCropsDataset
# # from TimeSen2Crop_Dataset import TimeSen2CropDataset
# from s2dataset import DENETHOR_S2Dataset
# from ConcatDataset import ConcatDataset


BavCrop_AVAILABLE_REGIONS = ["holl", "krum", "nowa"]
Denethor_TRAINVAL_REGIONS = []
TimeSen2Crop_TRAIN_REGIONS = ['33UXP', '33UWQ', '33UWP', '33UUP', '33TXN', '33TWN', '33TWM', '33TVM', '33TUN', '33TUM', '32TQT', '32TPT', '32TNT']
TimeSen2Crop_VAL_REGIONS = ['33TVN']
TimeSen2Crop_TEST_REGIONS = ['33UVP']#, '2019_33UVP']

class TrainValSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # Copy all attributes from the original dataset
        self.__dict__.update(dataset.__dict__)

    def __getattr__(self, attr):
        # Delegate method calls to the original dataset if not found in Subset
        return getattr(self.dataset, attr)



def get_partitioned_dataset_BavarianCrops(
        root,
        class_mapping,
        sequence_aggregator,
        classes_to_exclude,
        most_important_dates_file=None,
        num_dates_to_consider=1,
        dates_removal=False,
        with_spectral_diff_as_input=False,
        target_regions=BavCrop_AVAILABLE_REGIONS):
    assert len(target_regions) > 0, 'At least one region must be supplied'

    train_datasets = []
    valid_datasets = []
    test_datasets = []


    raw_sequence_lengths = np.array([], dtype=np.int8)
    for region in target_regions:
        assert region in BavCrop_AVAILABLE_REGIONS, 'The region must be in one of the predefined regions'

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


def get_partitioned_dataset_DenethorS2(
        root,
        class_mapping,
        sequence_aggregator,
        classes_to_exclude,
        most_important_dates_file=None,
        num_dates_to_consider=1,
        dates_removal=False,
        with_spectral_diff_as_input=False,
        trainvalid_split=0.25):

    raw_sequence_lengths = np.array([], dtype=np.int8)
    train_region = 's2-utm-33N-18E-242N'
    test_region = 's2-utm-33N-17E-243N'

    trainvalid_dataset = DENETHOR_S2Dataset(root=root,
                                             partition="trainvalid",
                                             classmapping=class_mapping,
                                             region=train_region,
                                             sequence_aggregator=sequence_aggregator,
                                             classes_to_exclude=classes_to_exclude,
                                             most_important_dates_file = most_important_dates_file,
                                             num_dates_to_consider=num_dates_to_consider,
                                             dates_removal=dates_removal,
                                             with_spectral_diff_as_input=with_spectral_diff_as_input)    
                
    indices = list(range(len(trainvalid_dataset)))
    np.random.RandomState(0).shuffle(indices)
    split = int(np.floor(trainvalid_split * len(trainvalid_dataset)))
    train_indices, val_indices = indices[split:], indices[:split]

    train_seq_lengths = trainvalid_dataset.sequencelengths[indices[split:]]
    val_seq_lengths = trainvalid_dataset.sequencelengths[indices[:split]]

    raw_sequence_lengths = np.append(raw_sequence_lengths, train_seq_lengths)
    raw_sequence_lengths = np.append(raw_sequence_lengths, val_seq_lengths)

    train_subset = TrainValSubset(trainvalid_dataset, train_indices)
    val_subset = TrainValSubset(trainvalid_dataset, val_indices)

    test_dataset = DENETHOR_S2Dataset(root=root,
                                         partition="test",
                                         classmapping=class_mapping,
                                         region=test_region,
                                         sequence_aggregator=sequence_aggregator,
                                         classes_to_exclude=classes_to_exclude,
                                         most_important_dates_file = most_important_dates_file,
                                         num_dates_to_consider=num_dates_to_consider,
                                         dates_removal=dates_removal,
                                         with_spectral_diff_as_input=with_spectral_diff_as_input)   

    raw_sequence_lengths = np.append(raw_sequence_lengths, test_dataset.sequencelengths)

    np.save(os.path.join(root, "sequence_lengths.npy"), raw_sequence_lengths)
    max_sequence_length = np.amax(raw_sequence_lengths)

    print("Max sequence length is {}".format(max_sequence_length))

    train_dataset = ConcatDataset([train_subset], max_sequence_length)
    val_dataset = ConcatDataset([val_subset], max_sequence_length)
    test_dataset = ConcatDataset([test_dataset], max_sequence_length)

    return train_subset, val_subset, test_dataset



def get_partitioned_dataset_TimeSen2Crop(
        root,
        class_mapping,
        sequence_aggregator,
        classes_to_exclude,
        most_important_dates_file=None,
        num_dates_to_consider=1,
        dates_removal=False,
        with_spectral_diff_as_input=False,
        train_regions=TimeSen2Crop_TRAIN_REGIONS,
        valid_regions=TimeSen2Crop_VAL_REGIONS,
        test_regions=TimeSen2Crop_TEST_REGIONS):
    
    assert len(train_regions) > 0, 'At least one train region must be supplied'
    assert len(valid_regions) > 0, 'At least one valid region must be supplied'
    assert len(test_regions) > 0, 'At least one test region must be supplied'

    train_datasets = []
    valid_datasets = []
    test_datasets = []

    raw_sequence_lengths = np.array([], dtype=np.int8)
    for region in train_regions:
        assert region in TimeSen2Crop_TRAIN_REGIONS, 'The region must be in one of the predefined train regions'

        train_dataset = TimeSen2CropDataset(root=root,
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
        
        raw_sequence_lengths = np.append(raw_sequence_lengths, train_dataset.sequencelengths)
        train_datasets.append(train_dataset)

    for region in valid_regions:
        assert region in TimeSen2Crop_VAL_REGIONS, 'The region must be in one of the predefined val regions'

        valid_dataset = TimeSen2CropDataset(root=root,
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
        
        raw_sequence_lengths = np.append(raw_sequence_lengths, valid_dataset.sequencelengths)
        valid_datasets.append(valid_dataset)


    for region in test_regions:
        assert region in TimeSen2Crop_TEST_REGIONS, 'The region must be in one of the predefined test regions'

        test_dataset = TimeSen2CropDataset(root=root,
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

        raw_sequence_lengths = np.append(raw_sequence_lengths, test_dataset.sequencelengths)
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


if __name__ == "__main__":

    sequence_aggregator = resolve_sequence_aggregator("right_padding",
                                                        82,
                                                        None is not None)

    train_dataset, val_dataset, test_dataset = get_partitioned_dataset_DenethorS2(
                                                    root="/home/luca/luca_docker/datasets/DENETHOR",
                                                    class_mapping="/home/luca/luca_docker/datasets/DENETHOR/classmapping9.csv",
                                                    sequence_aggregator = sequence_aggregator,
                                                    classes_to_exclude=None,
                                                    most_important_dates_file=None,
                                                    num_dates_to_consider=1,
                                                    dates_removal=False,
                                                    with_spectral_diff_as_input=False,
                                                    trainvalid_split=0.25)
    
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    print('train classweights:', 1/train_dataset.classweights)
    print('val classweights:', 1/val_dataset.classweights)
    print('test classweights:', 1/test_dataset.classweights)

    X, positions, y, y_ndvi, parcel_id = train_dataset[0]

    for i in range(X.shape[1]):
        plt.plot(X[:, i], label='Band {}'.format(i + 1))
    plt.plot(y_ndvi, label='NDVI', linestyle='--', color='black')
    plt.xlabel('Time')
    plt.ylabel('Reflectance')
    plt.title('Time Series of Reflectance Values')
    plt.legend()
    plt.savefig('reflectance_time_series_train.png')