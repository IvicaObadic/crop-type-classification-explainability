"""
Credits to https://github.com/MarcCoru/crop-type-mapping
"""

import torch
import torch.utils.data
import pandas as pd
import os
import json
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import random

from datasets.sequence_aggregator import *
from datasets.TimeSen2Crop_constants import *
from datasets.util_functions import *

class TimeSen2CropDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            root,
            partition,
            classmapping,
            region,
            sequence_aggregator,
            mode=None,
            classes_to_exclude=None,
            scheme="random",
            validfraction=0.1,
            most_important_dates_file=None,
            num_dates_to_consider=1,
            dates_removal=True,
            with_spectral_diff_as_input=False,
            shuffle=True,
            cache=True,
            seed=0):
        assert (mode in ["trainvalid", "traintest"] and scheme=="random") or (mode is None and scheme=="blocks") # <- if scheme random mode is required, else None
        assert scheme in ["random","blocks"]
        assert partition in ["train","test","trainvalid","valid"]
        assert isinstance(sequence_aggregator, SequenceAggregator)

        self.validfraction = validfraction
        self.scheme = scheme

        # ensure that different seeds are set per partition
        seed += sum([ord(ch) for ch in partition])
        self.seed = seed
        np.random.seed(self.seed)

        self.mode = mode

        self.root = root
        self.sequence_aggregator = sequence_aggregator

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        if classes_to_exclude is not None:
            self.mapping = self.mapping[~self.mapping["classname"].isin(classes_to_exclude)]
            crop_type_names = self.mapping["classname"].unique().tolist()
            self.mapping["id"] = self.mapping["classname"].map(lambda crop_type: crop_type_names.index(crop_type))

        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.nclasses = len(self.classes)
        self.region = region
        self.partition = partition
        self.most_important_dates = self.get_most_important_dates(most_important_dates_file)
        self.num_dates_to_consider = num_dates_to_consider
        self.dates_removal = dates_removal
        self.with_spectral_diff_as_input = with_spectral_diff_as_input
        self.shuffle = shuffle

        self.data_folder = "{root}/csv/{region}".format(root=self.root, region=self.region)

        print("Initializing TimeSen2CropDataset {} partition in {}".format(self.partition, self.region))

        self.cache = os.path.join(
            self.root,
            "npy",
            "{}_classes".format(self.nclasses))
        self.cache = append_occluded_classes_label(self.cache, classes_to_exclude)
        self.cache = append_spectral_diff_label(self.cache, self.with_spectral_diff_as_input)
        self.cache = os.path.join(self.cache, scheme, region, partition)

        self.acquisition_dates = pd.read_csv(os.path.join(self.data_folder, 'dates.csv'))['acquisition_date']

        if self.most_important_dates is not None:
            key_dates_usage = "kept"
            if self.dates_removal:
                key_dates_usage = "removed"
            self.cache = os.path.join(self.cache,
                                      "{}_dates_{}".format(num_dates_to_consider, key_dates_usage))

        print("read {} classes".format(self.nclasses))

        # self.read_ids = self.read_partition_ids()

        if cache and self.cache_exists():
            print("precached dataset files found at " + self.cache)
            self.load_cached_dataset()
        else:
            print("no cached dataset found. iterating through csv folders in " + str(self.data_folder))
            self.cache_dataset()

        print(self)

    def __str__(self):
        base_description = "Dataset {}. region {}. partition {}.".format(self.root, self.region, self.partition)
        instance_statistics = "No observations for the selected dates"
        if len(self.X) > 0:
            instance_statistics = "X:{}, y:{} with {} classes".format(
            str(len(self.X)) +"x"+ str(self.X[0].shape), self.y.shape, self.nclasses)

        return base_description + instance_statistics
    
    def read_partition_ids(self):

        partition_ids = []
        for cl in self.classes:
            class_folder = os.path.join(self.data_folder, str(cl))
            ids = [f'{self.region}_{cl}_{f[:-4]}' for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
            partition_ids += ids

        if self.shuffle: 
            random.shuffle(partition_ids)
        return partition_ids

    def cache_dataset(self):
        """
        Iterates though the data folders and stores y, ids, classweights, and sequencelengths
        X is loaded at with getitem
        """
        #ids = self.split(self.partition)

        ids = self.read_partition_ids()
        assert len(ids) > 0

        self.X = []
        X_dict = {}
        self.missing_key_dates_obs = list()
        self.stats = dict(
            not_found=list()
        )
        self.ids = list()
        self.samples = list()
        self.labels = []
        self.flags = []

        for id_string in tqdm.tqdm(ids):
            
            id = int(id_string.split('_')[2])
            label = int(id_string.split('_')[1])
            id_file = f"{self.data_folder}/{label}/{id}.csv"

            if os.path.exists(id_file):
                self.samples.append(id_file)

                X, flags, parcel_missing_key_dates_obs = self.load(id_file)

                if X is not None and parcel_missing_key_dates_obs is not None:
                    self.X.append(X)
                    X_dict[id_string] = X.tolist()
                    self.labels.append(label)
                    self.flags.append(flags)
                    self.ids.append(id_string)
                    self.missing_key_dates_obs.append(parcel_missing_key_dates_obs)

            else:
                self.stats["not_found"].append(id_file)

        if len(self.X) == 0:
            self.y = -1
            self.sequencelengths = np.array([0])
            self.max_sequence_length = 0
            self.ndims = -1
            self.hist = -1
            self.classweights=-1
            self.flags = -1
        else:
            self.y = np.array(self.labels)
            self.sequencelengths = np.array([np.array(X).shape[0] for X in self.X])
            self.max_sequence_length = self.sequencelengths.max()
            self.ndims = self.X[0].shape[1]
            self.hist,_ = np.histogram(self.y, bins=self.nclasses)
            self.classweights = 1 / (self.hist + 1e-6)
            self.flags = np.array(self.flags)

        self.cache_variables(self.y, self.sequencelengths, self.ids, self.ndims, X_dict, self.classweights, self.flags, self.missing_key_dates_obs)


    def cache_variables(self, y, sequencelengths, ids, ndims, X_dict, classweights, flags, missing_key_dates_obs):
        os.makedirs(self.cache, exist_ok=True)
        # cache
        np.save(os.path.join(self.cache, "classweights.npy"), classweights)
        np.save(os.path.join(self.cache, "y.npy"), y)
        np.save(os.path.join(self.cache, "ndims.npy"), ndims)
        np.save(os.path.join(self.cache, "sequencelengths.npy"), sequencelengths)
        np.save(os.path.join(self.cache, "ids.npy"), ids)
        json.dump(X_dict, open(os.path.join(self.cache, "X.json"), 'w'))
        np.save(os.path.join(self.cache, "flags.npy"), flags)
        np.save(os.path.join(self.cache, "missing_key_dates_obs.npy"), missing_key_dates_obs)

    def load_cached_dataset(self):
        # load
        self.classweights = np.load(os.path.join(self.cache, "classweights.npy"))
        self.y = np.load(os.path.join(self.cache, "y.npy"))
        self.ndims = int(np.load(os.path.join(self.cache, "ndims.npy")))
        self.sequencelengths = np.load(os.path.join(self.cache, "sequencelengths.npy"))
        self.max_sequence_length = self.sequencelengths.max()
        # self.ids = np.load(os.path.join(self.cache, "ids.npy"))
        X_dict = json.load(open(os.path.join(self.cache, 'X.json')))
        self.X = []
        self.ids = []
        for id, X_values in X_dict.items():
            self.ids.append(id)
            self.X.append(np.array(X_values))
        self.missing_key_dates_obs = np.load(os.path.join(self.cache, "missing_key_dates_obs.npy"), allow_pickle=True)

    def cache_exists(self):
        weightsexist = os.path.exists(os.path.join(self.cache, "classweights.npy"))
        yexist = os.path.exists(os.path.join(self.cache, "y.npy"))
        ndimsexist = os.path.exists(os.path.join(self.cache, "ndims.npy"))
        sequencelengthsexist = os.path.exists(os.path.join(self.cache, "sequencelengths.npy"))
        idsexist = os.path.exists(os.path.join(self.cache, "ids.npy"))
        Xexists = os.path.exists(os.path.join(self.cache, "X.json"))
        return yexist and sequencelengthsexist and idsexist and ndimsexist and Xexists and weightsexist

    def clean_cache(self):
        os.remove(os.path.join(self.cache, "classweights.npy"))
        os.remove(os.path.join(self.cache, "y.npy"))
        os.remove(os.path.join(self.cache, "ndims.npy"))
        os.remove(os.path.join(self.cache, "sequencelengths.npy"))
        os.remove(os.path.join(self.cache, "ids.npy"))
        os.remove(os.path.join(self.cache, "X.json"))
        os.removedirs(self.cache)

    def load(self, csv_file):
        """['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'Flag']"""

        sample = pd.read_csv(csv_file, index_col=0).dropna().reset_index()
        sample.columns = BANDS + ['Flag']

        if sample.empty:
            return None, None, None

        flags = sample["Flag"].values
        sample["TIMESTAMP"] = pd.to_datetime(self.acquisition_dates, format='%Y%m%d').dt.date

        sample = sample.groupby('TIMESTAMP').mean().reset_index()
        sample[BANDS] = sample[BANDS] * NORMALIZING_FACTOR
        final_bands_to_use = BANDS
        if self.with_spectral_diff_as_input:
            sample[BANDS] = sample[BANDS].diff(axis=1)
            # the first column is NaN column because it has no previous element to calculate the difference
            sample = sample.dropna(axis=1)
            final_bands_to_use = BANDS[1:]

        missing_key_dates_obs = 0
        if self.most_important_dates is not None:
            obs_acq_dates_as_str = sample[["TIMESTAMP"]].astype(str)
            dates_to_consider = self.most_important_dates.iloc[0:self.num_dates_to_consider]
            if self.dates_removal:
                dates_to_consider = self.most_important_dates.iloc[self.num_dates_to_consider:]
            indices_to_take = obs_acq_dates_as_str[["TIMESTAMP"]].isin(dates_to_consider.index)
            sample = sample.loc[indices_to_take["TIMESTAMP"]]
            missing_key_dates_obs = len(indices_to_take.index) - len(sample.index)

        if sample.empty:
            return None, missing_key_dates_obs

        observation_dates = pd.DataFrame({
            "YEAR": sample["TIMESTAMP"].apply(lambda x: x.year),
            "MONTH": sample["TIMESTAMP"].apply(lambda x: x.month),
            "DAY":  sample["TIMESTAMP"].apply(lambda x: x.day)}).to_numpy(dtype=np.float64)

        #create the resulting dataset consisting of the date and the spectral band values
        #scale the values in range [0 (everything is absorbed) - 1 (everything is reflected)]
        reflectances = (sample[final_bands_to_use]).to_numpy(dtype=np.float64)
        X = np.concatenate((observation_dates, reflectances), axis=1)

        return X, flags, missing_key_dates_obs


    def update_max_sequence_length(self, max_sequence_length):
        """
        When the sequences are padded, the maximum sequence length needs to be computed
        across the different dataset partitions and across different regions.
        Sets the maximum sequence length required for the padding.
        :param max_sequence_length: the new max_sequence_length values
        :return:
        """
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        X = np.copy(self.X[idx])
        # print(X.shape)
        parcel_id = self.ids[idx]

        X, positions = self.sequence_aggregator.aggregate_sequence(parcel_id, X, self)
        y_ndvi = (X[:,6] - X[:,2]) / (X[:,6] + X[:,2] + 1e-05)

        y = np.array([self.y[idx]] * X.shape[0])  # repeat y for each entry in x

        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)
        y_ndvi = torch.from_numpy(y_ndvi).type(torch.FloatTensor)
        positions = torch.from_numpy(positions).type(torch.LongTensor)

        return X, positions, y, y_ndvi, parcel_id
    
    def plot_ndvi(self,idx):
        X, positions, y, y_ndvi, parcel_id = self.__getitem__(idx)
        
        ndvi_timeseries = np.arange(len(y_ndvi))
        plt.plot(ndvi_timeseries, y_ndvi, label='getitem')
        plt.title(f'Phenological cycle for sample id {parcel_id}')
    
    def get_most_important_dates(self, most_important_dates_file):
        if most_important_dates_file is None:
            return None

        most_important_dates_file = os.path.join(self.root, most_important_dates_file)
        assert os.path.exists(most_important_dates_file)

        most_important_dates = pd.read_csv(most_important_dates_file, index_col=0)
        return most_important_dates

    def calculate_spectral_indices(self):
        """
        Calculates the NDVI index for each parcel in the dataset.
        :return: A dictionary with keys representing the parcel ids
                 and the values being dataframes containing the spectral indices at each time point
        """
        print("Calculating spectral indices for dataset: {}_{}".format(self.region, self.partition))
        spectral_indices_dfs = []

        for idx in range(len(self)):
            parcel_id = self.ids[idx]
            parcel_reflectance = self.X[idx]

            parcel_reflectance = pd.DataFrame(
                parcel_reflectance,
                columns=DATE_COLUMN_NAMES + BANDS)

            parcel_reflectance = add_timestamp_column_from_date_columns(parcel_reflectance)
            parcel_reflectance.drop(["YEAR", "MONTH", "DATE"], axis=1, inplace=True)
            parcel_reflectance["NDVI"] = (parcel_reflectance[NEAR_INFRARED_BAND] - parcel_reflectance[VISIBLE_RED_BAND]) /\
                                         (parcel_reflectance[NEAR_INFRARED_BAND] + parcel_reflectance[VISIBLE_RED_BAND])
            parcel_reflectance["Crop type"] = self.classname[self.y[idx]]
            parcel_reflectance["PARCEL_ID"] = parcel_id
            parcel_reflectance.set_index("PARCEL_ID", inplace=True)

            spectral_indices_dfs.append(parcel_reflectance)

        parcel_spectral_indices = pd.concat(spectral_indices_dfs)
        parcel_spectral_indices["Date"] = pd.to_datetime(parcel_spectral_indices["TIMESTAMP"], format="%Y-%m-%d")
        return parcel_spectral_indices




if __name__ == "__main__":
    train_regions = ['33UXP', '33UWQ', '33UWP', '33UUP', '33TXN', '33TWN', '33TWM', '33TVM', '33TUN', '33TUM', '32TQT', '32TPT', '32TNT']
    val_regions = ['33TVN']
    test_regions = ['33UVP', '2019_33UVP']
    sequence_aggregator = resolve_sequence_aggregator("right_padding",
                                                      29,
                                                      None is not None)
    
    for train_reg in train_regions:
        train_dataset = TimeSen2CropDataset(root="/home/luca/luca_docker/datasets/TimeSen2Crop",
                                                partition="train",
                                                classmapping="/home/luca/luca_docker/datasets/TimeSen2Crop/classmapping.csv",
                                                region=train_reg,
                                                sequence_aggregator=sequence_aggregator,
                                                classes_to_exclude=None,
                                                scheme="blocks",
                                                most_important_dates_file=None,
                                                num_dates_to_consider=1,
                                                dates_removal=False,
                                                with_spectral_diff_as_input=False)

    # for val_reg in val_regions:
    #     test_dataset = TimeSen2CropDataset(root="/home/luca/luca_docker/datasets/TimeSen2Crop",
    #                                             partition="valid",
    #                                             classmapping="/home/luca/luca_docker/datasets/TimeSen2Crop/classmapping.csv",
    #                                             region=val_reg,
    #                                             sequence_aggregator=sequence_aggregator,
    #                                             classes_to_exclude=None,
    #                                             scheme="blocks",
    #                                             most_important_dates_file=None,
    #                                             num_dates_to_consider=1,
    #                                             dates_removal=False,
    #                                             with_spectral_diff_as_input=False)
        
    # for test_reg in test_regions:
    #     test_dataset = TimeSen2CropDataset(root="/home/luca/luca_docker/datasets/TimeSen2Crop",
    #                                             partition="test",
    #                                             classmapping="/home/luca/luca_docker/datasets/TimeSen2Crop/classmapping.csv",
    #                                             region=test_reg,
    #                                             sequence_aggregator=sequence_aggregator,
    #                                             classes_to_exclude=None,
    #                                             scheme="blocks",
    #                                             most_important_dates_file=None,
    #                                             num_dates_to_consider=1,
    #                                             dates_removal=False,
    #                                             with_spectral_diff_as_input=False)
    
    all_indices = train_dataset.calculate_spectral_indices()

    X, positions, y, y_ndvi, parcel_id = next(iter(train_dataset))
    parcel_indices = all_indices[:30]
    timestamps = np.arange(30)
    # parcel_indices = all_indices[str(parcel_id)]

    print(parcel_indices['NDVI'], parcel_indices['TIMESTAMP'])

    train_dataset.plot_ndvi(0)
    plt.plot(timestamps, parcel_indices['NDVI'], label='calculate_spectral_indices')
    plt.legend()
    plt.show()
    plt.savefig(f'ndvi_series_{parcel_id}.png')
    
    for i,train_batch in enumerate(train_dataset):
        print(train_batch[0].shape)
        print(train_batch[3].shape)
        if i==1:
            break