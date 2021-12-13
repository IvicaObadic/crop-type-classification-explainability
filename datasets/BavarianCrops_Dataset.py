"""
Credits to https://github.com/MarcCoru/crop-type-mapping
"""

import torch
import torch.utils.data
import pandas as pd
import os
import numpy as np
import tqdm

from datasets.sequence_aggregator import *
from datasets.constants import *
from datasets.util_functions import *

class BavarianCropsDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            root,
            partition,
            classmapping,
            region,
            sequence_aggregator,
            mode=None,
            classes_to_exclude=None,
            shuffle_sequences=False,
            scheme="random",
            cache=True,
            seed=0,
            validfraction=0.1):
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

        if scheme=="random":
            if mode == "traintest":
                self.trainids = os.path.join(self.root, "ids", "random", region+"_train.txt")
                self.testids = os.path.join(self.root, "ids", "random", region+"_test.txt")
            elif mode == "trainvalid":
                self.trainids = os.path.join(self.root, "ids", "random", region+"_train.txt")
                self.testids = None

            self.read_ids = self.read_ids_random
        elif scheme=="blocks":
            self.trainids = os.path.join(self.root, "ids", "blocks", region+"_train.txt")
            self.testids = os.path.join(self.root, "ids", "blocks", region+"_test.txt")
            self.validids = os.path.join(self.root, "ids", "blocks", region + "_valid.txt")

            self.read_ids = self.read_ids_blocks

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("nutzcode")
        if classes_to_exclude is not None:
            self.mapping = self.mapping[~self.mapping["classname"].isin(classes_to_exclude)]
            crop_type_names = self.mapping["classname"].unique().tolist()
            self.mapping["id"] = self.mapping["classname"].map(lambda crop_type: crop_type_names.index(crop_type))

        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.nclasses = len(self.classes)
        self.shuffle_sequences = shuffle_sequences
        self.region = region
        self.partition = partition
        self.data_folder = "{root}/csv/{region}".format(root=self.root, region=self.region)

        print("Initializing BavarianCropsDataset {} partition in {} with sequence shuffling = {}"
              .format(self.partition, self.region, self.shuffle_sequences))

        self.cache = os.path.join(self.root, "npy", str(self.nclasses), ",".join(self.classname), scheme, region, partition)

        print("read {} classes".format(self.nclasses))

        if cache and self.cache_exists() and not self.mapping_consistent_with_cache():
            self.clean_cache()

        if cache and self.cache_exists() and self.mapping_consistent_with_cache():
            print("precached dataset files found at " + self.cache)
            self.load_cached_dataset()
        else:
            print("no cached dataset found. iterating through csv folders in " + str(self.data_folder))
            self.cache_dataset()

        self.hist, _ = np.histogram(self.y, bins=self.nclasses)

        print(self)

    def __str__(self):
        return "Dataset {}. region {}. partition {}. Sequence shuffling = {} X:{}, y:{} with {} classes".format(
            self.root, self.region, self.partition, self.shuffle_sequences, str(len(self.X)) +"x"+ str(self.X[0].shape), self.y.shape, self.nclasses)

    def read_ids_random(self):
        assert isinstance(self.seed, int)
        assert isinstance(self.validfraction, float)
        assert self.partition in ["train", "valid", "test"]
        assert self.trainids is not None
        assert os.path.exists(self.trainids)

        np.random.seed(self.seed)

        """if trainids file provided and no testids file <- sample holdback set from trainids"""
        if self.testids is None:
            assert self.partition in ["train", "valid"]

            print("partition {} and no test ids file provided. Splitting trainids file in train and valid partitions".format(self.partition))

            with open(self.trainids,"r") as f:
                ids = [int(id) for id in f.readlines()]
            print("Found {} ids in {}".format(len(ids), self.trainids))

            np.random.shuffle(ids)

            validsize = int(len(ids) * self.validfraction)
            validids = ids[:validsize]
            trainids = ids[validsize:]

            print("splitting {} ids in {} for training and {} for validation".format(len(ids), len(trainids), len(validids)))

            assert len(validids) + len(trainids) == len(ids)

            if self.partition == "train":
                return trainids
            if self.partition == "valid":
                return validids

        elif self.testids is not None:
            assert self.partition in ["train", "test"]

            if self.partition=="test":
                with open(self.testids,"r") as f:
                    test_ids = [int(id) for id in f.readlines()]
                print("Found {} ids in {}".format(len(test_ids), self.testids))
                return test_ids

            if self.partition == "train":
                with open(self.trainids, "r") as f:
                    train_ids = [int(id) for id in f.readlines()]
                return train_ids

    def read_ids_blocks(self):
        assert self.partition in ["train", "valid", "test", "trainvalid"]
        assert os.path.exists(self.validids)
        assert os.path.exists(self.testids)
        assert os.path.exists(self.trainids)
        assert self.scheme == "blocks"
        assert self.mode is None

        def read(filename):
            with open(filename, "r") as f:
                ids = [int(id) for id in f.readlines()]
            return ids

        if self.partition == "train":
            ids = read(self.trainids)
        elif self.partition == "valid":
            ids = read(self.validids)
        elif self.partition == "test":
            ids = read(self.testids)
        elif self.partition == "trainvalid":
            ids = read(self.trainids) + read(self.validids)
        return ids

    def cache_dataset(self):
        """
        Iterates though the data folders and stores y, ids, classweights, and sequencelengths
        X is loaded at with getitem
        """
        #ids = self.split(self.partition)

        ids = self.read_ids()
        assert len(ids) > 0

        self.X = list()
        self.nutzcodes = list()
        self.stats = dict(
            not_found=list()
        )
        self.ids = list()
        self.samples = list()
        #i = 0
        for id in tqdm.tqdm(ids):

            id_file = self.data_folder+"/{id}.csv".format(id=id)
            if os.path.exists(id_file):
                self.samples.append(id_file)

                X, label = self.load(id_file)

                if X is not None and label is not None:
                    label = label[0]
                    if label in self.mapping.index:
                        self.X.append(X)
                        self.nutzcodes.append(label)
                        self.ids.append(id)
            else:
                self.stats["not_found"].append(id_file)

        self.y = self.applyclassmapping(self.nutzcodes)

        self.sequencelengths = np.array([np.array(X).shape[0] for X in self.X])
        assert len(self.sequencelengths) > 0
        self.max_sequence_length = self.sequencelengths.max()
        self.ndims = self.X[0].shape[1]
        self.hist,_ = np.histogram(self.y, bins=self.nclasses)
        self.classweights = 1 / self.hist

        self.cache_variables(self.y, self.sequencelengths, self.ids, self.ndims, self.X, self.classweights)

    def mapping_consistent_with_cache(self):
        # cached y must have the same number of classes than the mapping
        return True
        #return len(np.unique(np.load(os.path.join(self.cache, "y.npy")))) == self.nclasses

    def cache_variables(self, y, sequencelengths, ids, ndims, X, classweights):
        os.makedirs(self.cache, exist_ok=True)
        # cache
        np.save(os.path.join(self.cache, "classweights.npy"), classweights)
        np.save(os.path.join(self.cache, "y.npy"), y)
        np.save(os.path.join(self.cache, "ndims.npy"), ndims)
        np.save(os.path.join(self.cache, "sequencelengths.npy"), sequencelengths)
        np.save(os.path.join(self.cache, "ids.npy"), ids)
        #np.save(os.path.join(self.cache, "dataweights.npy"), dataweights)
        np.save(os.path.join(self.cache, "X.npy"), X)

    def load_cached_dataset(self):
        # load
        self.classweights = np.load(os.path.join(self.cache, "classweights.npy"))
        self.y = np.load(os.path.join(self.cache, "y.npy"))
        self.ndims = int(np.load(os.path.join(self.cache, "ndims.npy")))
        self.sequencelengths = np.load(os.path.join(self.cache, "sequencelengths.npy"))
        self.max_sequence_length = self.sequencelengths.max()
        self.ids = np.load(os.path.join(self.cache, "ids.npy"))
        self.X = np.load(os.path.join(self.cache, "X.npy"), allow_pickle=True)

    def cache_exists(self):
        weightsexist = os.path.exists(os.path.join(self.cache, "classweights.npy"))
        yexist = os.path.exists(os.path.join(self.cache, "y.npy"))
        ndimsexist = os.path.exists(os.path.join(self.cache, "ndims.npy"))
        sequencelengthsexist = os.path.exists(os.path.join(self.cache, "sequencelengths.npy"))
        idsexist = os.path.exists(os.path.join(self.cache, "ids.npy"))
        Xexists = os.path.exists(os.path.join(self.cache, "X.npy"))
        return yexist and sequencelengthsexist and idsexist and ndimsexist and Xexists and weightsexist

    def clean_cache(self):
        os.remove(os.path.join(self.cache, "classweights.npy"))
        os.remove(os.path.join(self.cache, "y.npy"))
        os.remove(os.path.join(self.cache, "ndims.npy"))
        os.remove(os.path.join(self.cache, "sequencelengths.npy"))
        os.remove(os.path.join(self.cache, "ids.npy"))
        #os.remove(os.path.join(self.cache, "dataweights.npy"))
        os.remove(os.path.join(self.cache, "X.npy"))
        os.removedirs(self.cache)

    def load(self, csv_file):
        """[, 'B1', 'B10', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
       'B8A', 'B9', 'QA10', 'QA20', 'QA60', 'doa', 'label', 'id']"""

        sample = pd.read_csv(csv_file, index_col=0)

        #the timestamp column has no column name in the csv file. Set the name and
        sample.rename_axis("TIMESTAMP", inplace=True)
        sample.reset_index(inplace=True)
        #extract the first timestamp
        sample["TIMESTAMP"] = sample["TIMESTAMP"].map(lambda timestamp: timestamp.split("_")[0])
        #convert the string timestamp column to a pandas datetime column
        sample["TIMESTAMP"] = pd.to_datetime(sample["TIMESTAMP"]).dt.date

        #subset only the relevant columns in the dataframe and remove rows with missing values
        columns_to_take = ["TIMESTAMP", "label"]
        columns_to_take.extend(BANDS)
        sample = sample[columns_to_take].dropna()

        if sample.empty:
            return None, None

        label = sample["label"].values
        sample = sample.groupby('TIMESTAMP').mean().reset_index()

        observation_dates = pd.DataFrame({
            "YEAR": sample["TIMESTAMP"].apply(lambda x: x.year),
            "MONTH": sample["TIMESTAMP"].apply(lambda x: x.month),
            "DAY":  sample["TIMESTAMP"].apply(lambda x: x.day)}).to_numpy(dtype=np.float64)
        #validate whether all observations are made in one calendar year
        assert np.all(observation_dates[:, 0] == observation_dates[0, 0]),\
            'The file {} contains observations for more than one year'.format(csv_file)

        #create the resulting dataset consisting of the date and the spectral band values
        #scale the values in range [0 (everything is absorbed) - 1 (everything is reflected)]
        reflectances = (sample[BANDS] * NORMALIZING_FACTOR).to_numpy(dtype=np.float64)
        X = np.concatenate((observation_dates, reflectances), axis=1)

        return X, label

    def applyclassmapping(self, nutzcodes):
        """uses a mapping table to replace nutzcodes (e.g. 451, 411) with class ids"""
        return np.array([self.mapping.loc[nutzcode]["id"] for nutzcode in nutzcodes])

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
        parcel_id = self.ids[idx]

        X, positions = self.sequence_aggregator.aggregate_sequence(parcel_id, X, self)

        y = np.array([self.y[idx]] * X.shape[0])  # repeat y for each entry in x

        if self.shuffle_sequences:
            shuffled_indices = np.random.permutation(X.shape[0])
            X = X[shuffled_indices]
            positions = positions[shuffled_indices]
            if str(parcel_id) in ["77137894", "75415581"]:
                print("Shuffled positions for parcel {}".format(parcel_id))
                print(positions)

        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.LongTensor)
        positions = torch.from_numpy(positions).type(torch.LongTensor)

        return X, positions, y, parcel_id

    def calculate_spectral_indices(self):
        """
        Calculates the NDVI index for each parcel in the dataset.
        :return: A dictionary with keys representing the parcel ids
                 and the values being dataframes containing the spectral indices at each time point
        """
        print("Calculating spectral indices for dataset: {}_{}".format(self.region, self.partition))
        spectral_indices = dict()

        for idx in range(len(self)):
            parcel_id = self.ids[idx]
            parcel_reflectance = self.X[idx]

            parcel_reflectance = pd.DataFrame(
                parcel_reflectance,
                columns=DATE_COLUMN_NAMES + BANDS)

            parcel_reflectance = add_timestamp_column_from_date_columns(parcel_reflectance)
            parcel_reflectance["NDVI"] = (parcel_reflectance[NEAR_INFRARED_BAND] - parcel_reflectance[VISIBLE_RED_BAND]) /\
                                         (parcel_reflectance[NEAR_INFRARED_BAND] + parcel_reflectance[VISIBLE_RED_BAND])
            parcel_reflectance["CLASS"] = self.classname[self.y[idx]]
            parcel_reflectance.set_index("TIMESTAMP", inplace=True)
            spectral_indices[parcel_id] = parcel_reflectance


        return spectral_indices
