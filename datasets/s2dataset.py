" Creds to https://github.com/lukaskondmann/DENETHOR"

import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset
import zipfile
# from sh import gunzip
from glob import glob
import pickle
import sentinelhub # this import is necessary for pickle loading
import geopandas as gpd
import rasterio as rio
from rasterio import features
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.sequence_aggregator import *
from datasets.constants import *
from datasets.util_functions import *

# from sequence_aggregator import *
# from constants import *
# from util_functions import *


# CLASSES = ["Wheat", "Rye", "Barley", "'Oats", "Corn", "Oil Seeds", "Root Crops", "Meadows", "Forage Crops"]
# CROP_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]'


class DENETHOR_S2Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            root,
            partition,
            classmapping,
            sequence_aggregator,
            region='s2-utm-33N-18E-242N',
            classes_to_exclude=None,
            preprocess_pixels = 'mean',
            s2_transform = True,
            validfraction=0.1,
            most_important_dates_file=None,
            num_dates_to_consider=1,
            dates_removal=True,
            with_spectral_diff_as_input=False,
            cache=True,
            seed=0):

        assert partition in ["trainvalid","test"]
        assert isinstance(sequence_aggregator, SequenceAggregator)

        self.validfraction = validfraction

        # ensure that different seeds are set per partition
        seed += sum([ord(ch) for ch in partition])
        self.seed = seed
        np.random.seed(self.seed)

        self.data_transform = s2_transform
        self.preprocess_pixels = preprocess_pixels

        self.root = root
        self.sequence_aggregator = sequence_aggregator

        if partition == 'trainvalid':
            self.dataset_path = f"{root}/sentinel-2/{region}-2018"
            self.labelgeojson = f"{root}/brandenburg_crops_train_2018.geojson"
        elif partition == 'test':
            self.dataset_path = f"{root}/sentinel-2/{region}-2019"
            self.labelgeojson = f"{root}/brandenburg_crops_test_2019.geojson"

        timestampfile = f"{self.dataset_path}/timestamp.pkl"
        self.timestamps = np.array(pickle.load(open(timestampfile,"rb")))
        
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

        print("Initializing DENETHORS2Dataset {} partition in {}".format(self.partition, self.region))

        self.cache = os.path.join(self.root, "npy")
        self.cache = append_occluded_classes_label(self.cache, classes_to_exclude)
        self.cache = append_spectral_diff_label(self.cache, self.with_spectral_diff_as_input)
        self.cache = os.path.join(self.cache, region, partition)

        if self.most_important_dates is not None: 
            key_dates_usage = "kept"
            if self.dates_removal:
                key_dates_usage = "removed"
            self.cache = os.path.join(self.cache,
                                      "{}_dates_{}".format(num_dates_to_consider, key_dates_usage))

        print("read {} classes".format(self.nclasses))

        if cache and self.cache_exists():
            print("precached dataset files found at " + self.cache)
            self.load_cached_dataset()
        else:
            print("no cached dataset found. iterating through S2 folders in " + str(self.dataset_path))
            self.cache_dataset()

        print(self)

    def __str__(self):
        base_description = "Dataset {}. region {}. partition {}.".format(self.root, self.region, self.partition)
        instance_statistics = "No observations for the selected dates"
        if len(self.X) > 0:
            instance_statistics = "X:{}, y:{} with {} classes".format(
            str(len(self.X)) +"x"+ str(self.X[0].shape), len(self.y), self.nclasses)

        return base_description + instance_statistics

    def s2transform(self, X, clp):
        X = X.astype(float)
        msk = clp > 128
        msk = np.broadcast_to(msk, X.shape)
        X[msk] = np.nan
        return X

    def cache_dataset(self, min_area=1000):
        """
        Iterates though the data folders and stores y, ids, classweights, and sequencelengths
        X is loaded at with getitem
        """
        #ids = self.split(self.partition

        with open(os.path.join(self.dataset_path, "bbox.pkl"), 'rb') as f:
            bbox = pickle.load(f)
            crs = str(bbox.crs)
            minx, miny, maxx, maxy = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y

        labels = gpd.read_file(self.labelgeojson)
        # project to same coordinate reference system (crs) as the imagery
        self.labels = labels = labels.to_crs(crs)

        mask = labels.geometry.area > min_area
        print(f"ignoring {(~mask).sum()}/{len(mask)} fields with area < {min_area}m2")
        labels = labels.loc[mask]

        bands = np.load(os.path.join(self.dataset_path, "data", "BANDS.npy"))
        clp = np.load(os.path.join(self.dataset_path, "data", "CLP.npy"))
        # bands = np.concatenate([bands, clp], axis=-1) # concat cloud probability
        _, width, height, _ = bands.shape

        transform = rio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

        self.fid_mask = features.rasterize(zip(labels.geometry, labels.fid), all_touched=True,
                                      transform=transform, out_shape=(width, height))
        assert len(np.unique(self.fid_mask)) > 0, f"vectorized fid mask contains no fields. " \
                                             f"Does the label geojson {labelgeojson} cover the region defined by {self.dataset_path}?"

        self.crop_mask = features.rasterize(zip(labels.geometry, labels.crop_id), all_touched=True,
                                       transform=transform, out_shape=(width, height))
        assert len(np.unique(self.crop_mask)) > 0, f"vectorized fid mask contains no fields. " \
                                              f"Does the label geojson {labelgeojson} cover the region defined by {self.dataset_path}?"


        self.labels = labels

        self.X = []
        X_dict = {}
        self.y = []
        self.missing_key_dates_obs = list()
        self.stats = dict(
            not_found=list()
        )
        self.ids = list()
        
        print('bands shape:', bands.shape)

        for i, feature in tqdm(self.labels.iterrows()):

            y = feature.crop_id
            fid = feature.fid
            field_mask = self.fid_mask == fid

            field_bands = bands.transpose(0, 3, 1, 2)[:, :, field_mask]
            field_clp = clp.transpose(0, 3, 1, 2)[:, :, field_mask]

            if self.data_transform:
                field_bands = self.s2transform(field_bands, field_clp)
            
            X_sample = []

            for j in range(field_bands.shape[0]):

                X_sample_date = field_bands[j,:,:]
                
                if self.preprocess_pixels == 'mean':
                    X_sample_date = np.nanmean(X_sample_date, axis=-1) * NORMALIZING_FACTOR
                else:
                    X_sample_date = X_sample_date * NORMALIZING_FACTOR   # -> wrong dimensions; requires further preprocessing

                final_bands_to_use = DENETHOR_BANDS
                if self.with_spectral_diff_as_input:
                    X_sample_date = np.diff(X_sample_date, axis=-1)
                    final_bands_to_use = DENETHOR_BANDS[1:]
                
                missing_key_dates_obs = 0
                if not np.isnan(np.min(X_sample_date)):
                    # if self.most_important_dates is not None:
                    #     # Not reimplemented. Change this in case most_important_dates is used
                    #     obs_acq_dates_as_str = sample[["TIMESTAMP"]].astype(str)
                    #     dates_to_consider = self.most_important_dates.iloc[0:self.num_dates_to_consider]
                    #     if self.dates_removal:
                    #         dates_to_consider = self.most_important_dates.iloc[self.num_dates_to_consider:]
                    #     indices_to_take = obs_acq_dates_as_str[["TIMESTAMP"]].isin(dates_to_consider.index)
                    #     sample = sample.loc[indices_to_take["TIMESTAMP"]]
                    #     missing_key_dates_obs = len(indices_to_take.index) - len(sample.index)

                    observation_dates = np.array([self.timestamps[j].year, self.timestamps[j].month, self.timestamps[j].day]).astype(np.float64)
                    #scale the values in range [0 (everything is absorbed) - 1 (everything is reflected)]
                    #create the resulting dataset consisting of the date and the spectral band values
                    X_ = np.concatenate((observation_dates, X_sample_date)).tolist()
                    X_sample.append(X_)
                
            if len(X_sample) == 0:
                self.stats["not_found"].append(fid)
            else:
                self.X.append(np.array(X_sample))
                X_dict[str(fid)] = X_sample
                self.y.append(y-1)
                self.ids.append(fid)
                self.missing_key_dates_obs.append(missing_key_dates_obs)

        if len(self.X) == 0:
            self.y = -1
            self.sequencelengths = np.array([0])
            self.max_sequence_length = 0
            self.ndims = -1
            self.hist = -1
            self.classweights=-1
        else:
            self.sequencelengths = np.array([np.array(X).shape[0] for X in self.X])
            #assert len(self.sequencelengths) > 0
            self.max_sequence_length = self.sequencelengths.max()
            self.ndims = self.X[0].shape[1]
            self.hist,_ = np.histogram(self.y, bins=self.nclasses)
            self.classweights = 1 / self.hist

        self.cache_variables(self.y, self.sequencelengths, self.ids, self.ndims, X_dict, self.classweights, self.missing_key_dates_obs)


    def cache_variables(self, y, sequencelengths, ids, ndims, X_dict, classweights, missing_key_dates_obs):
        os.makedirs(self.cache, exist_ok=True)
        # cache
        np.save(os.path.join(self.cache, "classweights.npy"), classweights)
        np.save(os.path.join(self.cache, "y.npy"), y)
        np.save(os.path.join(self.cache, "ndims.npy"), ndims)
        np.save(os.path.join(self.cache, "sequencelengths.npy"), sequencelengths)
        np.save(os.path.join(self.cache, "ids.npy"), ids)
        # np.save(os.path.join(self.cache, "dataweights.npy"), dataweights)
        json.dump(X_dict, open(os.path.join(self.cache, "X.json"), 'w'))
        # np.save(os.path.join(self.cache, "X.npy"), X)
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
        for fid, X_values in X_dict.items():
            self.ids.append(int(fid))
            self.X.append(np.array(X_values))
        # self.X = np.load(os.path.join(self.cache, "X.npy"), allow_pickle=True)
        self.missing_key_dates_obs = np.load(os.path.join(self.cache, "missing_key_dates_obs.npy"), allow_pickle=True)

    def cache_exists(self):
        weightsexist = os.path.exists(os.path.join(self.cache, "classweights.npy"))
        yexist = os.path.exists(os.path.join(self.cache, "y.npy"))
        ndimsexist = os.path.exists(os.path.join(self.cache, "ndims.npy"))
        sequencelengthsexist = os.path.exists(os.path.join(self.cache, "sequencelengths.npy"))
        idsexist = os.path.exists(os.path.join(self.cache, "ids.npy"))
        Xexists = os.path.exists(os.path.join(self.cache, "X.json"))
        # Xexists = os.path.exists(os.path.join(self.cache, "X.npy"))
        return yexist and sequencelengthsexist and idsexist and ndimsexist and Xexists and weightsexist

    def clean_cache(self):
        os.remove(os.path.join(self.cache, "classweights.npy"))
        os.remove(os.path.join(self.cache, "y.npy"))
        os.remove(os.path.join(self.cache, "ndims.npy"))
        os.remove(os.path.join(self.cache, "sequencelengths.npy"))
        os.remove(os.path.join(self.cache, "ids.npy"))
        #os.remove(os.path.join(self.cache, "dataweights.npy"))
        os.remove(os.path.join(self.cache, "X.json"))
        os.removedirs(self.cache)


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

    def get_class_names(self):
        return self.classname

    def __getitem__(self, idx):
        X = np.copy(self.X[idx])

        parcel_id = self.ids[idx]

        X, positions = self.sequence_aggregator.aggregate_sequence(parcel_id, X, self)

        red_idx = DENETHOR_BANDS.index(VISIBLE_RED_BAND)
        nir_idx = DENETHOR_BANDS.index(NEAR_INFRARED_BAND)
        y_ndvi = (X[:,nir_idx] - X[:,red_idx]) / (X[:,nir_idx] + X[:,red_idx] + 1e-05)

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
                columns=DATE_COLUMN_NAMES + DENETHOR_BANDS)

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





if __name__ == '__main__':
    
    DENETHOR_root = "/home/luca/luca_docker/datasets/DENETHOR"
    data_rootpath = f"{DENETHOR_root}/sentinel-2/s2-utm-33N-18E-242N-2018"
    labelgeojson = f"{DENETHOR_root}/brandenburg_crops_train_2018.geojson"
    timsstampfile = f"{data_rootpath}/timestamp.pkl"

    # timestamps = np.array(pickle.load(open(timsstampfile,"rb")))
    # print(timestamps)
    # doy = np.array([dt.timetuple().tm_yday for dt in timestamps])

    sequence_aggregator = resolve_sequence_aggregator("random_sampling",
                                                        55,
                                                        None is not None)

    s2dataset = DENETHOR_S2Dataset(root="/home/luca/luca_docker/datasets/DENETHOR",
                                partition="train",
                                classmapping="/home/luca/luca_docker/datasets/DENETHOR/classmapping9.csv",
                                region='s2-utm-33N-18E-242N',
                                sequence_aggregator = sequence_aggregator,
                                classes_to_exclude=None,
                                most_important_dates_file=None,
                                num_dates_to_consider=1,
                                dates_removal=False,
                                with_spectral_diff_as_input=False)

    
    X, positions, y, y_ndvi, parcel_id = s2dataset[0]

    print('NDVI series for ID', parcel_id, positions)
