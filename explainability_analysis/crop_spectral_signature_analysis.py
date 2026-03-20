import pandas as pd
import numpy as np
import os
from datasets import dataset_utils
from datasets import sequence_aggregator
from datasets.util_functions import *


def calc_spectral_signature_per_time_frame(
        spectral_indices_per_parcel,
        agg_variable="CLASS",
        aggregation_time_frame="WEEK"):

    assert aggregation_time_frame in ["WEEK", "MONTH"]

    print("Aggregating the spectral indices based on the time frame method")
    if aggregation_time_frame == "WEEK":
        #map each timestamp to the week start date
        spectral_indices_per_parcel["WEEK"] = spectral_indices_per_parcel["TIMESTAMP"].dt.strftime("%W")
    elif aggregation_time_frame == "MONTH":
        spectral_indices_per_parcel["MONTH"] = spectral_indices_per_parcel["TIMESTAMP"].dt.to_period("M")

    return spectral_indices_per_parcel.groupby([agg_variable, aggregation_time_frame]).\
        agg("mean").reset_index()

def get_dataset_spectral_indices(dataset_folder="/home/luca/luca_docker/datasets/", dataset="BavarianCrops", partition='test', classes_to_exclude=None):

    assert partition in ['train', 'test'], "Partition must be one of 'train' or 'test'."
    dataset_folder = os.path.join(dataset_folder, dataset)
    base_num_classes = 12 if dataset == 'BavarianCrops' else 9
    class_mapping = os.path.join(dataset_folder, "classmapping{}.csv".format(base_num_classes))
    if dataset == 'BavarianCrops':
        train_set, valid_set, test_set = dataset_utils.get_partitioned_dataset_BavarianCrops(
            dataset_folder,
            class_mapping,
            sequence_aggregator.SequencePadder(),
            classes_to_exclude)
    else:
        train_set, valid_set, test_set = dataset_utils.get_partitioned_dataset_DenethorS2(
            dataset_folder,
            class_mapping,
            sequence_aggregator.SequencePadder(),
            classes_to_exclude)

    train_spectral_indices = train_set.calculate_spectral_indices()
    test_spectral_indices = test_set.calculate_spectral_indices()
    return train_spectral_indices, test_spectral_indices

def get_top_NDVI_obs_per_parcel(spectral_indices):
    healthy_vegetation_samples = None
    for parcel_id in spectral_indices.keys():
        parcel_spectral_index = spectral_indices[parcel_id].sort_values(by="NDVI", ascending=False).head(1)
        parcel_spectral_index["PARCEL_ID"] = parcel_id
        if healthy_vegetation_samples is None:
            healthy_vegetation_samples = parcel_spectral_index
        else:
            healthy_vegetation_samples = pd.concat([healthy_vegetation_samples, parcel_spectral_index])

    return healthy_vegetation_samples


def get_average_spectral_reflectance_curve(spectral_indices):
    healthy_vegetation_obs_per_valid_parcel = get_top_NDVI_obs_per_parcel(spectral_indices)
    healthy_vegetation_obs_per_valid_parcel = healthy_vegetation_obs_per_valid_parcel.drop(
        ["YEAR", "MONTH", "DATE", "PARCEL_ID", "NDVI", "CLASS"], axis=1)
    average_spectral_reflectance_curve = healthy_vegetation_obs_per_valid_parcel.mean().to_frame().reset_index()
    average_spectral_reflectance_curve.columns = ['BAND', 'SPECTRAL REFLECTANCE']

    healthy_vegetation_curve = pd.DataFrame({
        "BAND": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"],
        "LEFT": [433, 458, 543, 650, 698, 733, 773, 785, 855, 935, 1360, 1565, 2100],
        "RIGHT": [453, 523, 578, 680, 713, 748, 793, 899, 875, 955, 1390, 1655, 2280],
        "CENTRAL WAVELENGTH": [442.2, 492.1, 559.0, 664.9, 703.8, 739.1, 779.7, 832.9, 864.0, 943.2, 1376.9, 1610.4,
                               2185.7],
        "SPECTRAL SIGNATURE": "Healthy vegetation"})

    healthy_vegetation_curve = pd.melt(healthy_vegetation_curve,
                                       id_vars=["BAND", "CENTRAL WAVELENGTH", "SPECTRAL SIGNATURE"],
                                       var_name="BOUND",
                                       value_name="Wavelength (nm)")

    healthy_vegetation_curve = pd.merge(healthy_vegetation_curve, average_spectral_reflectance_curve, on="BAND")

    return healthy_vegetation_curve



# Function to exclude zeros at the end
def trim_trailing_zeros(ndvi_list):
    return np.trim_zeros(ndvi_list, 'b')

# Extracting both NDVI and NDVI predictions and plotting comparison
def get_ndvi_by_crop(data, crop_types):
    crop_ndvi = {crop: [] for crop in crop_types}
    crop_prediction = {crop: [] for crop in crop_types}
    
    for sample_id, (crop_type, ndvi_timeseries, prediction_timeseries) in data.items():
        if crop_type in crop_types:
            # Remove trailing zeros from both ndvi_timeseries and prediction_timeseries
            trimmed_ndvi = trim_trailing_zeros(ndvi_timeseries)
            trimmed_prediction = trim_trailing_zeros(prediction_timeseries)
            crop_ndvi[crop_type].append(trimmed_ndvi)
            crop_prediction[crop_type].append(trimmed_prediction)
    
    return crop_ndvi, crop_prediction

# Calculate average NDVI per timestep for each crop type
def average_ndvi_per_timestep(crop_ndvi, crop_prediction):
    averaged_ndvi = {}
    averaged_prediction = {}
    
    for crop in crop_ndvi:
        # Pad timeseries to the same length for averaging
        max_length = max(len(ts) for ts in crop_ndvi[crop])
        padded_ndvi = np.array([np.pad(ts, (0, max_length - len(ts)), 'constant', constant_values=np.nan)
                                for ts in crop_ndvi[crop]])
        padded_prediction = np.array([np.pad(ts, (0, max_length - len(ts)), 'constant', constant_values=np.nan)
                                      for ts in crop_prediction[crop]])
        
        # Compute mean ignoring NaNs
        averaged_ndvi[crop] = np.nanmean(padded_ndvi, axis=0)
        averaged_prediction[crop] = np.nanmean(padded_prediction, axis=0)
    
    return averaged_ndvi, averaged_prediction