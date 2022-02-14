import pandas as pd
import numpy as np


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