import pandas as pd
import numpy as np


def calc_spectral_signature_per_class_and_time_frame(
        spectral_indices_per_parcel,
        aggregation_time_frame="WEEK"):

    assert aggregation_time_frame in ["WEEK", "MONTH"]

    print("Concatenating all observations into a single dataframe")
    spectral_signature_for_all_parcels = pd.concat(spectral_indices_per_parcel.values())
    spectral_signature_for_all_parcels.drop(["YEAR", "MONTH", "DATE"], axis=1)
    print("Aggregating the spectral indices based on the time frame method")
    print(spectral_signature_for_all_parcels.head(10))
    if aggregation_time_frame == "WEEK":
        #map each timestamp to the week start date
        spectral_signature_for_all_parcels["WEEK"] = spectral_signature_for_all_parcels.index.strftime("%W")
    elif aggregation_time_frame == "MONTH":
        spectral_signature_for_all_parcels["MONTH"] = spectral_signature_for_all_parcels["TIMESTAMP"].dt.to_period("M")

    return spectral_signature_for_all_parcels.groupby(["CLASS", aggregation_time_frame]).\
        agg(["mean", "std", "max"]).reset_index()


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