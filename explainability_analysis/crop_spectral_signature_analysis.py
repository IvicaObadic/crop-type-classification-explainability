import pandas as pd


def calc_spectral_signature_per_class_and_time_frame(
        spectral_indices_per_parcel,
        parcel_id_to_class_mapping,
        aggregation_time_frame="WEEK"):

    assert aggregation_time_frame in ["WEEK", "MONTH"]

    print("Concatenating all observations into a single dataframe")
    spectral_signature_for_all_parcels = pd.concat(spectral_indices_per_parcel.values())

    print("Aggregating the spectral indices based on the time frame method")
    if aggregation_time_frame == "WEEK":
        #map each timestamp to the week start date
        spectral_signature_for_all_parcels["WEEK"] = spectral_signature_for_all_parcels["TIMESTAMP"].dt.to_period("W").\
            apply(lambda r: r.start_time.strftime('%Y-%m-%d'))
    elif aggregation_time_frame == "MONTH":
        spectral_signature_for_all_parcels["MONTH"] = spectral_signature_for_all_parcels["TIMESTAMP"].dt.to_period("M")

    return spectral_signature_for_all_parcels.groupby(["CLASS", aggregation_time_frame]).mean().reset_index()
