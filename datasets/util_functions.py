import os
import pandas as pd

from datasets.constants import *


def add_timestamp_column_from_date_columns(input_df):
    """
    Creates a timestamp column which joins the year, date and month column from the input dataset
    :param input_df: a dataframe where the first column represents the YEAR, the second column represents the MONTH
                     and the third column represents the date
    :return: the input dataframe with added 'TIMESTAMP' column
    """
    input_df[DATE_COLUMN_NAMES] = input_df[DATE_COLUMN_NAMES].astype(int)
    input_df['TIMESTAMP'] = input_df[input_df.columns[0:3]].apply(
        lambda x: '/'.join(x.astype(str)),
        axis=1)
    input_df['TIMESTAMP'] = pd.to_datetime(input_df['TIMESTAMP'])
    return input_df


def append_occluded_classes_label(label_str, classes_to_occlude):
    if classes_to_occlude is None:
        return label_str
    return os.path.join(label_str, "wo_" + ",".join(classes_to_occlude))


def append_spectral_diff_label(label_str, with_spectral_diff_as_input):
    if not with_spectral_diff_as_input:
        return label_str
    return os.path.join(label_str, "with_spectral_diff_as_input")