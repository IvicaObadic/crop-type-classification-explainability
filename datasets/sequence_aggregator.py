from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import os
import time

from datasets.constants import *
from datasets.util_functions import add_timestamp_column_from_date_columns

class SequenceAggregator(ABC):

    def __init__(self):
        super(SequenceAggregator, self).__init__()
        self.timestamp = None

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp

    @abstractmethod
    def aggregate_sequence(self, parcel_id, X, dataset):
        """
        Aggregates a sequence of temporal observations for a single parcel

        :param parcel_id: the id of the parcel which is aggregated
        :param X: an numpy 2d matrix of size Tx15 for the given parcel where
            T is the number of temporal observations for the pixel and
            the first three columns refer to the year, month and date where each observation was taken
            and the next 12 column refer to the top-of-atmosphere reflectance values for each of the 12 spectral bands.
        :param dataset: an instance of BavarianCrops_Dataset required for reading additional properties of the data required for sampling

        :return: a numpy matrix of size T_aggregated x 12 and
                 a boolean numpy 1-D array of length T_aggregated that indicates which indices are padded
        """
        pass

    @abstractmethod
    def get_label(self):
        pass

    def extract_num_days_since_beggining_of_year(self, dates):
        dates_df = pd.DataFrame(dates, columns=DATE_COLUMN_NAMES)
        dates_with_timestamp_df = add_timestamp_column_from_date_columns(dates_df)
        dates_with_timestamp_df["DAYS_SINCE_BEGINNING"] = \
            (dates_with_timestamp_df['TIMESTAMP'] - STARTING_ACQUISITION_DATE)
        dates_with_timestamp_df["DAYS_SINCE_BEGINNING"] = dates_with_timestamp_df["DAYS_SINCE_BEGINNING"].\
            apply(lambda x:x.days)

        return dates_with_timestamp_df["DAYS_SINCE_BEGINNING"].to_numpy(dtype=np.int16)

    def get_num_training_times(self):
        return 1


class SequenceSampler(SequenceAggregator):

    def __init__(self, time_points_to_sample, fixed_sampling=False):
        """
        Initializes the sampling sequence aggregator.

        :param time_points_to_sample: number of points to sample
        :param fixed_sampling: whether always to sample the same set of points for a given observation
        """
        super(SequenceSampler, self).__init__()
        self.time_points_to_sample = time_points_to_sample
        self.fixed_sampling = fixed_sampling


    def aggregate_sequence(self, parcel_id, X, dataset):
        """
        The sampling sequence aggregator randomly samples fixed number of time points from the entire sequence
        :param parcel_id
        :param X:
        :param dataset:
        :return:
        """
        assert dataset.seed is not None, 'The seed must be set in the dataset for random sampling'

        if self.fixed_sampling:
            #reset the seed such that always the same indices are sampled
            np.random.seed(dataset.seed + parcel_id)

        idxs = np.random.choice(X.shape[0], self.time_points_to_sample, replace=False)
        idxs.sort()
        X = X[idxs]

        #the first three columns contain info about the the date of the observation
        positions = self.extract_num_days_since_beggining_of_year(X[:, 0:3])

        return X[:, 3:], positions

    def get_label(self):

        assert self.timestamp is not None, "The timestamp must always be set"

        type_of_sampler = "random"
        if self.fixed_sampling:
            type_of_sampler = "fixed"
        label = os.path.join(
            "sampling",
            "{}_{}_obs_{}".format(type_of_sampler, self.time_points_to_sample, self.timestamp))
        return label

    def get_num_training_times(self):
        return 5

class SequencePadder(SequenceAggregator):

    def __init__(self):
        """
        Initializes the padding sequence aggregator.
        """
        super(SequenceAggregator, self).__init__()

    def aggregate_sequence(self, parcel_id, X, dataset):
        """
        The padding sequence aggregator pads the sequence to the maximum sequence length observed in the data.
        The padding consists of adding observations filled with PADDING_VALUE at the end of the sequence.

        :param parcel_id
        :param X:
        :param dataset:
        :return:
        """
        assert dataset.max_sequence_length is not None, 'The seed must be set for random sampling'

        #first three columns contain info about the the date of the observation
        positions = self.extract_num_days_since_beggining_of_year(X[:, :3])
        raw_sequence_length = X.shape[0]
        npad = dataset.max_sequence_length - raw_sequence_length
        positions = np.append(positions, np.array([-1]*npad))
        X = np.pad(X, [(0, npad), (0, 0)], 'constant', constant_values=PADDING_VALUE)

        X = X[:, 3:]

        return X, positions

    def get_label(self):
        return 'right_padding'


class WeeklySequenceAggregator(SequenceAggregator):
    """
    Aggregates the sequence by taking averages of the spectral channels on a weekly level
    """

    def __init__(self):
        """
        Initializes the weekly sequence aggregator
        """
        super(SequenceAggregator, self).__init__()

    def aggregate_sequence(self, parcel_id, X, dataset):

        X_df = pd.DataFrame(X, columns=DATE_COLUMN_NAMES + BANDS)
        X_df = add_timestamp_column_from_date_columns(X_df)
        X_df["WEEK"] = X_df["TIMESTAMP"].dt.isocalendar().week

        aggregated_data = X_df.groupby(["WEEK"], as_index=False)[BANDS].mean()

        # check if there is data for all weeks in the calendar year
        all_weeks_in_a_year = set([(week + 1) for week in range(0, 52)])
        weeks_with_missing_data = all_weeks_in_a_year.difference(set(aggregated_data.WEEK))
        if len(weeks_with_missing_data) > 0:
            print('For parcel {} there are no observations in the following weeks: {}'.format(
                parcel_id, ', '.join([str(week) for week in weeks_with_missing_data])))

        aggregated_data = aggregated_data[BANDS].to_numpy(dtype=np.float64)
        positions = np.arange(0, aggregated_data.shape[0])

        return aggregated_data, positions

    def get_label(self):
        return 'weekly_average'

def resolve_sequence_aggregator(seq_aggr_name, time_points_to_sample=None):

    if seq_aggr_name == "weekly_average":
        return WeeklySequenceAggregator()
    elif seq_aggr_name == "right_padding":
        return SequencePadder()
    elif seq_aggr_name == "random_sampling":
        return SequenceSampler(time_points_to_sample=time_points_to_sample)
    else:
        return SequenceSampler(time_points_to_sample=time_points_to_sample, fixed_sampling=True)