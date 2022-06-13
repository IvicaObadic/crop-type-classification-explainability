"""
Credits to https://github.com/MarcCoru/crop-type-mapping
"""

import torch
import pandas as pd
import numpy as np
import bisect
import warnings

class ConcatDataset(torch.utils.data.Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, max_sequence_length):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = [dataset for dataset in datasets if dataset.max_sequence_length > 0]
        for dataset in self.datasets:
            dataset.update_max_sequence_length(max_sequence_length)

        self.max_sequence_length = max_sequence_length
        self.nclasses = datasets[0].nclasses
        self.mapping = datasets[0].mapping
        self.classes = datasets[0].classes
        self.ndims = datasets[0].ndims
        self.classweights = datasets[0].classweights
        self.classname = datasets[0].classname
        self.hist = np.array([d.hist for d in self.datasets]).sum(0)
        self.partition = self.datasets[0].partition

        self.y = np.concatenate([d.y for d in self.datasets], axis=0)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def get_class_names(self):
        return self.datasets[0].classname

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    def calculate_spectral_indices(self):
        spectral_indices_dfs = []

        for dataset in self.datasets:
            spectral_indices_dfs.append(dataset.calculate_spectral_indices())

        return pd.concat(spectral_indices_dfs).reset_index()