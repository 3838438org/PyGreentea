from __future__ import print_function

import unittest

import numpy as np

from data_io.dataset_reading import reopen_dataset, get_numpy_dataset
from load_datasets import train_dataset


class TestDVIDLoading(unittest.TestCase):
    def test_dvid_loading_into_numpy_datasets(self):
        for dataset in train_dataset:
            n_samples = 1000
            n_dimensions = 3
            size = n_samples * n_dimensions
            max_extent = min(dataset['data'].shape[-n_dimensions:])
            origins = np.random.randint(0, max_extent, size).reshape((n_samples, n_dimensions))
            list_of_slices = [[slice(x, x + 100, 1) for x in origin] for origin in origins]
            for idx, slices in enumerate(list_of_slices):
                with reopen_dataset(dataset) as reopened_dataset:
                    numpy_dataset = get_numpy_dataset(reopened_dataset, slices, slices, True)
                    data_mean = np.mean(numpy_dataset['data'])
                    mask_mean = np.mean(numpy_dataset['mask'])
                    components_mean = np.mean(numpy_dataset['components'])
                    print(idx, "has means:",
                          "data %05.3f" % data_mean,
                          "mask %05.3f" % mask_mean,
                          "components %012.3f" % components_mean,
                          "at", slices)
                    if mask_mean == 0:
                        print("skipping", idx)
                    else:
                        print("checking", idx)
                        self.assertLessEqual(np.max(numpy_dataset['data']), 1)
                        self.assertGreaterEqual(np.min(numpy_dataset['data']), 0)
