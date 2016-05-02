from __future__ import print_function

import unittest

import h5py
from libdvid.voxels import VoxelsAccessor
import numpy as np

from data_io import DataLoader
from data_io.dataset_reading import get_numpy_dataset
from load_datasets import get_train_dataset


class TestDataLoader(unittest.TestCase):
    def execute_test_for(self, dataset_to_test, offset=(0, 0, 0)):
        input_shape = (100, 110, 120)
        output_shape = (40, 30, 80)
        borders = tuple([(in_ - out_) / 2 for (in_, out_) in zip(input_shape, output_shape)])
        input_slices = tuple([slice(x, x + l) for x, l in zip(offset, input_shape)])
        output_slices = tuple([slice(x + b, x + b + l) for x, b, l in zip(offset, borders, output_shape)])
        numpy_dataset = get_numpy_dataset(dataset_to_test, input_slices, output_slices, True)
        data_loader = DataLoader(1, [dataset_to_test], input_shape, output_shape)
        data_loader.start_refreshing_shared_dataset(0, dataset_index=0, offset=offset)
        dataset, index_of_shared_dataset = data_loader.get_dataset()
        for key in ['data', 'components', 'label', 'mask']:
            self.assertEqual(dataset[key].shape, numpy_dataset[key].shape)
            np.testing.assert_almost_equal(dataset[key], numpy_dataset[key])
            # self.assertIs(dataset[key].dtype, numpy_dataset[key].dtype)

    # @mock.patch('data_io.get_numpy_dataset', side_effect=mock_get_numpy_dataset)
    def test_loads_data_chunks_from_dvid(self):
        train_dataset = get_train_dataset(VoxelsAccessor)
        dataset_to_test = train_dataset[0]
        self.execute_test_for(dataset_to_test, offset=(3000, 3000, 3000))

    def test_loads_data_chunks_from_hdf5_fibsem(self):
        train_dataset = get_train_dataset(h5py.File)
        dataset_to_test = train_dataset[0]
        self.execute_test_for(dataset_to_test)

# import mock
# from operator import mul


# def mock_get_numpy_dataset(dataset, input_slices, output_slices, transform):
#     random_state = np.random.RandomState(seed=0)
#     input_shape = tuple([s_.stop - s_.start for s_ in input_slices])
#     output_shape = tuple([s_.stop - s_.start for s_ in output_slices])
#     n_input_pixels = reduce(mul, input_shape)
#     n_output_pixels = reduce(mul, output_shape)
#     return dict(
#         data=random_state.uniform(size=n_input_pixels).reshape(input_shape).astype(np.float32),
#         label=random_state.randint(0, 2, size=3 * n_output_pixels).reshape((3,) + output_shape).astype(np.int32),
#         components=random_state.randint(10000000, size=n_output_pixels).reshape(output_shape).astype(np.int32),
#         mask=random_state.randint(0, 2, size=n_output_pixels).reshape(output_shape).astype(np.uint8)
#     )


if __name__ == "__main__":
    unittest.main()

