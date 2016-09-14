from __future__ import print_function

import unittest

import numpy as np
import h5py
import malis

from data_io.dataset_reading import get_numpy_dataset
from data_io.util import replace_array_except_whitelist
from dvision import DVIDDataInstance
from dvision.component_filtering import get_good_components
from tests.data_io.load_datasets import get_train_dataset


class TestGetNumpyDataset(unittest.TestCase):
    def test_works_with_hdf5_fibsem(self):
        train_dataset = get_train_dataset(h5py.File)
        dataset = train_dataset[0]
        origin = (105, 100, 100)
        numpy_dataset, expected_dataset = self.get_datasets(dataset, origin)
        for key in ['data', 'label', 'mask']:
            self.assertEqual(numpy_dataset[key].shape, expected_dataset[key].shape)
            np.testing.assert_almost_equal(
                numpy_dataset[key],
                expected_dataset[key],
                err_msg=str((key, "result", numpy_dataset[key], "desired", expected_dataset[key])),
            )

    def test_works_with_dvid(self):
        train_dataset = get_train_dataset(DVIDDataInstance)
        dataset = train_dataset[0]
        dataset['body_names_to_exclude'] = ['out']
        origin = (3000, 3000, 3000)
        numpy_dataset, expected_dataset = self.get_datasets(dataset, origin)
        for key in ['data', 'label', 'mask']:
            self.assertEqual(numpy_dataset[key].shape, expected_dataset[key].shape)
            np.testing.assert_almost_equal(
                numpy_dataset[key],
                expected_dataset[key],
                err_msg=str((key, "result", numpy_dataset[key], "desired", expected_dataset[key])),
            )

    @staticmethod
    def get_datasets(dataset, origin):
        # output_shape = (40, 30, 80)
        # input_shape = (100, 110, 120)
        output_shape = (2,3,4)
        input_shape = tuple([x + 2 for x in output_shape])
        borders = tuple([(in_ - out_) / 2 for (in_, out_) in zip(input_shape, output_shape)])
        input_slices = tuple([slice(x, x + l) for x, l in zip(origin, input_shape)])
        output_slices = tuple([slice(x + b, x + b + l) for x, b, l in zip(origin, borders, output_shape)])
        expected_dataset = dict()
        data_slices = [slice(0, l) for l in dataset['data'].shape]
        data_slices[-3:] = input_slices
        data_slices = tuple(data_slices)
        expected_data_array = np.array(dataset['data'][data_slices], dtype=np.float32)
        expected_data_array = expected_data_array.reshape((1,) + input_shape)
        expected_data_array /= (2.0 ** 8)
        expected_dataset['data'] = expected_data_array
        components_slices = [slice(0, l) for l in dataset['components'].shape]
        components_slices[-3:] = output_slices
        components_slices = tuple(components_slices)
        expected_components_array = np.array(dataset['components'][components_slices]).reshape((1,) + output_shape)
        if type(dataset['components']) is DVIDDataInstance:
            print("Is DVIDDataInstance...")
            print("uniques before:", np.unique(expected_components_array))
            dvid_uuid = dataset['components'].uuid
            body_names_to_exclude = dataset.get('body_names_to_exclude')
            good_bodies = get_good_components(dvid_uuid, body_names_to_exclude)
            expected_components_array = \
                replace_array_except_whitelist(expected_components_array, 0, good_bodies)
            print("uniques after:", np.unique(expected_components_array))
        expected_dataset['components'] = expected_components_array
        components_for_affinity_generation = expected_components_array.reshape(output_shape)
        expected_label = malis.seg_to_affgraph(components_for_affinity_generation, malis.mknhood3d())
        expected_dataset['label'] = expected_label
        if type(dataset['components']) is DVIDDataInstance:
            expected_mask = np.array(expected_components_array > 0).astype(np.uint8)
        else:
            expected_mask = np.ones(shape=(1,) + output_shape, dtype=np.uint8)
        expected_dataset['mask'] = expected_mask
        numpy_dataset = get_numpy_dataset(dataset, input_slices, output_slices, True)
        return numpy_dataset, expected_dataset


if __name__ == "__main__":
    unittest.main()
