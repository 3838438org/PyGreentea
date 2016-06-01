from __future__ import print_function

from os.path import join

import h5py
import malis
import numpy as np

from libdvid.voxels import VoxelsAccessor

from .config import using_in_memory, path_to_training_datasets, training_dataset_names, dataset_source_type, dvid_uuid


def get_train_dataset(dataset_source_type_, using_in_memory=False):
    train_dataset = []
    for dname in training_dataset_names:
        dataset = dict()
        h5_filenames = dict(
            data=join(path_to_training_datasets, dname, 'im_uint8.h5'),
            components=join(path_to_training_datasets, dname, 'groundtruth_seg_thick.h5'),
            # label=join(path_to_training_datasets, dname, 'groundtruth_aff.h5'),
            mask=join(path_to_training_datasets, dname, 'mask.h5'),
        )
        dvid_data_names = dict(
            data='grayscale',
            components='labels',
        )
        dvid_hostname = 'emdata2.int.janelia.org:7000'
        dataset['name'] = dname
        dataset['nhood'] = malis.mknhood3d()
        for key in ['data', 'components']:
            if dataset_source_type_ == VoxelsAccessor:
                if key in dvid_data_names:
                    data_name = dvid_data_names[key]
                    dataset[key] = VoxelsAccessor(hostname=dvid_hostname, uuid=dvid_uuid, data_name=data_name)
            elif dataset_source_type_ == h5py.File:
                dataset[key] = h5py.File(h5_filenames[key], 'r')['main']
                if using_in_memory:
                    dataset[key] = np.array(dataset[key])
                    if key != 'label':
                        dataset[key] = dataset[key].reshape((1,) + dataset[key].shape)
                    if key == 'data':
                        dataset[key] = dataset[key] / 2. ** 8
            elif dataset_source_type_ == 'hdf5 file paths':
                if key in h5_filenames:
                    dataset[key] = h5_filenames[key]
        # dataset['transform'] = {}
        # dataset['transform']['scale'] = (0.8, 1.2)
        # dataset['transform']['shift'] = (-0.2, 0.2)
        train_dataset.append(dataset)
    return train_dataset


# train_dataset = get_train_dataset(dataset_source_type, using_in_memory)
