from __future__ import print_function

import warnings
from contextlib import contextmanager

import h5py
import numpy as np
from libdvid.voxels import VoxelsAccessor

import PyGreentea as pygt
from data_io.util import get_zero_padded_array_slice


def get_numpy_dataset(original_dataset, input_slice, output_slice, transform):
    dataset_numpy = dict()
    input_data_slices = [slice(0, l) for l in original_dataset['data'].shape]
    n_spatial_dimensions = len(input_slice)
    input_data_slices[-n_spatial_dimensions:] = input_slice
    if pygt.DEBUG:
        print("input_data_slices:", input_data_slices)
    original_data_slice = get_zero_padded_array_slice(original_dataset['data'], input_data_slices)
    data_slice = np.array(original_data_slice, dtype=np.float32)
    if original_data_slice.dtype.kind == 'i' or np.max(data_slice) > 1:
        data_slice = data_slice / (2.0 ** 8)
    if transform:
        if 'transform' in original_dataset:
            lo, hi = original_dataset['transform']['scale']
            data_slice = 0.5 + (data_slice-0.5)*np.random.uniform(low=lo, high=hi)
            lo, hi = original_dataset['transform']['shift']
            data_slice = data_slice + np.random.uniform(low=lo, high=hi)
        elif pygt.DEBUG:
            print("WARNING: source data doesn't have 'transform' attribute.")
    dataset_numpy['data'] = data_slice
    # load outputs if desired
    if output_slice is not None:
        component_slices = [slice(0, l) for l in original_dataset['components'].shape]
        component_slices[-len(output_slice):] = output_slice
        if pygt.DEBUG:
            print("component_slices:", component_slices)
        dataset_numpy['components'] = get_zero_padded_array_slice(original_dataset['components'], component_slices)
        if 'label' in original_dataset:
            label_shape = original_dataset['label'].shape
            label_slice = (slice(0, label_shape[0]),) + output_slice
            dataset_numpy['label'] = get_zero_padded_array_slice(original_dataset['label'], label_slice)
        else:
            # compute affinities from components
            dataset_numpy['label'] = pygt.malis.seg_to_affgraph(dataset_numpy['components'], original_dataset['nhood'])
            if pygt.DEBUG:
                warnings.warn("Computing affinity labels because 'label' wasn't provided in data source.", UserWarning)
        if 'mask' in original_dataset:
            dataset_numpy['mask'] = get_zero_padded_array_slice(original_dataset['mask'], output_slice)
            dataset_numpy['mask'] = dataset_numpy['mask'].astype(np.uint8)
        else:
            if type(original_dataset['components']) is VoxelsAccessor:
                # infer mask values: 1 if component is nonzero, 0 otherwise
                assumed_output_mask = np.not_equal(dataset_numpy['components'], 0).astype(np.uint8)
                if pygt.DEBUG:
                    warnings.warn("No mask provided. Setting to 1 where components != 0.", UserWarning)
            else:
                # assume no masking
                assumed_output_mask = np.ones_like(dataset_numpy['components'], dtype=np.uint8)
                if pygt.DEBUG:
                    warnings.warn("No mask provided. Setting to 1 where outputs exist.", UserWarning)
            dataset_numpy['mask'] = assumed_output_mask
    return dataset_numpy


def get_array_view_of_hdf5_dataset(h5_file_path, h5_dataset_key, use_numpy_memmap=False):
    h5_file = h5py.File(h5_file_path, 'r')
    h5_dataset = h5_file[h5_dataset_key]
    if use_numpy_memmap:
        h5_dataset_memory_offset = h5_dataset.id.get_offset()
        numpy_memmap_is_possible = \
            h5_dataset.chunks is None and h5_dataset.compression is None and h5_dataset_memory_offset > 0
        if numpy_memmap_is_possible:
            dtype = h5_dataset.dtype
            shape = h5_dataset.shape
            memory_mapped_array = np.memmap(
                h5_file_path, mode='r', shape=shape,
                offset=h5_dataset_memory_offset, dtype=dtype)
            return memory_mapped_array
        else:
            warnings.warn("Couldn't use numpy.memmap with", h5_dataset_key, "at", h5_file_path)
    return h5_dataset


@contextmanager
def reopen_h5py_dataset(dataset):
    opened_dataset = dict(dataset)
    for key in dataset:
        dataset_value = dataset[key]
        if type(dataset_value) is h5py.Dataset:
            h5_file_path = dataset_value.file.filename
            h5_dataset_key = dataset_value.name
            array_view = get_array_view_of_hdf5_dataset(h5_file_path, h5_dataset_key)
            opened_dataset[key] = array_view
            if pygt.DEBUG:
                print('opened', h5_dataset_key, 'in', h5_file_path)
    yield opened_dataset
    for key in opened_dataset:
        if type(opened_dataset[key]) is h5py.Dataset:
            if pygt.DEBUG:
                print('closing', opened_dataset[key].name, 'in', opened_dataset[key].file.filename)
            opened_dataset[key].file.close()


@contextmanager
def reopen_dvid_dataset(dataset):
    opened_dataset = dict(dataset)
    for key in dataset:
        dataset_value = dataset[key]
        if type(dataset_value) is VoxelsAccessor:
            hostname = dataset_value.hostname
            uuid = dataset_value.uuid
            data_name = dataset_value.data_name
            new_voxels_accessor = VoxelsAccessor(hostname, uuid, data_name)
            opened_dataset[key] = new_voxels_accessor
            if pygt.DEBUG:
                print('opened', data_name, 'from', uuid, "at", hostname)
    yield opened_dataset


def reopen_dataset(dataset):
    if type(dataset['data']) is h5py.Dataset:
        return reopen_h5py_dataset(dataset)
    elif type(dataset['data']) is VoxelsAccessor:
        return reopen_dvid_dataset(dataset)
    else:
        return dataset