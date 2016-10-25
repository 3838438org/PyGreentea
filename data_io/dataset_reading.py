from __future__ import print_function

import malis
import multiprocessing
import warnings
from contextlib import contextmanager

import h5py
import numpy as np
from scipy import ndimage

from data_io import logger
import dvision
from dvision.component_filtering import get_good_components
from .util import get_zero_padded_array_slice, replace_array_except_whitelist,\
    replace_infrequent_values, erode_value_blobs

dvid_classes = set()
dvid_classes.add(dvision.DVIDDataInstance)
try:
    from libdvid.voxels import VoxelsAccessor
    dvid_classes.add(VoxelsAccessor)
except ImportError:
    VoxelsAccessor = None
    pass


def condense_and_split_components(components, output_shape, malis_neighborhood):
    '''
    :param components: numpy array with component values
    :param output_shape: tuple with spatial dimensions of components
    :param malis_neighborhood: array definition of malis neighborhood
    :return: numpy array of same shape as components, with new component values
    '''
    original_shape = components.shape
    components_for_malis = components.reshape(output_shape)
    affinities = malis.seg_to_affgraph(components_for_malis, malis_neighborhood)
    recomputed_components, _ = malis.connected_components_affgraph(affinities.astype(np.int32), malis_neighborhood)
    recomputed_components = recomputed_components.reshape(original_shape)
    return recomputed_components


def shift_up_component_values(components):
    '''
    :param components: numpy array with component values
    :return: numpy array of same shape as components, with new component values
    '''
    nonzero_components = np.not_equal(components, 0)
    components += 1
    components *= nonzero_components
    return components


def get_outputs(original_dataset, output_slice):
    output_shape = tuple([slice_.stop - slice_.start for slice_ in output_slice])
    n_spatial_dimensions = len(output_slice)
    components_shape = (1,) + output_shape
    mask_shape = (1,) + output_shape
    affinities_shape = (n_spatial_dimensions,) + output_shape
    component_slices = [slice(0, l) for l in original_dataset['components'].shape]
    component_slices[-n_spatial_dimensions:] = output_slice
    logger.debug("component_slices: {}".format(component_slices))
    components_array = get_zero_padded_array_slice(original_dataset['components'], component_slices)
    source_class = type(original_dataset['components'])
    components_are_from_dvid = source_class in dvid_classes
    exclude_strings = original_dataset.get('body_names_to_exclude', [])
    if exclude_strings and components_are_from_dvid:
        dvid_uuid = original_dataset['components'].uuid
        components_to_keep = get_good_components(dvid_uuid, exclude_strings)
        logger.debug("components before: {}".format(list(np.unique(components_array))))
        components_array = replace_array_except_whitelist(components_array, 0, components_to_keep)
        logger.debug("components after: {}".format(list(np.unique(components_array))))
    minimum_component_size = original_dataset.get('minimum_component_size', 0)
    if minimum_component_size > 0:
        components_array = replace_infrequent_values(components_array, minimum_component_size, 0)
    component_erosion_steps = original_dataset.get('component_erosion_steps', 0)
    if component_erosion_steps > 0:
        components_array = erode_value_blobs(
            components_array,
            steps=component_erosion_steps,
            values_to_ignore=(0,))
    components_for_malis = components_array.reshape(output_shape)
    affinities_from_components = malis.seg_to_affgraph(
        components_for_malis,
        original_dataset['nhood'])
    components_array, _ = malis.connected_components_affgraph(
        affinities_from_components,
        original_dataset['nhood'])
    components_array = shift_up_component_values(components_array)
    components_array = components_array.reshape(components_shape)
    if 'label' in original_dataset:
        label_shape = original_dataset['label'].shape
        label_slices = [slice(0, l) for l in label_shape]
        label_slices[-n_spatial_dimensions:] = output_slice
        affinities_array = get_zero_padded_array_slice(original_dataset['label'], label_slices)
    else:
        # compute affinities from components
        logger.debug("Computing affinity labels from components because 'label' wasn't provided in data source.")
        affinities_array = affinities_from_components
    assert affinities_array.shape == affinities_shape, \
        "affinities_array.shape is {actual} but should be {desired}".format(
            actual=str(affinities_array.shape), desired=str(affinities_shape))
    if 'mask' in original_dataset:
        mask_array = get_zero_padded_array_slice(original_dataset['mask'], output_slice)
    else:
        if components_are_from_dvid:
            # infer mask values: 1 if component is nonzero, 0 otherwise
            mask_array = np.not_equal(components_array, 0)
            logger.debug("No mask provided. Setting to 1 where components != 0.")
        else:
            # assume no masking
            mask_array = np.ones_like(components_array, dtype=np.uint8)
            logger.debug("No mask provided. Setting to 1 where outputs exist.")
    mask_dilation_steps = original_dataset.get('mask_dilation_steps', 0)
    if mask_dilation_steps > 0:
        mask_array = ndimage.binary_dilation(mask_array, iterations=mask_dilation_steps)
    mask_array = mask_array.astype(np.uint8)
    mask_array = mask_array.reshape(mask_shape)
    return components_array, affinities_array, mask_array


def get_numpy_dataset(original_dataset, input_slice, output_slice, transform):
    dataset_numpy = dict()
    dataset_numpy["name"] = "{}_at_input_{}_and_output_{}".format(original_dataset.get("name", "Untitled"), input_slice, output_slice)
    n_spatial_dimensions = len(input_slice)
    image_slices = [slice(0, l) for l in original_dataset['data'].shape]
    image_slices[-n_spatial_dimensions:] = input_slice
    logger.debug("image_slices: {}".format(image_slices))
    image_is_zero_padded = original_dataset.get("image_is_zero_padded", False)
    if image_is_zero_padded:
        source_image = original_dataset["data"][image_slices]
    else:
        source_image = get_zero_padded_array_slice(original_dataset['data'], image_slices)
    image = np.array(source_image, dtype=np.float32)
    image_scaling_factor = original_dataset.get('image_scaling_factor', None)
    if image_scaling_factor is None and source_image.dtype.kind in ('i', 'u'):  # integer, signed or unsigned
        image_scaling_factor = 0.5 ** 8
        message = """Data reader is scaling your image data by a factor of
                     1/256 because it's an integer data type and no scaling
                     factor was provided. If you don't like this default
                     behavior, then provide a dataset['image_scaling_factor']
                     key-value pair in your training dataset."""\
                     .format(isf=image_scaling_factor)
        warnings.warn(message)
    if image_scaling_factor is not None:
        if image_scaling_factor == 1.0:
            # congratulations, you have successfully prevented data scaling
            pass
        else:
            logger.debug("Scaling image by {isf}".format(isf=image_scaling_factor))
            image = np.multiply(image, image_scaling_factor)
    if transform:
        if 'transform' in original_dataset:
            lo, hi = original_dataset['transform']['scale']
            image = 0.5 + (image - 0.5) * np.random.uniform(low=lo, high=hi)
            lo, hi = original_dataset['transform']['shift']
            image = image + np.random.uniform(low=lo, high=hi)
        else:
            logger.debug("source data doesn't have 'transform' attribute.")
    if image.ndim == n_spatial_dimensions:
        new_shape = (1,) + image.shape
        image = image.reshape(new_shape)
    dataset_numpy['data'] = image
    # load outputs if desired
    if output_slice is not None:
        component_erosion_steps = original_dataset.get('component_erosion_steps', 0)
        dilation_amount = 1 + component_erosion_steps
        dilated_output_slices = tuple([slice(s.start - dilation_amount, s.stop + dilation_amount, s.step) for s in output_slice])
        components, affinities, mask = get_outputs(original_dataset, dilated_output_slices)
        de_dilation_slices = (Ellipsis,) + tuple([slice(dilation_amount, -dilation_amount) for _ in output_slice])
        dataset_numpy['components'] = components[de_dilation_slices]
        dataset_numpy['label'] = affinities[de_dilation_slices]
        dataset_numpy['mask'] = mask[de_dilation_slices]
        dataset_numpy['nhood'] = original_dataset['nhood']
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
            logger.debug('opened {} in {}'.format(h5_dataset_key, h5_file_path))
    yield opened_dataset
    for key in opened_dataset:
        if type(opened_dataset[key]) is h5py.Dataset:
            logger.debug('closing {} in {}'.format(opened_dataset[key].name, opened_dataset[key].file.filename))
            opened_dataset[key].file.close()


@contextmanager
def reopen_libdvid_voxelsaccessor_dataset(dataset):
    opened_dataset = dict(dataset)
    for key in dataset:
        dataset_value = dataset[key]
        if type(dataset_value) is VoxelsAccessor:
            hostname = dataset_value.hostname
            uuid = dataset_value.uuid
            data_name = dataset_value.data_name
            new_voxels_accessor = VoxelsAccessor(hostname, uuid, data_name)
            opened_dataset[key] = new_voxels_accessor
            logger.debug('opened {} at {} from {}'.format(data_name, uuid, hostname))
    yield opened_dataset


@contextmanager
def yield_thing(thing):
    '''
    dummy function for dataset objects that don't need to be reinitialized
    '''
    yield thing


def reopen_dataset(dataset):
    if type(dataset['data']) is h5py.Dataset:
        return reopen_h5py_dataset(dataset)
    elif type(dataset['data']) is VoxelsAccessor:
        return reopen_libdvid_voxelsaccessor_dataset(dataset)
    else:
        return yield_thing(dataset)
