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
from data_io.minibatches.augmentation import reflect_and_swap_dataset
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


def get_outputs(original_dataset, output_slices):
    output_shape = tuple([slice_.stop - slice_.start for slice_ in output_slices])
    components_raw = get_raw_components(original_dataset, output_slices)
    mask_raw = get_raw_mask(original_dataset, output_slices, components_raw)
    components = remove_named_components(original_dataset, components_raw)
    components = remove_small_components(original_dataset, components)
    components = erode_components(original_dataset, output_shape, components, mask_raw)
    components, affinities = \
        compress_and_shift_up_components(original_dataset, output_shape, components)
    mask_euclidean = get_euclidean_mask(original_dataset, components, mask_raw)
    components, affinities, mask_euclidean = check_and_update_shapes(components, affinities, mask_euclidean, output_shape)
    return components, affinities, mask_euclidean


def get_euclidean_mask(original_dataset, components, mask_raw):
    if "mask" in original_dataset:
        # if the user provided a mask, then always use that
        return mask_raw
    mask_euclidean = np.not_equal(components, 0)
    mask_euclidean = np.logical_and(mask_euclidean, mask_raw)
    mask_euclidean = dilate_mask(original_dataset, mask_euclidean)
    mask_euclidean = mask_euclidean.astype(np.uint8)
    return mask_euclidean


def check_and_update_shapes(components, affinities, mask, output_shape):
    n_spatial_dimensions = len(output_shape)
    components_shape = (1,) + output_shape
    affinities_shape = (n_spatial_dimensions,) + output_shape
    mask_shape = (1,) + output_shape
    components = components.reshape(components_shape)
    assert affinities.shape == affinities_shape, \
        "affinities.shape is {actual} but should be {desired}".format(
            actual=str(affinities.shape), desired=str(affinities_shape))
    mask = mask.reshape(mask_shape)
    return components, affinities, mask


def dilate_mask(original_dataset, mask):
    mask_dilation_steps = original_dataset.get('mask_dilation_steps', 0)
    if mask_dilation_steps > 0:
        mask = ndimage.binary_dilation(mask, iterations=mask_dilation_steps)
    return mask


def get_raw_mask(original_dataset, output_slice, components):
    if 'mask' in original_dataset:
        mask = get_zero_padded_array_slice(original_dataset['mask'], output_slice)
    else:
        source_class = type(original_dataset['components'])
        components_are_from_dvid = source_class in dvid_classes
        if components_are_from_dvid:
            # infer mask values: 1 if component is nonzero, 0 otherwise
            mask = np.not_equal(components, 0)
            logger.debug("No mask provided. Setting to 1 where components != 0.")
        else:
            # assume no masking
            mask = np.ones_like(components, dtype=np.uint8)
            logger.debug("No mask provided. Setting to 1 where outputs exist.")
    mask = mask.astype(np.uint8)
    return mask


def compress_and_shift_up_components(original_dataset, output_shape, components):
    affinities_from_components = malis.seg_to_affgraph(
        components.reshape(output_shape),
        original_dataset['nhood'])
    components, _ = malis.connected_components_affgraph(
        affinities_from_components,
        original_dataset['nhood'])
    components = shift_up_component_values(components)
    return components, affinities_from_components


def erode_components(original_dataset, output_shape, components, mask_raw):
    component_erosion_steps = original_dataset.get('component_erosion_steps', 0)
    if component_erosion_steps > 0:
        only_xy = original_dataset.get('component_erosion_only_xy', False)
        components = erode_value_blobs(
            components.reshape(output_shape),
            steps=component_erosion_steps,
            values_to_ignore=(0,),
            only_xy=only_xy,
            mask=mask_raw.reshape(output_shape),
        )
    return components


def remove_small_components(original_dataset, components):
    minimum_component_size = original_dataset.get('minimum_component_size', 0)
    if minimum_component_size > 0:
        components = replace_infrequent_values(components, minimum_component_size, 0)
    return components


def remove_named_components(original_dataset, components):
    exclude_strings = original_dataset.get('body_names_to_exclude', [])
    source_class = type(original_dataset['components'])
    components_are_from_dvid = source_class in dvid_classes
    if exclude_strings and components_are_from_dvid:
        dvid_uuid = original_dataset['components'].uuid
        components_to_keep = get_good_components(dvid_uuid, exclude_strings)
        logger.debug("components before: {}".format(list(np.unique(components))))
        components = replace_array_except_whitelist(components, 0, components_to_keep)
        logger.debug("components after: {}".format(list(np.unique(components))))
    return components


def get_affinities(original_dataset, output_slice, affinities_from_components):
    if 'label' in original_dataset:
        label_shape = original_dataset['label'].shape
        label_slices = [slice(0, l) for l in label_shape]
        n_spatial_dimensions = len(output_slice)
        label_slices[-n_spatial_dimensions:] = output_slice
        affinities = get_zero_padded_array_slice(original_dataset['label'], label_slices)
    else:
        # compute affinities from components
        logger.debug("Computing affinity labels from components because 'label' wasn't provided in data source.")
        affinities = affinities_from_components
    return affinities


def get_raw_components(original_dataset, output_slices):
    component_slices = [slice(0, l) for l in original_dataset['components'].shape]
    n_spatial_dimensions = len(output_slices)
    component_slices[-n_spatial_dimensions:] = output_slices
    logger.debug("component_slices: {}".format(component_slices))
    components_raw = get_zero_padded_array_slice(original_dataset['components'], component_slices)
    return components_raw


def get_z_distorted_image(image_array, probability_zero, probability_low_contrast, contrast_scalar):
    z_axis_number = image_array.ndim - 3  # works with shapes like (1, Z, Y, X)
    z_length = image_array.shape[z_axis_number]
    probs = np.random.uniform(size=z_length)
    setting_to_zero_for = probs < probability_zero
    reducing_contrast_for = np.logical_and(
        probability_zero < probs,
        probs < (probability_low_contrast + probability_zero)
    )
    for z in range(z_length):
        z_slices = [slice(None) for _ in image_array.shape]
        z_slices[z_axis_number] = z
        z_slices = tuple(z_slices)
        if setting_to_zero_for[z]:
            image_array[z_slices] = 0
        elif reducing_contrast_for[z]:
            z_array = image_array[z_slices]
            z_mean = z_array.mean()
            z_array = z_array - z_mean
            z_array = z_array * contrast_scalar
            z_array = z_array + z_mean
            image_array[z_slices] = z_array
    return image_array


def get_input(original_dataset, input_slice, transform):
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
    image_distortion_z_axis = original_dataset.get("image_distortion_z_axis", None)
    if image_distortion_z_axis:
        assert isinstance(image_distortion_z_axis, dict)
        probability_zero = image_distortion_z_axis.get("probability_zero", 0.0)
        probability_low_contrast = image_distortion_z_axis.get("probability_low_contrast", 0.0)
        contrast_scalar = image_distortion_z_axis.get("contrast_scalar", 1.0)
        assert probability_low_contrast + probability_zero < 1.0
        image = get_z_distorted_image(image, probability_zero, probability_low_contrast, contrast_scalar)
    if image.ndim == n_spatial_dimensions:
        new_shape = (1,) + image.shape
        image = image.reshape(new_shape)
    return image


def get_numpy_dataset(original_dataset, input_slice, output_slice, transform):
    dataset_numpy = dict()
    dataset_numpy["name"] = "{}_at_input_{}_and_output_{}".format(original_dataset.get("name", "Untitled"), input_slice, output_slice)
    image = get_input(original_dataset, input_slice, transform)
    # load outputs if desired
    if output_slice is not None:
        component_erosion_steps = original_dataset.get('component_erosion_steps', 0)
        dilation_amount = 1 + component_erosion_steps
        dilated_output_slices = tuple(slice(s.start - dilation_amount, s.stop + dilation_amount, s.step) for s in output_slice)
        de_dilation_slices = (Ellipsis,) + tuple(slice(dilation_amount, -dilation_amount) for _ in output_slice)
        components, affinities, mask = get_outputs(original_dataset, dilated_output_slices)
        mask_threshold = float(original_dataset.get('mask_threshold', 0))
        mask_fraction_of_this_batch = np.mean(mask[de_dilation_slices])
        good_enough = mask_fraction_of_this_batch > mask_threshold
        if not good_enough:
            return None
        simple_augment = original_dataset.get("simple_augment", False)
        if simple_augment:
            dataset_to_augment = dict(
                name=dataset_numpy["name"],
                data=image,
                components=components,
                mask=mask,
                nhood=original_dataset['nhood']
            )
            augmented_dilated_dataset = simple_augment_minibatch(dataset_to_augment)
            components = augmented_dilated_dataset["components"]
            affinities = augmented_dilated_dataset["label"]
            mask = augmented_dilated_dataset["mask"]
            image = augmented_dilated_dataset["data"]
        dataset_numpy['components'] = components[de_dilation_slices]
        dataset_numpy['label'] = affinities[de_dilation_slices]
        dataset_numpy['mask'] = mask[de_dilation_slices]
        dataset_numpy['nhood'] = original_dataset['nhood']
    dataset_numpy['data'] = image
    return dataset_numpy


def simple_augment_minibatch(dataset_numpy):
    message = "before simple aug {}... \t{: <25}{}\t{: <25}{}\t{: <25}{}" \
        .format((0, 0, 0, 0),
                dataset_numpy["data"].shape, dataset_numpy["data"].mean(),
                dataset_numpy["components"].shape, dataset_numpy["components"].mean(),
                dataset_numpy["mask"].shape, dataset_numpy["mask"].mean())
    logger.debug(message)
    reflectx, reflecty, reflectz, swapxy = np.random.randint(low=0, high=2, size=4)
    dataset_numpy = reflect_and_swap_dataset(dataset_numpy, reflectx, reflecty, reflectz, swapxy)
    message = "after  simple aug {}... \t{: <25}{}\t{: <25}{}\t{: <25}{}" \
        .format((reflectx, reflecty, reflectz, swapxy),
                dataset_numpy["data"].shape, dataset_numpy["data"].mean(),
                dataset_numpy["components"].shape, dataset_numpy["components"].mean(),
                dataset_numpy["mask"].shape, dataset_numpy["mask"].mean())
    logger.debug(message)
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
