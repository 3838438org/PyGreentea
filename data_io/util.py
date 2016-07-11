import numpy as np
from scipy import ndimage


def data_loader_should_be_used_with(data_arrays):
    # if 'data' is a numpy array, we assume data_arrays's contents are already in-memory
    data_is_in_memory = isinstance(data_arrays[0]['data'], np.ndarray)
    return not data_is_in_memory


def get_zero_padded_slice_from_array_by_offset(array, origin, shape):
    assert len(array.shape) == len(origin)
    assert len(array.shape) == len(shape)
    result = np.zeros(shape=shape, dtype=array.dtype)
    source_slices = tuple([
        slice(max(0, offset), min(slice_width+offset, source_width), 1)
        for offset, slice_width, source_width
        in zip(origin, shape, array.shape)
    ])
    target_slices = tuple([
        slice(max(-offset, 0), min(slice_width, source_width-offset), 1)
        for offset, slice_width, source_width
        in zip(origin, shape, array.shape)
    ])
    source_data = array[source_slices]
    result[target_slices] = source_data
    return result


def get_zero_padded_array_slice(array, slices):
    origin = [slice.start for slice in slices]
    shape = [slice.stop - slice.start for slice in slices]
    return get_zero_padded_slice_from_array_by_offset(array, origin, shape)


def get_slices_from_dataset_offset(offset, input_shape, output_shape=None):
    if output_shape is None:
        output_slice = None
    elif max(output_shape) == 0:
        output_slice = None
    else:
        borders = tuple([(in_ - out_) / 2 for (in_, out_) in zip(input_shape, output_shape)])
        output_slice = tuple([slice(offset[i] + borders[i], offset[i] + borders[i] + output_shape[i], 1)
                              for i in range(len(offset))])
    input_slice = tuple([slice(offset[i], offset[i] + input_shape[i], 1)
                         for i in range(len(offset))])
    return input_slice, output_slice


def replace_array_values(array, value_mappings):
    # implementations taken from
    # http://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    u, inv = np.unique(array, return_inverse=True)
    result = np.array([value_mappings.get(x, x) for x in u])[inv].reshape(array.shape)
    # # other implementation...
    # replace_values = np.vectorize(lambda value: value_mappings.get(value, value))
    # result = replace_values(array)
    # # another...
    # new_array = array.copy().flatten()
    # for k in value_mappings:
    #     new_array[array == k] = value_mappings[k]
    # result = new_array
    # # another...
    # for k in value_mappings:
    #     array[array == k] = value_mappings[k]
    # result = array
    return result


def replace_array_except_whitelist(array, new_value, whitelist_values):
    value_mappings = {v: v for v in whitelist_values}
    u, inv = np.unique(array, return_inverse=True)
    replaced_values = [value_mappings.get(x, new_value) for x in u]
    result = np.array(replaced_values, dtype=array.dtype)[inv].reshape(array.shape)
    return result


def replace_infrequent_values(array, count_threshold, new_value):
    uniques, inverse_indices, counts = np.unique(array, return_inverse=True, return_counts=True)
    new_uniques = np.copy(uniques)
    for idx, count in enumerate(counts):
        if count < count_threshold:
            new_uniques[idx] = new_value
    result = np.array(new_uniques, dtype=array.dtype)[inverse_indices].reshape(array.shape)
    return result


def create_transformed_array(array_original, reflectz, reflecty, reflectx, swapxy, angle, rotation_order, scaling_factor=None):
    array = array_original
    if array.ndim > 3:
        try:
            array = array.reshape(array.shape[-3:])
        except:
            raise ValueError("Can't transform ndarray with more than 3 dimensions with length > 1. "
                             "This array's shape is {}".format(array.shape))
    if scaling_factor is not None:
        array = array * scaling_factor
    if reflectz:
        array = array[::-1, :, :]
    if reflecty:
        array = array[:, ::-1, :]
    if reflectx:
        array = array[:, :, ::-1]
    if swapxy:
        array = array.transpose((0, 2, 1))
    if angle > 0:
        new_array = ndimage.rotate(array.astype(np.float32),
                                   angle,
                                   axes=(1, 2),
                                   order=rotation_order,
                                   cval=0)
    else:
        new_array = array
    if new_array.dtype != array_original.dtype:
        print('dtype mismatch: new_array.dtype = {0}, array_original.dtype = {1}'
              .format(new_array.dtype, array_original.dtype
                      ))
    return new_array


def erode_value_blobs(array, steps=1, values_to_ignore=tuple(), new_value=0):
    unique_values = list(np.unique(array))
    all_entries_to_keep = np.zeros(shape=array.shape, dtype=np.bool)
    for unique_value in unique_values:
        entries_of_this_value = array == unique_value
        if unique_value in values_to_ignore:
            all_entries_to_keep = np.logical_or(entries_of_this_value, all_entries_to_keep)
        else:
            eroded_unique_indicator = ndimage.binary_erosion(entries_of_this_value, iterations=steps)
            all_entries_to_keep = np.logical_or(eroded_unique_indicator, all_entries_to_keep)
    result = array * all_entries_to_keep
    if new_value != 0:
        eroded_entries = np.logical_not(all_entries_to_keep)
        new_values = new_value * eroded_entries
        result += new_values
    return result
