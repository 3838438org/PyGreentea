import numpy as np


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
