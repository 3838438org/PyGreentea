import numpy as np


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
