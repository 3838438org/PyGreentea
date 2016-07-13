import numpy as np

from .array_value_replacement import replace_array_values, replace_array_except_whitelist, replace_infrequent_values
from .augmentation import create_transformed_array
from .slice_offset_conversion import get_slices_from_dataset_offset
from .value_blob_erosion import erode_value_blobs
from .zero_padded_array_slicing import get_zero_padded_array_slice, get_zero_padded_slice_from_array_by_offset


def data_loader_should_be_used_with(data_arrays):
    # if 'data' is a numpy array, we assume data_arrays's contents are already in-memory
    data_is_in_memory = isinstance(data_arrays[0]['data'], np.ndarray)
    return not data_is_in_memory
