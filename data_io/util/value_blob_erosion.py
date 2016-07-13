import numpy as np
from scipy import ndimage


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
