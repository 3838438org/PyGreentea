import numpy as np


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
