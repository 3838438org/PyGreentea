import numpy as np
from scipy import ndimage


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
