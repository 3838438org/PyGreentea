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
