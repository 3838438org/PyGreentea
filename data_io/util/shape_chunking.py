import itertools
import math


def chunkify_shape(shape, chunk_shape, integral_block_shape=None):
    assert len(shape) == len(chunk_shape), "shape and chunk_shape have different # dimensions"
    if integral_block_shape is not None:
        if any((c % i) != 0 for c, i in zip(chunk_shape, integral_block_shape)):
            raise ValueError("chunk_shape must be an integer multiple of integral_block_shape along every axis")
        shape_1 = tuple(int(math.ceil(1.0 * s / i)) for s, i in zip(shape, integral_block_shape))
        chunk_shape_1 = tuple(c / i for c, i in zip(chunk_shape, integral_block_shape))
        offsets_1 = chunkify_shape(shape_1, chunk_shape_1)
        offsets = tuple(tuple(ibs * o1 for ibs, o1 in zip(integral_block_shape, os1)) for os1 in offsets_1)
        return offsets
    offsets = [None for _ in shape]
    for i, i_offsets in enumerate(offsets):
        extra = shape[i] % chunk_shape[i]
        inner_length = shape[i] - extra
        inner_offsets = tuple(range(0, inner_length, chunk_shape[i]))
        if extra > 0:
            edge_offset = (shape[i] - chunk_shape[i],)
            offsets[i] = inner_offsets + edge_offset
        else:
            offsets[i] = inner_offsets
    return tuple(itertools.product(*offsets))
