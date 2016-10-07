import itertools


def chunkify_shape(shape, chunk_shape):
    assert len(shape) == len(chunk_shape), "shape and chunk_shape have different # dimensions"
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
    # result = itertools.product(*[range(0, s + c - 1, c)[:-1] for s, c in zip(shape, chunk_shape)])
    return tuple(itertools.product(*offsets))


if __name__ == '__main__':
    chunk_offsets = chunkify_shape((1,), (1,))
    assert chunk_offsets == ((0,),), chunk_offsets

    chunk_offsets = chunkify_shape((2,), (1,))
    assert chunk_offsets == ((0,), (1,)), chunk_offsets

    chunk_offsets = chunkify_shape((7,), (5,))
    assert chunk_offsets == ((0,), (2,)), chunk_offsets

    chunk_offsets = chunkify_shape((3, 4), (2, 2))
    assert chunk_offsets == ((0, 0), (0, 2), (1, 0), (1, 2)), chunk_offsets
