from data_io.util import get_zero_padded_array_slice


class ZeroPaddedArray(object):
    def __init__(self, array_like):
        self.array_like = array_like
        self.shape = array_like.shape
        self.dtype = array_like.dtype

    def __getitem__(self, item):
        return get_zero_padded_array_slice(self.array_like, item)

    def __setitem__(self, key, value):
        raise NotImplementedError
