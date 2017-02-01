class ArrayWrapper(object):
    array_like = None

    @property
    def dtype(self):
        return self.array_like.dtype

    @property
    def shape(self):
        return self.array_like.shape

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, item):
        return self.array_like[item]

    def __setitem__(self, key, value):
        self.array_like[key] = value
        return
