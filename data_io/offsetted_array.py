class OffsettedArray(object):
    def __init__(self, array_like, offset, shape):
        assert len(array_like.shape) == len(offset), (array_like.shape, offset)
        assert len(array_like.shape) == len(shape), (array_like.shape, shape)
        self.array_like = array_like
        self.offset = offset
        self.shape = shape
        self.dtype = self.array_like.dtype

    def _convert_item(self, item):
        new_item = list(item)
        for i, s in enumerate(item):
            assert type(s) is slice, "sorry... OffsettedArray only works with slice objects"
            if i >= len(self.array_like.shape) - len(self.offset):
                new_item[i] = slice(s.start + self.offset[i], s.stop + self.offset[i], s.step)
        return tuple(new_item)

    def __getitem__(self, item):
        new_item = self._convert_item(item)
        print("fetching item", new_item)
        return self.array_like[new_item]

    def __setitem__(self, item, value):
        new_item = self._convert_item(item)
        print("setting item", new_item)
        self.array_like[new_item] = value
        return
