from __future__ import print_function

import collections
import numpy as np


class OffsettedArray(object):
    def __init__(self, array_like, offset, shape):
        assert len(array_like.shape) == len(offset), (array_like.shape, offset)
        assert len(array_like.shape) == len(shape), (array_like.shape, shape)
        self.array_like = array_like
        self.offset = offset
        self.shape = shape
        self.dtype = self.array_like.dtype

    @staticmethod
    def translate_index_expression(index_expression, dim_offset, dim_length):
        if type(index_expression) is int:
            # translate the location
            offsetted_index_expression = index_expression + dim_offset
        elif type(index_expression) is slice:
            # translate the slice
            start = (index_expression.start or 0) + dim_offset
            stop = (index_expression.stop or 0) + dim_offset
            offsetted_index_expression = np.s_[start:stop]
        elif index_expression is None:
            # get everything
            start = dim_offset
            stop = dim_offset + dim_length
            offsetted_index_expression = np.s_[start:stop]
        else:
            raise TypeError("OffsettedArray only works with "
                            "index expressions of type slice, int or NoneType")
        return offsetted_index_expression

    def _convert_item(self, item):
        if not isinstance(item, collections.Iterable):
            # make sure item is iterable
            item = (item,)
        offsetted_item = list(item)
        if Ellipsis in item:
            raise TypeError("Indexing with an Ellipsis is not supported by OffsettedArray")
        for dim_index, index_expression in enumerate(item):
            dim_offset = self.offset[dim_index]
            dim_length = self.shape[dim_index]
            offsetted_index_expression = self.translate_index_expression(index_expression, dim_offset, dim_length)
            offsetted_item[dim_index] = offsetted_index_expression
        return tuple(offsetted_item)

    def __getitem__(self, item):
        new_item = self._convert_item(item)
        return self.array_like[new_item]

    def __setitem__(self, item, value):
        new_item = self._convert_item(item)
        self.array_like[new_item] = value
        return
