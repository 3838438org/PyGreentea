import unittest

import numpy as np
from data_io.util import replace_array_except_whitelist, replace_array_values

from .array_wrapper import ArrayWrapper


class FilteredBlacklistArray(ArrayWrapper):
    def __init__(self, array_like, blacklist, replacement=0):
        '''
        Enables slicing from array_like where blacklisted values in the source
        array are replaced with something else
        :param array_like: a slice-able array that returns a numpy array
        :param blacklist: iterable of values to replace in array_like
        :param replacement: value to use instead of original blacklisted values
        '''
        self.array_like = array_like
        self.blacklist = blacklist
        self.replacement = replacement

    def __getitem__(self, item):
        replacements = {value: self.replacement for value in self.blacklist}
        unfiltered_array = self.array_like[item]
        return replace_array_values(unfiltered_array, replacements)

    def __setitem__(self, key, value):
        raise NotImplementedError


class FilteredWhitelistArray(ArrayWrapper):
    def __init__(self, array_like, whitelist, replacement=0):
        '''
        Enables slicing from array_like where values not in whitelist are
        replaced with a `replacement` value
        :param array_like: a slice-able array that returns a numpy array
        :param whitelist: iterable of values to preserve
        :param replacement: value to use instead of non-whitelisted values
        '''
        self.array_like = array_like
        self.whitelist = whitelist
        self.replacement = replacement

    def __getitem__(self, item):
        return replace_array_except_whitelist(self.array_like, self.replacement, self.whitelist)

    def __setitem__(self, key, value):
        raise NotImplementedError


class TestFilteredWhitelistArray(unittest.TestCase):
    def test_filters_values(self):
        x = np.array([0, 1, 2])
        y = FilteredWhitelistArray(x, [1], replacement=3)
        np.testing.assert_array_equal(y[:], np.array([3, 1, 3]))

    def test_works_with_dvid(self):
        import dvision
        from dvision.component_filtering import get_good_components
        components = dvision.DVIDDataInstance("slowpoke3", 32773, "e402c09ddd0f45e980d9be6e9fcb9bd0", "labels1104")
        good_components = get_good_components(components.uuid, ("glia",))
        print(len(good_components))
        # assert good_components is None, good_components
        components = FilteredWhitelistArray(components, good_components)


class TestFilteredBlacklistArray(unittest.TestCase):
    def test_filters_values(self):
        x = np.array([0, 1, 2])
        y = FilteredBlacklistArray(x, [1], replacement=3)
        np.testing.assert_array_equal(y[:], np.array([0, 3, 2]))
        assert False, y[:]
