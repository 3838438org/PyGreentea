import unittest
from operator import mul

import numpy as np
import time

from data_io.util import replace_array_values, replace_array_except_whitelist


class TestReplaceArrayValues(unittest.TestCase):
    def test_replace_array_values(self):
        array = np.array([2, 0, 2, 3, 2, 3])
        replacements = {
            2: 0
        }
        result = replace_array_values(array, replacements)
        expected = np.array([0, 0, 0, 3, 0, 3])
        np.testing.assert_almost_equal(result, expected)

    def test_works_with_ndarrays(self):
        array = np.array([2, 0, 2, 3, 2, 3]).reshape((3, 2))
        replacements = {
            2: 0
        }
        result = replace_array_values(array, replacements)
        expected = np.array([0, 0, 0, 3, 0, 3]).reshape((3, 2))
        assert result.shape == (3, 2)
        np.testing.assert_almost_equal(result, expected)

    def test_doesnt_take_too_long_for_a_50_cube(self):
        too_long_seconds = 0.1
        width = 50
        shape = (width,) * 3
        size = reduce(mul, shape)
        array_to_replace_values = np.array([x % (size / 5) for x in range(size)], dtype=np.uint32)
        replacements = {y: 0 for y in range(0, size / 5, 5)}
        start = time.time()
        replace_array_values(array_to_replace_values, replacements)
        execution_seconds = time.time() - start
        self.assertLess(execution_seconds, too_long_seconds)


class TestReplaceArrayExceptWhitelist(unittest.TestCase):
    def test_replace_array_values(self):
        array = np.array([2, 0, 2, 3, 2, 3])
        whitelist = [3]
        result = replace_array_except_whitelist(array, 0, whitelist)
        expected = np.array([0, 0, 0, 3, 0, 3])
        np.testing.assert_almost_equal(result, expected)

    def test_works_with_ndarrays(self):
        array = np.array([2, 0, 2, 3, 2, 3]).reshape((1, 3, 2))
        whitelist = [3]
        result = replace_array_except_whitelist(array, 0, whitelist)
        expected = np.array([0, 0, 0, 3, 0, 3]).reshape((1, 3, 2))
        assert result.shape == (1, 3, 2)
        np.testing.assert_almost_equal(result, expected)

    def test_doesnt_take_too_long_for_a_50_cube(self):
        too_long_seconds = 0.1
        width = 50
        shape = (width,) * 3
        size = reduce(mul, shape)
        array = np.array([x % (size / 5) for x in range(size)], dtype=np.uint32)
        whitelist = range(0, size / 5, 5)
        start = time.time()
        replace_array_except_whitelist(array, 0, whitelist)
        execution_seconds = time.time() - start
        self.assertLess(execution_seconds, too_long_seconds)
