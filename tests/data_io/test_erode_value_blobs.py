import unittest

import numpy as np

from data_io.util import erode_value_blobs

class TestErodeValueBlobs(unittest.TestCase):
    def test_erodes_with_default_arguments(self):
        X = np.array([1, 1, 1])
        X_eroded = erode_value_blobs(X)
        np.testing.assert_array_equal(X_eroded, np.array([0, 1, 0]))

    def test_erodes_with_more_than_one_step(self):
        X = np.array([1, 1, 1, 1, 1])
        X_eroded = erode_value_blobs(X, steps=2)
        expected = np.array([0, 0, 1, 0, 0])
        np.testing.assert_array_equal(X_eroded, expected)

    def test_erodes_completely_if_steps_is_big_enough(self):
        X = np.array([1, 1, 1])
        X_eroded = erode_value_blobs(X, steps=2)
        np.testing.assert_array_equal(X_eroded, np.array([0, 0, 0]))

    def test_can_ignore_values(self):
        X = np.array([1, 1, 1])
        X_eroded = erode_value_blobs(X, values_to_ignore=(1,))
        np.testing.assert_array_equal(X_eroded, X)

    def test_can_erode_multiple_values(self):
        X = np.array([1, 1, 1, 2, 2, 2])
        X_eroded = erode_value_blobs(X)
        np.testing.assert_array_equal(X_eroded, np.array([0, 1, 0, 0, 2, 0]))

    def test_can_set_eroded_entries_with_new_value(self):
        X = np.array([1, 1, 1])
        X_eroded = erode_value_blobs(X, new_value=2)
        np.testing.assert_array_equal(X_eroded, np.array([2, 1, 2]))

    def test_works_in_2d(self):
        X = \
            np.array([[1, 1, 1, 2, 2, 2],
                      [1, 1, 1, 2, 2, 2],
                      [1, 1, 1, 2, 2, 2]])
        X_eroded = erode_value_blobs(X)
        expected = \
            np.array([[0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 2, 0],
                      [0, 0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(X_eroded, expected)
