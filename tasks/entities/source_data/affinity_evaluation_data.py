from __future__ import print_function

from contextlib import contextmanager

import h5py
import malis
import numpy as np


class AffinityEvaluationData(object):
    def __init__(self, d, affinity_file_path, nhood):
        self.d = d
        self.affinity_file_path = affinity_file_path
        self.nhood = nhood
        self.dname = d['dname']
        self.new_shape = d['new_shape']
        self.ndim = len(self.new_shape)

    @property
    def components_cropped(self):
        components_cropped = self.get_centered_slice_from_source(self.d['components'])
        affgraph = malis.seg_to_affgraph(
            components_cropped, self.nhood)
        components_cropped, _ = malis.connected_components_affgraph(
            affgraph, self.nhood)
        components_cropped = components_cropped.astype(np.uint64)
        print(components_cropped.shape)
        assert components_cropped.shape == self.new_shape
        return components_cropped

    @contextmanager
    def open_aff_source(self):
        affh5_file = h5py.File(self.affinity_file_path, 'r')
        affh5 = affh5_file['main']
        yield affh5
        affh5_file.close()

    @property
    def aff(self):
        with self.open_aff_source() as affh5:
            print("shape of affh5:", affh5.shape)
            aff = self.get_centered_slice_from_source(affh5)
        aff = np.array(aff, dtype=np.float32)
        print("aff.shape: ", aff.shape)
        assert aff.shape == (3,) + self.new_shape
        return aff

    def get_centered_slice_from_source(self, source_array):
        source_shape = source_array.shape[-self.ndim:]
        offset = tuple([(o - n) / 2 for o, n in zip(source_shape, self.new_shape)])
        slices = (Ellipsis,) + tuple([slice(o, o + l) for o, l in zip(offset, self.new_shape)])
        print('Cropping from {} to {} with slices {}'.format(
            source_shape, self.new_shape, slices))
        new_array = source_array[slices]
        return new_array


