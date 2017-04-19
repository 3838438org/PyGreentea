from __future__ import print_function

import os

import h5py
import numpy as np


class NoAffinityFoundException(Exception):
    pass


class EvaluationData(object):
    def __init__(self, model, data_name="tstvol-520-2-h5"):
        self.model = model
        self.data_name = data_name
        self.offset = (0, 0, 0)  # z, y, x
        self.shape = (520, 520, 520)
        self.slices = tuple(slice(o, o + l) for o, l in zip(self.offset, self.shape))
        self.name = data_name
        self.description = os.path.join(self.model.description, self.data_name)

    @property
    def affinity(self):
        affinity_filename = "{}.h5".format(self.data_name)
        affinity_path = os.path.join(self.model.outputs_path, affinity_filename)
        if not os.path.exists(affinity_path):
            raise NoAffinityFoundException
        with h5py.File(affinity_path) as f:
            return f["main"][(slice(0, 3),) + self.slices]

    @property
    def truth(self):
        with h5py.File("/nrs/turaga/grisaitisw/data/FlyEM/fibsem_medulla_7col/%s/groundtruth_seg_thick.h5" % self.data_name) as f:
            truth = f["main"][self.slices]
        truth = np.array(truth, dtype=np.dtype("uint32"))
        print(np.mean(np.equal(truth, 0)), "is % 0 in truth")
        return truth


class TestData(object):
    def __init__(self):
        import numpy as np
        shape = (10,) * 3
        affinity_shape = (3,) + shape
        affinity_size = np.product(affinity_shape)
        self.affinity = np.random.uniform(low=0, high=1, size=affinity_size) \
            .reshape(affinity_shape) \
            .astype(np.dtype("float32"))
        self.truth = np.ones(shape, dtype=np.dtype("uint32"))
        self.name = "test"
        self.description = "Test dataset with shape {}".format(shape)
