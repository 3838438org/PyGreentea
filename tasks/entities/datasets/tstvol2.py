from __future__ import print_function

import h5py
import numpy as np

from data_io.util import erode_value_blobs


offset = (0, 0, 0)  # z, y, x
shape = (520,) * 3
slices = tuple(slice(o, o + l) for o, l in zip(offset, shape))

with h5py.File("/nrs/turaga/grisaitisw/affinities/run_1208_6/400000/tstvol-520-2-h5.h5") as f:
    affinity = f["main"][(slice(0, 3),) + slices]

with h5py.File("/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/tstvol-520-2-h5/groundtruth_seg.h5") as f:
    truth = f["main"][slices]
truth = np.array(truth, dtype=np.dtype("uint32"))
print(np.mean(np.equal(truth, 0)), "is % 0 in truth")
truth = erode_value_blobs(truth, steps=2)

name = "tstvol-520-2-h5"
