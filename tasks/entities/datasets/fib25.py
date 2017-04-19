import dvision
import numpy as np
import zarr

from data_io.util import erode_value_blobs

offset = (5440, 3008, 3712)  # z, y, x
shape = (50,) * 3

slices = tuple(slice(o, o + l) for o, l in zip(offset, shape))
slices = (slice(0, 3),) + slices

affinity_path = "/nrs/turaga/grisaitisw/affinities/run_1208_6/400000/fib25-e402c09-2017.01.20-17.37.16/"
affinity = zarr.open_array(affinity_path, "r")[slices]

truth = dvision.DVIDDataInstance("slowpoke3", 32788, "341", "truth")
truth = truth[slices[-3:]].astype(np.uint32)
print(np.mean(np.equal(truth, 0)), "is % 0 in truth")
truth = erode_value_blobs(truth, steps=2)
