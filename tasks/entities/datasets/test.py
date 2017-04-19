import numpy as np

shape = (10,) * 3
affinity_shape = (3,) + shape

affinity_size = np.product(affinity_shape)
affinity = np.random.uniform(low=0, high=1, size=affinity_size)\
    .reshape(affinity_shape)\
    .astype(np.dtype("float32"))

truth = np.ones(shape, dtype=np.dtype("uint32"))

name = "test"
