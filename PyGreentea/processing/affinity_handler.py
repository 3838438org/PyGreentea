import os
from contextlib import contextmanager
from operator import mul
import time

import h5py
import h5pyd
import numpy as np
import zarr


class H5PYDAffinityHandler(object):
    dataset_key = "main"

    def __init__(self, dname, model, iteration,
                 shape=(3, 10000, 10000, 10000), chunk_shape=(3, 32, 32, 32),
                 root_dir="/nobackup/turaga/grisaitisw/affinities"):
        self.dname = dname
        self.model = model
        self.iteration = str(iteration)
        self.shape = shape
        self.chunk_shape = chunk_shape
        self.root_dir = root_dir

    def open_file(self, mode="r"):
        domain_name = ".".join((self.dname, self.iteration, self.model, "hdfgroup.org"))
        print(domain_name)
        f = h5pyd.File(domain_name, mode, "http://slowpoke2:5000")
        return f

    @contextmanager
    def get_array(self, mode="r"):
        f = self.open_file(mode=mode)
        try:
            dataset = f[self.dataset_key]
        except KeyError:
            f.close()
            self.initialize_array()
            f = self.open_file(mode=mode)
            dataset = f[self.dataset_key]
        # dataset = f.require_dataset(
        #     name=self.dataset_key,
        #     shape=self.default_shape,
        #     dtype=np.float32,
        #     chunks=(3, 16, 16, 16),
        #     compression="gzip",
        #     compression_opts=9,
        # )
        yield dataset
        f.close()

    def initialize_array(self):
        f = self.open_file(mode="w")
        f.create_dataset(
            name=self.dataset_key,
            shape=self.shape,
            dtype=np.float32,
            chunks=self.chunk_shape,
            compression="gzip",
            compression_opts=9,
        )
        f.close()


class H5PYAffinityHandler(H5PYDAffinityHandler):
    def open_file(self, mode="r"):
        path = os.path.join(self.root_dir, self.model, self.iteration, self.dname + ".h5")
        f = h5py.File(path, mode)
        return f


class SimpleHDF5AffinitySaver(object):
    def __init__(self, dname, modelname, iteration, root_dir="/nobackup/turaga/grisaitisw/affinities"):
        self.dname = dname
        self.modelname = modelname
        self.iteration = str(iteration)
        self.save_dir = os.path.join(root_dir, self.modelname, self.iteration)

    def save(self, affinity_subarray, offset):
        offset_slices = ["{:05d}-{:05d}".format(x, x + l) for x, l in zip(offset, affinity_subarray.shape[-3:])]
        loc = "_".join(offset_slices)
        f5_filename = "{}_{}.h5".format(self.dname, loc)
        h5_path = os.path.join(self.save_dir, f5_filename)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("main", data=affinity_subarray)


class SimpleZarrAffinityHandler(H5PYDAffinityHandler):
    '''
    instance vars.... dname, model, iteration
    class cars... default dataset key, default shape
    '''
    @property
    def path(self):
        return os.path.join(self.root_dir, self.model, self.iteration, self.dname + ".zarr")

    @contextmanager
    def get_array(self, mode="r"):
        start = time.time()
        z = zarr.open_array(self.path, mode=mode, shape=self.shape)
        yield z
        print("opened array seconds:", time.time() - start)

    def initialize_array(self):
        print(self.path)
        zarr.open_array(self.path, mode='w', shape=self.shape,
                        chunks=self.chunk_shape, dtype=np.float32, fill_value=0)

    def save(self, array, offset):
        slices = (slice(0, 3),) + tuple(slice(x, x + l) for x, l in zip(offset, array.shape[-3:]))
        with self.get_array(mode="a") as a:
            a[slices] = array


if __name__ == "__main__":
    # sas = SimpleHDF5AffinitySaver("fib25", "test_model", 0)
    aff = np.random.uniform(size=3*1000).reshape((3, 10, 10, 10))
    # sas.save(aff, (4, 7, 9))
    szah = SimpleZarrAffinityHandler("test_data", "test_model", 0)
    szah.initialize_array()
    szah.save(aff, (4, 7, 9))
    with szah.get_array(mode="r") as a:
        actual = a[0:3, 4:14, 7:17, 9:19]
        print(np.mean(actual))
        print(np.mean(aff[:]))
        print(a.chunks)
