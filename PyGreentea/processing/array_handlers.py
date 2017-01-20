import os
from contextlib import contextmanager
from operator import mul
import time

import h5py
import h5pyd
import numpy as np
import zarr


class H5PYDArrayHandler(object):
    def __init__(self, path, h5_dataset_key, name="untitled"):
        self.path = path
        self.h5_dataset_key = h5_dataset_key
        self.name = name

    @property
    def shape(self):
        with self.get_array("r") as a:
            return a.shape

    @property
    def dtype(self):
        with self.get_array("r") as a:
            return a.dtype

    def open_file(self, mode="r"):
        subdirs = self.path.split("/")
        domain_name = ".".join(list(reversed(subdirs)) + ["hdfgroup.org"])
        print(domain_name)
        # f = h5pyd.File(domain_name, mode, "http://slowpoke2:5000")
        f = h5pyd.File(domain_name, "r", "http://slowpoke2:5000")
        return f

    @contextmanager
    def get_array(self, mode="r"):
        f = self.open_file(mode=mode)
        try:
            dataset = f[self.h5_dataset_key]
        except KeyError, e:
            f.close()
            if mode[0] == "r":
                raise e
            else:
                # self.initialize_array()
                f = self.open_file(mode=mode)
                dataset = f[self.h5_dataset_key]
        yield dataset
        f.close()

    def initialize_array(self, shape, chunk_shape, dtype=np.float32):
        # f = self.open_file(mode="w")
        # f.create_dataset(
        #     name=self.h5_dataset_key,
        #     shape=shape,
        #     dtype=dtype,
        #     chunks=chunk_shape,
        #     compression="gzip",
        #     compression_opts=9,
        # )
        # f.close()
        pass


class H5PYDAffinityHandler(object):
    dataset_key = "main"

    def __init__(self, dname, model, iteration,
                 shape=(3, 10000, 10000, 10000), chunk_shape=(3, 32, 32, 32),
                 root_dir="/nrs/turaga/grisaitisw/affinities"):
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
    def __init__(self, dname, modelname, iteration, root_dir="/nrs/turaga/grisaitisw/affinities"):
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


class ZarrAffinityHandler(H5PYDAffinityHandler):
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
        offset = (0,) + offset[-3:]
        assert len(self.shape) == len(array.shape), (self.shape, array.shape)
        assert len(self.shape) == len(offset), (self.shape, offset)
        shape = tuple(min(a, N - o) for a, N, o in zip(array.shape, self.shape, offset))
        print(shape)
        target_slices = tuple(slice(o, o + l) for o, l in zip(offset, shape))
        source_slices = tuple(slice(0, l) for l in shape)
        with self.get_array(mode="a") as a:
            a[target_slices] = array[source_slices]


class AffinitySaver(object):
    def __init__(self, affinity_array_like):
        self.affinity_array_like = affinity_array_like

    def save(self, array, offset):
        offset = (0,) + offset[-3:]
        assert len(self.affinity_array_like.shape) == len(array.shape), (self.affinity_array_like.shape, array.shape)
        assert len(self.affinity_array_like.shape) == len(offset), (self.affinity_array_like.shape, offset)
        shape = tuple(min(a, N - o) for a, N, o in zip(array.shape, self.affinity_array_like.shape, offset))
        print(shape)
        target_slices = tuple(slice(o, o + l) for o, l in zip(offset, shape))
        source_slices = tuple(slice(0, l) for l in shape)
        self.affinity_array_like[target_slices] = array[source_slices]

if __name__ == "__main__":
    # sas = SimpleHDF5AffinitySaver("fib25", "test_model", 0)
    aff = np.random.uniform(size=3*1000).reshape((3, 10, 10, 10))
    # sas.save(aff, (4, 7, 9))
    szah = ZarrAffinityHandler("test_data", "test_model", 0)
    szah.initialize_array()
    szah.save(aff, (4, 7, 9))
    with szah.get_array(mode="r") as a:
        actual = a[0:3, 4:14, 7:17, 9:19]
        print(np.mean(actual))
        print(np.mean(aff[:]))
        print(a.chunks)
