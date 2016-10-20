from contextlib import contextmanager


class GenericArrayHandler(object):
    def __init__(self, array_like, name):
        self.array_like = array_like
        self.name = name

    @contextmanager
    def open_array(self, mode=None):
        yield self.array_like


class H5PyArrayHandler(object):
    def __init__(self, path, key, name):
        self.path = path
        self.key = key
        self.name = name

    @contextmanager
    def open_array(self, mode="r"):
        mode = "r"
        import time
        import h5py
        start = time.time()
        with h5py.File(self.path, mode=mode) as h5_f:
            yield h5_f[self.key]
        print("opened array seconds:", time.time() - start)


class H5PyDArrayHandler(object):
    def __init__(self, path, key, name, host="http://slowpoke2:5000"):
        self.path = path
        self.key = key
        self.name = name
        self.host = host

    @property
    def domain_name(self):
        subdirs = self.path.split("/")
        domain_name = ".".join(list(reversed(subdirs)) + ["hdfgroup.org"])
        return domain_name

    @contextmanager
    def open_array(self, mode="r"):
        mode = "r"
        import h5pyd
        try:
            with h5pyd.File(self.domain_name, mode, self.host) as f:
                dataset = f[self.key]
                yield dataset
        except IOError, e:
            print(self.domain_name)
            raise e

class ZarrArrayHandler(object):
    def __init__(self, path, key, name, shape, chunk_shape, dtype):
        self.path = path
        self.key = key
        self.name = name
        self.shape = shape
        self.chunk_shape = chunk_shape
        self.dtype = dtype

    def initialize(self):
        with self.open_array("w") as _:
            pass

    @contextmanager
    def open_array(self, mode="r"):
        import time
        import zarr
        start = time.time()
        def _open():
            z = zarr.open_array(self.path, mode=mode, shape=self.shape,
                                chunks=self.chunk_shape, dtype=self.dtype, fill_value=0)
            return z
        try:
            z = _open()
        except KeyError:
            self.initialize()
            z = _open()
        yield z
        print("opened array seconds:", time.time() - start, self.path, mode)
