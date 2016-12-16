from __future__ import print_function

from contextlib import contextmanager


class BaseArrayHandler(object):
    """This is an abstract base class for array handlers."""

    @contextmanager
    def open_array(self, mode=None):
        '''
        :param mode: read, write, append, etc
        :return: an array-like object
        '''
        raise NotImplementedError

    def initialize(self):
        with self.open_array("w"):
            pass

    @property
    def shape(self):
        with self.open_array(mode="r") as a:
            return a.shape

    @property
    def dtype(self):
        with self.open_array(mode="r") as a:
            return a.dtype


class GenericArrayHandler(BaseArrayHandler):
    def __init__(self, array_like, name):
        self.array_like = array_like
        self.name = name

    @contextmanager
    def open_array(self, mode=None):
        yield self.array_like


class H5PyArrayHandler(BaseArrayHandler):
    def __init__(self, path, key, name):
        self.path = path
        self.key = key
        self.name = name

    @contextmanager
    def open_array(self, mode="r"):
        mode = "r"
        import h5py
        with h5py.File(self.path, mode=mode) as h5_f:
            yield h5_f[self.key]


class H5PyDArrayHandler(BaseArrayHandler):
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

    @contextmanager
    def open_array(self, mode="r"):
        import zarr
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


class VoxelsAccessorArrayHandler(BaseArrayHandler):
    def __init__(self, host, port, uuid, data_name):
        self.host = host
        self.port = port
        self.uuid = uuid
        self.data_name = data_name

    @contextmanager
    def open_array(self, mode="r"):
        from libdvid.voxels import VoxelsAccessor
        host_port = "{h}:{p}".format(h=self.host, p=self.port)
        va = VoxelsAccessor(host_port, self.uuid, self.data_name)
        yield va
