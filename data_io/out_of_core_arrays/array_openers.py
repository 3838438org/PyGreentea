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


class ZarrArrayHandler(BaseArrayHandler):
    def __init__(self, path, key, name, shape, chunk_shape, dtype):
        self.path = path
        self.key = key
        self.name = name
        self._shape = shape
        self._chunk_shape = chunk_shape
        self._dtype = dtype

    def initialize(self):
        import zarr
        zarr.open_array(self.path, mode="w", shape=self._shape,
                        chunks=self._chunk_shape, dtype=self._dtype, fill_value=0)
        return

    @contextmanager
    def open_array(self, mode="r"):
        # modes explained at http://zarr.readthedocs.io/en/latest/api/creation.html?highlight=mode#zarr.creation.open_array
        import zarr
        if mode in ("w", "w-"):
            z = zarr.open_array(self.path, mode=mode, shape=self._shape,
                                chunks=self._chunk_shape, dtype=self._dtype, fill_value=0)
        else:  # mode in ("a", "r", "r+"), or something else, which zarr can complain about if it wants to
            # if it doesn't exist, create it with the correct arguments
            try:
                z = zarr.open_array(self.path, mode=mode, fill_value=0)
            except:
                self.initialize()
                z = zarr.open_array(self.path, mode=mode, fill_value=0)
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
