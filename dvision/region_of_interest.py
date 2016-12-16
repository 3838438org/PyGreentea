import numpy as np

from dvision import dvid_requester, logger


class DVIDRegionOfInterest(object):
    dtype = np.uint8

    def __init__(self, hostname, port, node, name):
        self.hostname = hostname
        self.port = port
        self.node = node
        self.uuid = node
        self.name = name
        api_url = 'http://' + self.hostname + ':' + str(port) + '/api/'
        self.url_prefix = api_url + 'node/' + node + '/' + name + '/'
        self._info_cache = None
        self._roi_cache = None

    def is_masked(self, slices):
        return all(mask_value == 0 for mask_value in self[slices])

    def _make_url_for_slices(self, slices):
        n_spatial_dims = len(slices)
        axes_str = '_'.join([str(a) for a in range(n_spatial_dims)])
        shape = [s.stop - s.start for s in slices]
        shape = tuple(reversed(shape))
        shape_str = '_'.join(str(s) for s in shape)
        offset = [s.start for s in slices]
        offset = tuple(reversed(offset))
        offset_str = '_'.join([str(o) for o in offset])
        url = self.url_prefix + 'mask/' + axes_str + '/' + shape_str + '/' + offset_str
        return url

    def __getitem__(self, slices):
        for s in slices:
            if type(s) is not slice:
                raise TypeError("ROIs only work with slice objects, "
                                "not {} in {}".format(type(s), slices))
        url = self._make_url_for_slices(slices)
        print(url)
        response = dvid_requester.get(url)
        dvid_octet_stream = response.content
        array = np.fromstring(dvid_octet_stream, dtype=self.dtype)
        shape_of_slices = tuple([s.stop - s.start for s in slices])
        array = array.reshape(shape_of_slices)
        return array

    def __setitem__(self, key, value):
        raise NotImplementedError("Can't modify a ROI")
