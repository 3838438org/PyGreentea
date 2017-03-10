import numpy as np

from dvision import dvid_requester, DVIDDataInstance


class Block(object):
    def __init__(self, start, stop):
        """
        :param start: start coordinate
        :param stop: end of slice
        """
        self.start = start
        self.stop = stop

    @property
    def slices(self):
        return tuple(slice(x0, x1) for x0, x1 in zip(self.start, self.stop))


class DVIDRegionOfInterest(DVIDDataInstance):
    shape = None
    mask_value = 1

    def is_masked(self, slices):
        # True if mask is all zero, else False
        # http://stackoverflow.com/a/23567941/781938
        return not np.any(self[slices])

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

    def get_partition(self, batchsize=16):
        """
        :param batchsize: # of blocks along each face of each partition
        :return: list of subvolumes
        """
        # url = "http://slowpoke3:32788/api/node/341635bc8c864fa5acbaf4558122c0d5/seven_column_eroded7_z_gte_5024/partition?batchsize=1"
        url = self.url_prefix + 'partition?batchsize={}'.format(batchsize)
        response = dvid_requester.get(url)
        assert response.ok, response.content
        roi = response.json()
        roi_subvolumes = roi["Subvolumes"]
        chunks = tuple(
            (tuple(reversed(subvolume["MinPoint"])),
             tuple([max_ + 1 for max_ in reversed(subvolume["MaxPoint"])]))
            for subvolume in roi_subvolumes)
        return [Block(start=c[0], stop=c[1]) for c in chunks]

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
        if self.mask_value != 1:
            array *= self.mask_value
        return array

    def __setitem__(self, key, value):
        raise NotImplementedError("Can't modify a ROI")
