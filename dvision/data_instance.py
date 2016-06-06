import json

import numpy as np

from dvision import dtype_mappings, dvid_requester


class DVIDDataInstance(object):
    def __init__(self, hostname, port, node, name):
        self.hostname = hostname
        self.port = port
        self.node = node
        self.uuid = node
        self.name = name
        api_url = 'http://' + self.hostname + ':' + str(port) + '/api/'
        self.url_prefix = api_url + 'node/' + node + '/' + name + '/'
        self._info_cache = None

    @property
    def info(self):
        if self._info_cache is None:
            url = self.url_prefix + 'info'
            response = dvid_requester.get(url)
            if not response.ok: print(response.url, response.text)
            try:
                self._info_cache = response.json()
            except:
                pass
        return self._info_cache

    @property
    def dtype(self):
        typename = self.info["Base"]["TypeName"]
        dtype = dtype_mappings.get(typename, None)
        if dtype is None:
            error_message = "Unrecognized typename '{tn}'. "\
                            "Not sure how to parse that into a numpy array."\
                            .format(tn=typename)
            raise ValueError(error_message)
        return dtype

    @property
    def shape(self):
        info = self.info
        max_point = info['Extended']['MaxPoint']
        min_point = info['Extended']['MinPoint']
        if max_point is None:
            shape = (0, 0, 0)
        else:
            shape = [max_ - min_ + 1 for min_, max_ in zip(min_point, max_point)]
            shape = tuple(shape)
        return shape

    def __setitem__(self, slices, array):
        shape_of_slices = tuple([s.stop - s.start for s in slices])
        assert array.shape == shape_of_slices, (array.shape, shape_of_slices)
        n_spatial_dims = len(slices)
        axes_str = '_'.join([str(a) for a in range(n_spatial_dims)])
        shape = [s.stop - s.start for s in slices]
        shape_str = '_'.join(str(s) for s in shape)
        offset = [s.start for s in slices]
        offset_str = '_'.join([str(o) for o in offset])
        #  <api URL>/node/<UUID>/<data name>/raw/0_1_2/<size>/<offset>[?queryopts]
        url = self.url_prefix + 'raw/' + axes_str + '/' + shape_str + '/' + offset_str
        array_np = np.array(array, dtype=self.dtype)
        array_as_string = array_np.tostring()
        response = dvid_requester.post(url, data=array_as_string)
        assert response.ok, response.content
        return 

    def __getitem__(self, slices):
        def as_slice(s):
            if type(s) is slice: 
                return s
            else: 
                return slice(s, s + 1, 1)
        slices = tuple(map(as_slice, slices))
        shape_of_slices = tuple([s.stop - s.start for s in slices])
        n_spatial_dims = len(slices)
        axes_str = '_'.join([str(a) for a in range(n_spatial_dims)])
        shape = [s.stop - s.start for s in slices]
        shape_str = '_'.join(str(s) for s in shape)
        offset = [s.start for s in slices]
        offset_str = '_'.join([str(o) for o in offset])
        url = self.url_prefix + 'raw/'+ axes_str + '/' + shape_str + '/' + offset_str + '/nD'
        response = dvid_requester.get(url)
        dvid_octet_stream = response.content
        array = np.fromstring(dvid_octet_stream, dtype=self.dtype)
        array = array.reshape(shape_of_slices)
        return array


class DVIDDataInstanceImageURLGetter(object):
    def __init__(self, dvid_data_instance, image_file_type='jpg'):
        self.dvid_data_instance = dvid_data_instance
        self.image_file_type = image_file_type

    def __getitem__(self, slices):
        raise NotImplementedError("slice handling not implemented")
        shape_of_slices = tuple([s.stop - s.start for s in slices])
        assert min(shape_of_slices) == 1
        n_spatial_dims = len(slices)
        axes_str = '_'.join([str(a) for a in range(n_spatial_dims)])
        shape = [s.stop - s.start for s in slices]
        shape_str = '_'.join(str(s) for s in shape)
        offset = [s.start for s in slices]
        offset_str = '_'.join([str(o) for o in offset])
        # http://slowpoke1:22201/api/node/d2bbf3e3d36541feb8f4baa9b86a73a2/tstvol_2_image/raw/0_1/520_520/0_0_100
        url = self.dvid_data_instance.url_prefix + 'raw/'+ axes_str + '/' + \
              + shape_str + '/' + offset_str + '/' + self.image_file_type
        return url
