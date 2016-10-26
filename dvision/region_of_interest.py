import numpy as np

from dvision import dvid_requester, logger


class RegionOfInterest(object):
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

    @property
    def info(self):
        return

    @property
    def _roi(self):
        if self._roi_cache is None:
            url = self.url_prefix + 'roi'
            # http://slowpoke3:32770/api/node/6a5a7387b4ce4333aa18d9c8d8647f58/alpha_123_roi_dilated/roi
            roi = dvid_requester.get(url)
            self._roi_cache = roi
        return self._roi_cache

    def is_masked(self, slices):
        return False
