import pprint
import warnings

from dvision import dvid_requester, DVIDDataInstance


class DVIDRepo(object):
    def __init__(self, hostname, port, root_uuid=None):
        self.hostname = hostname
        self.port = port
        self.url_prefix = 'http://' + hostname + ':' + str(port) + '/api/'
        if not root_uuid:
            root_uuid = self._create_repo()
        self.root_uuid = root_uuid
        return

    def _create_repo(self):
        response = dvid_requester.post(
            url=self.url_prefix + 'repos',
            json=dict()
        )
        assert response.ok
        pprint.pprint(response.headers)
        pprint.pprint(response.json())
        root_uuid = response.json()['root']
        return root_uuid

    def create_data_instance(self, name, typename):
        json = dict(typename=typename, dataname=name)
        url = self.url_prefix + "repo/" + str(self.root_uuid) + "/" + "instance"
        res = dvid_requester.post(
            url=url,
            json=json
        )
        if not res.ok: warnings.warn(res.text)
        data_instance = DVIDDataInstance(self.hostname, self.port, self.root_uuid, name)
        return data_instance
