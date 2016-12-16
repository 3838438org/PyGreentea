import warnings

from dvision import dvid_requester, DVIDDataInstance


class DVIDRepository(object):
    def __init__(self, hostname, port, root_uuid=None):
        self.hostname = hostname
        self.port = port
        self.url_prefix = 'http://' + hostname + ':' + str(port) + '/api/'
        self.root_uuid = root_uuid
        return

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
