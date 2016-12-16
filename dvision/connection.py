from .repository import DVIDRepository
from .requester import DVIDRequester


class DVIDConnection(object):
    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self.url_prefix = 'http://' + hostname + ':' + str(port) + '/api/'
        self.dvid_requester = DVIDRequester([hostname])

    def create_repo(self, name=None, description=None):
        repo_metadata = dict()
        if name:
            repo_metadata['alias'] = name
        if description:
            repo_metadata['description'] = description
        response = self.dvid_requester.post(
            url=self.url_prefix + 'repos',
            json=repo_metadata
        )
        assert response.ok, (response.status, response.text)
        root_uuid = response.json()['root']
        return DVIDRepository(self.hostname, self.port, root_uuid)
