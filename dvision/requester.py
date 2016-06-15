from dvision import logger

import requests
from requests.adapters import HTTPAdapter


class DVIDRequester(object):
    def __init__(self, hostname_whitelist):
        self.whitelist = hostname_whitelist
        self.session = requests.Session()

    def get(self, *args, **kwargs):
        logger.debug("Getting url " + repr(args))
        with requests.Session() as session:
            adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
            session.mount('http://', adapter)
            return session.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        url_args = list(args) + [kwargs.get('url', '')]
        hostname_is_ok = any([
                any([hostname in url_arg for hostname in self.whitelist])
                for url_arg in url_args])
        if hostname_is_ok:
            return self.session.post(*args, **kwargs)
        else:
            raise ValueError("posting to servers other than {wl} not allowed" \
                             "you requested {url_args}" \
                             .format(wl=self.whitelist, url_args=url_args))
