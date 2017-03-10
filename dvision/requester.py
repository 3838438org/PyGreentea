from __future__ import print_function

from socket import error as SocketError

import requests
from requests.adapters import HTTPAdapter
from retrying import retry

from dvision import logger


def is_network_error(exception):
    """
    Retry if a network-related exception occurs. Fail otherwise.
    """
    return any([isinstance(exception, SocketError), isinstance(exception, requests.RequestException)])


class DVIDRequester(object):
    def __init__(self, hostname_whitelist):
        self.whitelist = hostname_whitelist
        self.session = requests.Session()

    @retry(wait_exponential_multiplier=100, wait_exponential_max=60000,
           retry_on_exception=is_network_error, wrap_exception=True)
    def get(self, *args, **kwargs):
        logger.debug("Getting url " + repr(args))
        with requests.Session() as session:
            adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
            session.mount('http://', adapter)
            response = session.get(*args, **kwargs)
            if response.ok:
                return response
            else:
                with open("/groups/turaga/home/grisaitisw/bad_response.txt", 'a') as f:
                    f.write(response.text)
                raise Exception("Bad response: {}".format(response.text))

    @retry(wait_exponential_multiplier=100, wait_exponential_max=10000,
           retry_on_exception=is_network_error, wrap_exception=True)
    def post(self, *args, **kwargs):
        url_args = list(args) + [kwargs.get('url', '')]
        hostname_is_ok = any([
                any([hostname in url_arg for hostname in self.whitelist])
                for url_arg in url_args])
        if hostname_is_ok:
            response = self.session.post(*args, **kwargs)
            if response.ok:
                return response
            else:
                raise Exception("Bad response: {}".format(response.text))
        else:
            raise ValueError("posting to servers other than {wl} not allowed" \
                             "you requested {url_args}" \
                             .format(wl=self.whitelist, url_args=url_args))
