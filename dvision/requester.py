import requests


class DVIDRequester(object):
    def __init__(self, hostname_whitelist):
        self.whitelist = hostname_whitelist

    def get(self, *args, **kwargs):
        return requests.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        url_args = list(args) + [kwargs.get('url', '')]
        hostname_is_ok = any([
                any([hostname in url_arg for hostname in self.whitelist]) 
                for url_arg in url_args])
        if hostname_is_ok:
            return requests.post(*args, **kwargs)
        else:
            raise ValueError("posting to servers other than {wl} not allowed"\
                             "you requested {url_args}"\
                             .format(wl=self.whitelist, url_args=url_args))
        return
