class DVIDServer(object):
    block_shape = (32, 32, 32)

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
