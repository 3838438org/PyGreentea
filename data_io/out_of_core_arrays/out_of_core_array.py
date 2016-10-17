class OutOfCoreArray(object):
    def __init__(self, array_opener):
        self.array_opener = array_opener

    def __getitem__(self, item):
        with self.array_opener.open_array(mode="r") as a:
            return a[item]

    def __setitem__(self, item, value):
        with self.array_opener.open_array(mode="a") as a:
            a[item] = value
            return

    @property
    def shape(self):
        with self.array_opener.open_array(mode="r") as a:
            return a.shape
    
    @property
    def dtype(self):
        with self.array_opener.open_array(mode="r") as a:
            return a.dtype
