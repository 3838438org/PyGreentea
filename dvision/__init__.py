import itertools
import json
import math
import os
import pprint
import warnings

import h5py
import numpy as np
import requests

from .requester import DVIDRequester
dvid_requester = DVIDRequester(['slowpoke1'])

dtype_mappings = {
    "imagetile": None,
    "googlevoxels": None,
    "roi": None,
    "uint8blk": np.uint8,
    "labelvol": None,
    "annotation": None,
    "multichan16": None,
    "rgba8blk": None,
    "labelblk": np.uint64,
    "keyvalue": None,
    "labelgraph": None,
}

from .data_instance import DVIDDataInstance
from .repo import DVIDRepo
from .connection import DVIDConnection


