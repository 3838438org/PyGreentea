import logging

logging.basicConfig()

logger = logging.getLogger('dvision')
logger.setLevel(logging.INFO)

from .requester import DVIDRequester

dvid_requester = DVIDRequester(['slowpoke1'])

import numpy as np

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
from .region_of_interest import DVIDRegionOfInterest
from .repository import DVIDRepository
from .connection import DVIDConnection
