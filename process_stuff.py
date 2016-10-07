from __future__ import print_function

from contextlib import contextmanager
from pprint import pprint
import time

import h5pyd
import ipyparallel
import numpy as np

import data_io
from data_io.util.shape_chunking import chunkify_shape
import dvision


image = dvision.DVIDDataInstance("slowpoke3", 32773, "e402c09ddd0f45e980d9be6e9fcb9bd0", "grayscale")
chunk_shape = (116 * 4,) * 3
chunk_offsets = chunkify_shape(image.shape, chunk_shape=chunk_shape)

image_chunks = [data_io.OffsettedArray(image, offset, chunk_shape) for offset in chunk_offsets]

from PyGreentea.processing import Processor
from PyGreentea.processing.affinity_handler import SimpleZarrAffinityHandler as AffinityHandler


model = "test_model"
iteration = 310000
affinity_shape = (3,) + image.shape
affinity_chunk_shape = (3,) + (29 * 2,) * 3
target = AffinityHandler("fib25-e402c09", model, iteration, affinity_shape,
                         affinity_chunk_shape, root_dir="/scratch/affinities")
target.initialize_array()

ipp_client = ipyparallel.Client(profile="greentea")
executor = ipp_client.load_balanced_view()

net_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_test_big.prototxt".format(model)
caffemodel_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_iter_{}.caffemodel".format(model, iteration)
processor = Processor(net_path, caffemodel_path, executor)

async_results = [processor.process(ic, target) for ic in image_chunks]
while any([not ar.ready() for ar in async_results]):
    for ar in async_results:
        if not ar.ready():
            with open('/groups/turaga/home/grisaitisw/Desktop/ipyparallel-log-{}.txt'.format(ar.engine_id), 'w') as f:
                print(ar.stdout)
                print(ar.stderr)
    time.sleep(1)
for ar in async_results:
    done_result = ar.get()
    for item in done_result:
        pprint(item)

# with target.get_array(mode="r") as a:
#     print(a[0, 0:2, 0:2, 0:2])
#     for offset in chunk_offsets:
#         origin = (0,) + offset
#         lengths = (3,) + chunk_shape
#         slices = tuple(slice(o, o+2) for o, l in zip(origin, lengths))
#         print(slices)
#         print(a[slices])
#         slices = tuple(slice(o, o+l) for o, l in zip(origin, lengths))
#         print(np.mean(a[slices]))
