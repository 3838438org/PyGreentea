from __future__ import print_function

import os
from pprint import pprint
import time

import ipyparallel
import numpy as np

import data_io
import dvision
from PyGreentea.processing import Processor, ZarrAffinityHandler as AffinityHandler
from PyGreentea.processing.array_handlers import AffinitySaver
from data_io import OutOfCoreArray
from data_io.out_of_core_arrays import H5PyDArrayHandler, ZarrArrayHandler
from data_io.util.shape_chunking import chunkify_shape
from data_io.zero_padded_array import ZeroPaddedArray


model = "test_model"
iteration = "310000"
net_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_test_big.prototxt".format(model)
caffemodel_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_iter_{}.caffemodel".format(model, iteration)
net_output_shape = (116,) * 3

image = dvision.DVIDDataInstance("slowpoke3", 32773, "e402c09ddd0f45e980d9be6e9fcb9bd0", "grayscale")
dname = "fib25-e402c09"
# image = h5py.File("/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/tstvol-520-2-h5/im_uint8.h5", "r")["main"]
# dname = "tstvol-520-2-h5"
# image_opener = H5PyDArrayHandler("FlyEM/fibsem_medulla_7col/{}/im_uint8".format(dname), "main", dname)
# image = OutOfCoreArray(image_opener)
image = ZeroPaddedArray(image)
print(np.mean(image[0:10, 0:10, 0:10]))
print(image.shape)
batch_job_shape = (116 * 8,) * 3
spatial_chunk_shape = (116 * 4,) * 3
chunk_offsets = chunkify_shape(image.shape, batch_job_shape, integral_block_shape=spatial_chunk_shape)
print("chunk_offsets: ", chunk_offsets)

def make_offsetted_array(source_array, offset, shape):
    assert all([o <= l for o, l in zip(offset, source_array.shape)]), "offset exceeds source array"
    assert all([o >= 0 for o in offset]), "offset is negative; not sure if this'll behave appropriately"
    margin = tuple(l - o for l, o in zip(source_array.shape, offset))
    restricted_shape = tuple(min(s, m) for s, m in zip(shape, margin))
    print(offset, shape, restricted_shape)
    return data_io.OffsettedArray(source_array, offset, restricted_shape)

image_chunks = [make_offsetted_array(image, offset, batch_job_shape) for offset in chunk_offsets]

# 1. All job offsets are multiples of the affinity's chunk shape on disk
assert all([all([(o % scs == 0) for o, scs in zip(ic.offset, spatial_chunk_shape)]) for ic in image_chunks])
# 2. All job shapes are at least as big as the net's shape
assert all([all([s >= n for s, n in zip(ic.shape, net_output_shape)]) for ic in image_chunks])
# 3. All job regions are contained inside the source image
assert all([all([(0 <= o and o + s <= N) for o, s, N in zip(ic.offset, ic.shape, image.shape)]) for ic in image_chunks])
# 4. Batch regions do not overlap
# ... how to assert this?

# Ideally...
# 5. Job shapes are multiples of the net shape


# Two types of batchifying... Difference is at edge of source shape
# 1. fixed shape. last offset can overlap, start anywhere. Used in process()
# 2. Offsets don't overlap, start at multiples of a fundamental block shape. shape has a lower bound, but can vary. 

'''
Shapes...
1. image
2. job region
3. disk chunk
4. net output
'''


# affinity_shape = (3,) + tuple((int(math.ceil(1.0 * l / b))) * b for l, b in zip(image.shape, spatial_chunk_shape))
affinity_shape = (3,) + image.shape
affinity_chunk_shape = (3,) + spatial_chunk_shape
print(spatial_chunk_shape, image.shape, affinity_shape, affinity_chunk_shape)
print("chunks are size", np.prod(affinity_chunk_shape) * 4 / 100000, "MB")

root_dir = "/scratch/affinities"
# affinity_path = os.path.join(root_dir, model, iteration, dname + ".zarr")
#
# target = AffinityHandler(dname, model, iteration, affinity_shape,
#                          affinity_chunk_shape, root_dir="/scratch/affinities")

timestr = time.strftime("-%Y.%m.%d-%H.%M.%S")

affinity_opener = ZarrArrayHandler(
    path=os.path.join(root_dir, model, iteration, dname + timestr),
    key="main",
    name=dname,
    shape=affinity_shape,
    chunk_shape=affinity_chunk_shape,
    dtype=np.float32
)
affinity_arr = OutOfCoreArray(affinity_opener)
# target = AffinitySaver(affinity_arr)
affinity_chunks = [data_io.OffsettedArray(affinity_arr, (0,) + ic.offset, (3,) + ic.shape) for ic in image_chunks]

ipp_client = ipyparallel.Client(
    url_file="/groups/turaga/home/grisaitisw/.ipython/profile_greentea/security/ipcontroller-client.json",
    timeout=60 * 60  # 1 hour
)

executor = ipp_client.load_balanced_view()

processor = Processor(net_path, caffemodel_path, executor)

async_results = [processor.process(ic, ac) for ic, ac in zip(image_chunks, affinity_chunks)]
printed_ars = set()
while any([not ar.ready() for ar in async_results]):
    for ar in async_results:
        if ar.ready():
            if ar not in printed_ars:
                # print(dir(ar))
                print(ar.stdout)
                print(ar.stderr)
                printed_ars.add(ar)
for ar in async_results:
    try:
        done_result = ar.get()
    except ipyparallel.error.EngineError, e:
        print(e)
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

# slices = tuple(slice(o, o + l, 1) for o, l in zip(offsets, input_dims))
# from data_io.zero_padded_array import ZeroPaddedArray
# data_slice = ZeroPaddedArray(data_array)[slices]
