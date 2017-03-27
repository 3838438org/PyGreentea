from __future__ import print_function

import os
import time
from pprint import pprint

import data_io
import ipyparallel
import jobs.run_1208_6_010000 as job
import numpy as np
from data_io import OutOfCoreArray
from data_io.out_of_core_arrays import ZarrArrayHandler
from data_io.util.shape_chunking import chunkify_shape
from data_io.zero_padded_array import ZeroPaddedArray
from executors.ipyparallel_executor import executor
from tasks.processing import Processor

model = job.model
iteration = job.iteration
net_path = job.net_path
caffemodel_path = job.caffemodel_path
net_output_shape = job.net_output_shape
image = job.image
dname = job.dname


image = ZeroPaddedArray(image)
print(np.mean(image[0:10, 0:10, 0:10]))
print(image.shape)
batch_job_shape = tuple(x * 4 for x in net_output_shape)
spatial_chunk_shape = net_output_shape
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
affinity_chunks = [data_io.OffsettedArray(affinity_arr, (0,) + ic.offset, (3,) + ic.shape) for ic in image_chunks]

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
