from __future__ import print_function

import multiprocessing
import sys
import time
import traceback
from operator import mul
from os.path import join

import h5py
import malis
import numpy as np
from functools import reduce

import PyGreentea as pygt
from .dataset_reading import get_numpy_dataset, reopen_dataset
from .util import get_slices_from_dataset_offset

''' where this will be used:
* train()
  * getting unordered batches of a dataset, specified with offset, input size and output size
  * behavior should be like a queue. at the end of each training iteration, train() specifies a replacement
* process()
  * getting batches from a dataset, specified with offset and input size
'''

if pygt.DEBUG:
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)


def update_shared_dataset(index_of_shared, index_of_which_dataset, input_slice,
                          output_slice, transform=True, make_dataset_offset=None):
    start_time = time.time()
    shared_dataset = shared_datasets[index_of_shared]
    dataset_is_ready = False
    while not dataset_is_ready:
        original_dataset = datasets[index_of_which_dataset]
        with reopen_dataset(original_dataset) as opened_dataset:
            dataset_numpy = get_numpy_dataset(opened_dataset, input_slice, output_slice, transform)
        if 'mask' in dataset_numpy:
            mask_threshold = float(original_dataset.get('mask_threshold', 0))
            mask_fraction_of_this_batch = np.mean(dataset_numpy['mask'])
            if mask_fraction_of_this_batch <= mask_threshold:
                if make_dataset_offset is not None:
                    index_of_which_dataset, offset = make_dataset_offset(datasets)
                    input_shape = tuple([s.stop - s.start for s in input_slice])
                    if output_slice is not None:
                        output_shape = tuple([s.stop - s.start for s in output_slice])
                    else:
                        output_shape = None
                    input_slice, output_slice = get_slices_from_dataset_offset(
                        offset, input_shape, output_shape)
                    if pygt.DEBUG:
                        print(multiprocessing.current_process().name,
                              "Skipping: dataset", index_of_which_dataset,
                              "output_slice", output_slice,
                              "mask %06.4f" % mask_fraction_of_this_batch)
                else:
                    return "DataLoader worker encountered a 100% masked" \
                           "datachunk, but doesn't know how to replace it."
            else:
                if pygt.DEBUG:
                    print(multiprocessing.current_process().name,
                          "Using: dataset", index_of_which_dataset,
                          "output_slice", output_slice,
                          "mask %06.4f" % mask_fraction_of_this_batch)
                dataset_is_ready = True
        else:
            dataset_is_ready = True
    for key in shared_dataset:
        source_array = dataset_numpy[key].astype(dtypes[key])
        target_mp_array = shared_dataset[key]
        if pygt.DEBUG:
            print(multiprocessing.current_process().name,
                  "storing dataset_numpy['", key, "']",
                  "with dtype", source_array.dtype,
                  "shape", source_array.shape)
        target_mp_array[:] = source_array.flatten()
    if pygt.DEBUG:
        print(multiprocessing.current_process().name,
              "Refreshing DataLoader dataset #", index_of_shared,
              "took %05.2fs" % (time.time() - start_time))
    return


class DataLoaderException(Exception):
    pass


def execute_function(function, function_kwargs):
    try:
        return function(**function_kwargs)
    except:
        raise DataLoaderException("".join(traceback.format_exception(*sys.exc_info())))


class DataLoader(object):
    def __init__(self, size, datasets, input_shape, output_shape=None, n_workers=1, dataset_offset_func=None):
        self.size = size
        self.datasets = datasets
        self.input_shape = input_shape
        self.outputs_are_ignored = output_shape is None
        self.output_shape = output_shape or (0, 0, 0)
        self.make_dataset_offset = dataset_offset_func
        self._list = list()
        self.shapes = {
            'data': (1,) + self.input_shape,
            'components': (1,) + self.output_shape,
            'label': (3,) + self.output_shape,
            'mask': (1,) + self.output_shape,
        }
        self.dtypes = {
            'data': np.float32,
            'components': np.int32,
            'label': np.int32,
            'mask': np.uint8,
        }
        self.keys_to_ignore = []
        if self.outputs_are_ignored:
            self.keys_to_ignore = ['label', 'components', 'mask']
            for output_key in self.keys_to_ignore:
                self.dtypes.pop(output_key)
                self.shapes.pop(output_key)
        sizes = dict()
        for key, shape in self.shapes.iteritems():
            sizes[key] = reduce(mul, shape)
        if pygt.DEBUG:
            print("sizes: ", sizes)
        self.shared_datasets = []
        for n in range(size):
            shared_dataset = dict()
            for key in self.dtypes:
                size = sizes[key]
                dtype = self.dtypes[key]
                ctype = type(np.ctypeslib.as_ctypes(dtype(0)))
                if pygt.DEBUG:
                    print("creating {key}'s multiprocessing.Array with "
                          "ctype {c} and size {s}".format(key=key, c=ctype, s=size))
                shared_dataset[key] = multiprocessing.Array(ctype, size, lock=False)
            self.shared_datasets.append(shared_dataset)
        self.pool = multiprocessing.Pool(
            processes=n_workers,
            initializer=self._initialize_pool,
            initargs=(),
            maxtasksperchild=10
        )
        self.ready_shared_datasets = []
        return

    def _initialize_pool(self):
        np.random.seed()
        global shared_datasets
        shared_datasets = self.shared_datasets
        global datasets
        datasets = self.datasets
        global dtypes
        dtypes = self.dtypes

    def get_dataset(self, copy=False):
        wait_start_time = None
        logging_time_threshold = 20
        logging_period = 1
        while len(self.ready_shared_datasets) == 0:
            if wait_start_time is None:
                wait_start_time = time.time()
                print("Waiting for dataset...")
            else:
                time_been_waiting = time.time() - wait_start_time
                if time_been_waiting > logging_time_threshold:
                    print("been waiting for", time_been_waiting)
                    logging_time_threshold += logging_period
            time.sleep(0.01)
        if wait_start_time is not None:
            print("Waited for dataset for %05.2fs" % (time.time() - wait_start_time))
        if pygt.DEBUG:
            print("Workers alive:", sum([proc.is_alive() for proc in self.pool._pool]))
        dataset_metadata = self.ready_shared_datasets.pop(0)
        index_of_shared_dataset = dataset_metadata['shared']
        index_of_given_dataset = dataset_metadata['real']
        new_dataset = dict()
        new_dataset['offset'] = dataset_metadata['offset']
        shared_dataset = self.shared_datasets[index_of_shared_dataset]
        given_dataset = self.datasets[index_of_given_dataset]
        for key in shared_dataset:
            dtype = self.dtypes[key]
            new_dataset[key] = np.frombuffer(shared_dataset[key], dtype)
            if pygt.DEBUG:
                print(key, new_dataset[key].shape, self.shapes[key])
            new_dataset[key] = new_dataset[key].reshape(self.shapes[key])
            if copy:
                new_dataset[key] = new_dataset[key].copy()
        for key in given_dataset:
            if key in shared_dataset or key in self.keys_to_ignore:
                # we already loaded it, or we want to ignore it
                pass
            else:
                # get the value from the original dataset dict
                new_dataset[key] = given_dataset[key]
        return new_dataset, index_of_shared_dataset

    def start_refreshing_shared_dataset(self, shared_dataset_index, offset=None, dataset_index=None, transform=True,
                                        wait=False):
        if offset is None or dataset_index is None:
            if self.make_dataset_offset is None:
                raise ValueError("Data loader wasn't given offset & which dataset to refresh, "
                                 "but can't make offsets itself.")
            dataset_index, offset = self.make_dataset_offset(self.datasets)
            if pygt.DEBUG:
                print("DataLoader decided to load dataset #", dataset_index, "at offset", offset)
        input_slice, output_slice = get_slices_from_dataset_offset(offset, self.input_shape, self.output_shape)
        dataset_metadata = dict(real=dataset_index, shared=shared_dataset_index, offset=offset)

        def pool_callback(return_value):
            return self.ready_shared_datasets.append(dataset_metadata)

        kwargs_for_refresh = dict(
            index_of_shared=shared_dataset_index,
            index_of_which_dataset=dataset_index,
            input_slice=input_slice,
            output_slice=output_slice,
            transform=transform,
            make_dataset_offset=self.make_dataset_offset,
        )
        async_result = self.pool.apply_async(
            func=execute_function,
            kwds=dict(
                function=update_shared_dataset,
                function_kwargs=kwargs_for_refresh,
            ),
            callback=pool_callback
        )
        if wait:
            final_result = async_result.get()
            if final_result is not None:
                print(final_result)  # probably an error
        return shared_dataset_index, async_result

    def destroy(self):
        self.pool.terminate()
        return


if __name__ == '__main__':
    path = '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/'
    datasets = []
    for dname in ['trvol-250-1-h5', 'trvol-250-2-h5']:
        datasets.append(
            dict({
                'name': dname,
                'data': h5py.File(join(path, dname, 'im_uint8.h5'), 'r')['main'],
                'components': h5py.File(join(path, dname, 'groundtruth_seg_thick.h5'), 'r')['main'],
                'nhood': malis.mknhood3d(),
                'transform': dict({'scale': (0.8, 1.2), 'shift': (-0.2, 0.2)})
            })
        )
    queue_size = 1
    q = DataLoader(queue_size,
                   datasets=datasets,
                   input_shape=(80, 80, 80),
                   output_shape=(60, 60, 60),
                   n_workers=1
                   )
    for j in range(len(datasets)):
        i = 0  # index of shared dataset to use
        shared_dataset_index, async_result = q.start_refreshing_shared_dataset(i, (15, 25, 35), j, wait=True)
        print("{}'s async_result.get(): {}".format(datasets[j]['name'], async_result.get()))
        dataset_result, index_of_shared_dataset = q.get_dataset(copy=False)
        print('start - ********************************************************************************')
        for key, value in dataset_result.iteritems():
            try:
                print(key, value.dtype, value.shape, type(value), value[0, 5, 50, 20:30], np.mean(value))
            except:
                print(key, value)
        print('end   - ********************************************************************************')
