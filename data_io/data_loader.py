from __future__ import print_function

import multiprocessing
import sys
import time
import traceback
from functools import reduce
from operator import mul

import numpy as np
from data_io import logger
from data_io.minibatches.greentea_minibatch import get_numpy_dataset
from data_io.out_of_core_arrays.array_reopening import reopen_dataset

from .util import get_slices_from_dataset_offset

''' where this will be used:
* train()
  * getting unordered batches of a dataset, specified with offset, input size and output size
  * behavior should be like a queue. at the end of each training iteration, train() specifies a replacement
* process()
  * getting batches from a dataset, specified with offset and input size
'''


def update_shared_dataset(index_of_shared, index_of_which_dataset, input_slice,
                          output_slice, transform=True, make_dataset_offset=None):
    start_time = time.time()
    shared_dataset = shared_datasets[index_of_shared]
    dataset_is_ready = False
    while not dataset_is_ready:
        original_dataset = datasets[index_of_which_dataset]
        with reopen_dataset(original_dataset) as opened_dataset:
            dataset_numpy = get_numpy_dataset(opened_dataset, input_slice, output_slice, transform)
        if dataset_numpy is None:
            message = "Skipping dataset #{0} at output_slice {1}"\
                .format(index_of_which_dataset, output_slice)
            logger.debug(message)
            if make_dataset_offset is not None:
                index_of_which_dataset, offset = make_dataset_offset(datasets)
                input_shape = tuple([s.stop - s.start for s in input_slice])
                if output_slice is not None:
                    output_shape = tuple([s.stop - s.start for s in output_slice])
                else:
                    output_shape = None
                input_slice, output_slice = get_slices_from_dataset_offset(
                    offset, input_shape, output_shape)
            else:
                return "DataLoader worker encountered a 100% masked" \
                       "datachunk, but doesn't know how to replace it."
        else:
            message = "Using dataset #{0} at output_slice {1}"\
                .format(index_of_which_dataset, output_slice)
            logger.debug(message)
            dataset_is_ready = True
    for key in shared_dataset:
        source_array = dataset_numpy[key].astype(dtypes[key])
        target_mp_array = shared_dataset[key]
        message = "storing dataset_numpy['{key}'] ({dt}, {shape})".format(
            key=key, dt=source_array.dtype, shape=source_array.shape)
        logger.debug(message)
        target_mp_array[:] = source_array.flatten()
    message = "Refreshing DataLoader dataset #{0} took {1}".format(
        index_of_shared, "%05.2fs" % (time.time() - start_time))
    logger.debug(message)
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
        self.n_workers = n_workers
        self.make_dataset_offset = dataset_offset_func
        self._list = list()
        self.shapes = {
            'data': (1,) + self.input_shape,
            'components': (1,) + self.output_shape,
            'components_negative': (1,) + self.output_shape,
            'label': (3,) + self.output_shape,
            'mask': (1,) + self.output_shape,
        }
        self.dtypes = {
            'data': np.float32,
            'components': np.int32,
            'components_negative': np.int32,
            'label': np.int32,
            'mask': np.uint8,
        }
        self.keys_to_ignore = []
        if self.outputs_are_ignored:
            self.keys_to_ignore = ['label', 'components', 'components_negative', 'mask']
            for output_key in self.keys_to_ignore:
                self.dtypes.pop(output_key)
                self.shapes.pop(output_key)
        self.refreshes_in_progress = []
        self.pool = None
        self.reset_pool()
        return

    def _initialize_worker(self):
        np.random.seed()
        global shared_datasets
        shared_datasets = self.shared_datasets
        global datasets
        datasets = self.datasets
        global dtypes
        dtypes = self.dtypes

    def _initialize_shared_memory_arrays(self):
        shared_datasets = []
        sizes = dict()
        for key, shape in self.shapes.iteritems():
            sizes[key] = reduce(mul, shape)
        logger.debug("sizes: {}".format(sizes))
        for n in range(self.size):
            shared_dataset = dict()
            for key in self.dtypes:
                size = sizes[key]
                dtype = self.dtypes[key]
                ctype = type(np.ctypeslib.as_ctypes(dtype(0)))
                message = "creating {key}'s multiprocessing.Array with ctype {c} "\
                          "and size {s}".format(key=key, c=ctype, s=size)
                logger.debug(message)
                shared_dataset[key] = multiprocessing.Array(ctype, size, lock=False)
            shared_datasets.append(shared_dataset)
        return shared_datasets

    def reset_pool(self):
        self.destroy()
        self.ready_shared_datasets = []
        self.shared_datasets = self._initialize_shared_memory_arrays()
        self.pool = multiprocessing.Pool(
            processes=self.n_workers,
            initializer=self._initialize_worker,
            initargs=(),
            maxtasksperchild=1000
        )
        refreshes_that_were_in_progress = list(self.refreshes_in_progress)
        self.refreshes_in_progress = []
        for dataset_metadata in refreshes_that_were_in_progress:
            self.start_refreshing_shared_dataset(
                shared_dataset_index=dataset_metadata["shared"],
                offset=dataset_metadata["offset"],
                dataset_index=dataset_metadata["real"],
                transform=dataset_metadata["transform"],
                wait=True
            )

    def get_dataset(self, copy=False, timeout=5 * 60):
        wait_start_time = time.time()
        time_been_waiting = 0
        logging_time_threshold = 20
        logging_period = 1
        while len(self.ready_shared_datasets) == 0:
            time_been_waiting = time.time() - wait_start_time
            if time_been_waiting > logging_time_threshold:
                print("been waiting for", time_been_waiting)
                logging_time_threshold += logging_period
            if timeout:
                if time_been_waiting > timeout:
                    print("Still waiting on", self.refreshes_in_progress)
                    self.reset_pool()
            time.sleep(0.01)
        if time_been_waiting > 0:
            print("Waited for dataset for %05.2fs" % (time.time() - wait_start_time))
        dataset_metadata = self.ready_shared_datasets.pop(0)
        index_of_shared_dataset = dataset_metadata['shared']
        index_of_given_dataset = dataset_metadata['real']
        new_dataset = dict()
        new_dataset['offset'] = dataset_metadata['offset']
        new_dataset["dataset_index"] = index_of_given_dataset
        shared_dataset = self.shared_datasets[index_of_shared_dataset]
        given_dataset = self.datasets[index_of_given_dataset]
        for key in shared_dataset:
            dtype = self.dtypes[key]
            new_dataset[key] = np.frombuffer(shared_dataset[key], dtype)
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
            message = "DataLoader decided to load dataset #{0} at offset {1}"\
                .format(dataset_index, offset)
            logger.debug(message)
        input_slice, output_slice = get_slices_from_dataset_offset(offset, self.input_shape, self.output_shape)
        dataset_metadata = dict(real=dataset_index, shared=shared_dataset_index, offset=offset, transform=transform)
        self.refreshes_in_progress.append(dataset_metadata)

        def pool_callback(return_value):
            self.refreshes_in_progress.remove(dataset_metadata)
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
        try:
            self.pool.terminate()
            self.pool.join()
        except AttributeError:
            pass
        return
