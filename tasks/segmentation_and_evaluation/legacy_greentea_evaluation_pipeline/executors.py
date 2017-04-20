import multiprocessing
import traceback

import sys
# try:
#     from concurrent.futures import ProcessPoolExecutor
# except ImportError:
#     pass


class LocalExecutorException(Exception):
    pass

def execute_function(wrapped_function, args):
    '''
    this function wraps function calls made to multiprocessing.Process workers. 
    it's useful because, if an error occurs in a worker, multiprocessing doesn't
        print traceback messages. But this does. 
    '''
    print(args)
    try:
        return wrapped_function(*args)
    except:
        traceback_text = "".join(traceback.format_exception(*sys.exc_info()))
        print(traceback_text)
        raise LocalExecutorException(traceback_text)


# class LocalExecutor(ProcessPoolExecutor):
#     def map(self, fn, *iterables, **kwargs):
#         futures_to_join = []
#         sequence_of_args = iterables[0]
#         for args in sequence_of_args:
#             kwargs = dict(wrapped_function=fn, args=(args,))
#             future = self.submit(execute_function, **kwargs)
#             futures_to_join.append(future)
#         result = [future.result() for future in futures_to_join]
#         return result


class LocalExecutor(object):
    def __init__(self, n_workers):
        # self.pool = multiprocessing.Pool(n_workers)
        pass

    def map(self, func, iterable, chunksize=None):
        results_to_get = []
        for args in iterable:
            kwargs = dict(wrapped_function=func, args=(args,))
            print(kwargs)
            result_async = self.pool.apply_async(execute_function, kwds=kwargs)
            results_to_get.append(result_async)
        results = [result_async.get() for result_async in results_to_get]
        return results

class SparkExecutor(object):
    def __init__(self, app_name):
        from pyspark import SparkConf, SparkContext
        conf = SparkConf().setAppName(app_name)
        conf.setAll([
            ("spark.task.cpus", "1"),
            ("spark.executor.cores", "2"),  # 2 gives 8, 1 gives 4
        ])
        self.spark_context = SparkContext(conf=conf)

    def map(self, func, iterable):
        result = self.spark_context \
                .parallelize(iterable, len(iterable)) \
                .map(func) \
                .collect()
        return result
