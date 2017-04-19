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
