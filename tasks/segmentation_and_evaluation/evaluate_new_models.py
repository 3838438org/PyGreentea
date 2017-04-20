import logging
import sys

import numpy as np
from tasks.entities.models import AffinityModel, training_sets
from tasks.entities.source_data import group_models

from tasks.segmentation_and_evaluation.fragmenters import MahotasFragmenter
from tasks.segmentation_and_evaluation.seeding_and_segmenting import AffinityEvaluator
from tasks.segmentation_and_evaluation.affinity_evaluation_task import AffinityEvaluationTask, AlreadyEvaluatedException
from tasks.entities.datasets import EvaluationData, NoAffinityFoundException


logger = logging.Logger("Evaluation", level=logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


thresholds = list(np.arange(0.0, 0.4, 0.05)) + list(np.arange(0.4, 0.98, 0.005))
logger.debug("Making mahotas fragmenter")
mahotas_fragmenter = MahotasFragmenter()
logger.debug("Made mahotas fragmenter")


affinity_evaluators = [
    AffinityEvaluator(
        scoring_function="OneMinus<QuantileAffinity<AffinitiesType, 50, HistogramQuantileProviderSelect<256>::Value>>",
        discretize_queue=256,
        fragmenter=mahotas_fragmenter
    ),
    AffinityEvaluator(
        scoring_function="OneMinus<QuantileAffinity<AffinitiesType, 50>>",
        discretize_queue=0,
        fragmenter=None
    ),
]
data_names = ["tstvol-520-2-h5", "tstvol-520-1-h5"]
iterations = list(reversed(range(10000, 400001, 10000)))
model_names = reversed(training_sets["tstvol-520-1-h5"])
affinity_evaluation_tasks = [
    AffinityEvaluationTask(ae, EvaluationData(AffinityModel(m, i), dn), thresholds)
    for ae in affinity_evaluators
    for dn in data_names
    for m in group_models[dn]
    for i in iterations
    ]


"""
def run_job(task):
    logger.debug("starting task for {} with {}".format(task.data_source.description, task.affinity_evaluator.name))
    task.run()

pool = multiprocessing.Pool(20)
try:
    pool.map(run_job, affinity_evaluation_tasks)
except Exception as e:
    print(e)
finally:
    pool.terminate()
"""


class SparkExecutor(object):
    def __init__(self, app_name):
        from pyspark import SparkConf, SparkContext
        conf = SparkConf()
        print(dir(conf))
        # conf = conf.setAppName(app_name)
        conf.setAll([
            ("spark.app.name", app_name),
            ("spark.executor.memory", "16G"),
            ("spark.executor.cores", "1"),  # 2 gives 8, 1 gives 4
            ("spark.python.worker.memory", "512m"),
        ])
        print(conf.getAll())
        self.spark_context = SparkContext(conf=conf)
        print("SparkContext methods:")
        print(dir(self.spark_context))

    def map(self, func, iterable):
        result = self.spark_context \
                .parallelize(iterable, len(iterable)) \
                .map(func) \
                .collect()
        return result


def make_and_run_task(args):
    model_name, iteration, data_name, scoring_function, discretize_queue, fragment_source, thresholds = args
    model = AffinityModel(model_name, iteration)
    ds = EvaluationData(model, data_name)
    if fragment_source == 'mahotas':
        fragmenter = MahotasFragmenter()
    else:
        fragmenter = None
    ae = AffinityEvaluator(scoring_function, discretize_queue, fragmenter)
    aet = AffinityEvaluationTask(ae, ds, thresholds)
    try:
        aet.run()
    except (NoAffinityFoundException, AlreadyEvaluatedException):
        pass

args_list = [
    (
        task.data_source.model.name,
        task.data_source.model.iteration,
        task.data_source.name,
        task.affinity_evaluator.scoring_function,
        task.affinity_evaluator.discretize_queue,
        task.affinity_evaluator.fragment_source,
        task.thresholds
    )
    for task in affinity_evaluation_tasks
]


executor = SparkExecutor("Evaluate little volumes")
executor.map(make_and_run_task, args_list)
