from __future__ import print_function

import itertools
from pprint import pprint

from tasks.segmentation_and_evaluation import AffinityEvaluator
# from tasks.segmentation_and_evaluation import thresholds
thresholds = [0.1 * x for x in range(10)]
from tasks.segmentation_and_evaluation.fragmenters import MahotasFragmenter

mahotas_fragmenter = MahotasFragmenter()

arg_combinations = itertools.product(
    ("OneMinus<QuantileAffinity<AffinitiesType, 50>>",
     "OneMinus<QuantileAffinity<AffinitiesType, 50, HistogramQuantileProviderSelect<256>::Value>>"),
    (0, 256),
    (None, mahotas_fragmenter),
)

evaluators = [
    AffinityEvaluator(scoring_function=sf, discretize_queue=dq, fragmenter=f)
    for sf, dq, f in arg_combinations
    ]

pprint(evaluators)


def run_and_save_segmentation(evaluator):
    from tasks.entities.datasets import test as data_source
    evaluator.segment_and_evaluate(data_source.affinity, data_source.truth, thresholds)


# def map(f, iterable):
#     import multiprocessing
#     pool = multiprocessing.Pool(processes=len(evaluators))
#     pool.map(f, iterable)
#     pool.terminate()


map(run_and_save_segmentation, evaluators)
