from __future__ import print_function

import time

import waterz

from evaluation_result import EvaluationResult


scoring_function_abbreviations = {
    "OneMinus<QuantileAffinity<AffinitiesType, 50>>": "normal",
    "OneMinus<QuantileAffinity<AffinitiesType, 50, HistogramQuantileProviderSelect<256>::Value>>": "hist",
}


class AffinityEvaluator(object):
    def __init__(self, scoring_function="OneMinus<QuantileAffinity<AffinitiesType, 50>>", discretize_queue=0, fragmenter=None):
        self.scoring_function = scoring_function
        self.discretize_queue = discretize_queue
        self.fragmenter = fragmenter
        try:
            self.fragment_source = fragmenter.name
        except:
            self.fragment_source = "waterz"
        self.scoring_function_abbreviation = scoring_function_abbreviations.get(scoring_function, scoring_function)
        self.name = "{}_fragments-disc_queue_{}-merge_{}"\
            .format(self.fragment_source, self.discretize_queue, self.scoring_function_abbreviation)

    def segment_and_evaluate(self, affinity, truth, thresholds):
        try:
            fragments = self.fragmenter(affinity)
            fragment_source = self.fragmenter.name
        except:
            fragments = None
            fragment_source = "waterz"
        agglomerator = waterz.agglomerate(
            affinity,
            thresholds,
            gt=truth,
            scoring_function=self.scoring_function,
            discretize_queue=self.discretize_queue,
            fragments=fragments,
        )
        start = time.time()
        metrics = [m for _, m in agglomerator]
        duration = time.time() - start
        return EvaluationResult(
            metrics, affinity.shape, fragment_source, self.discretize_queue,
            self.scoring_function, self.name, thresholds, duration)

    def run_with_data_source(self, data_source, thresholds):
        r = self.segment_and_evaluate(data_source.affinity, data_source.truth, thresholds)
        return r
