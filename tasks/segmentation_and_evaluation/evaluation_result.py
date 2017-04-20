from __future__ import print_function

import os

import h5py
import numpy as np
import pandas as pd


def get_rands(merges, splits):
    return [2.0 * merge * split / (merge + split)
            for merge, split in zip(merges, splits)]


class EvaluationResult(object):
    def __init__(self, metrics, shape, fragment_source, discretize_queue, scoring_function, method_name, thresholds,
                 duration):
        self.metrics = metrics
        self.shape = shape
        self.fragment_source = fragment_source
        self.discretize_queue = discretize_queue
        self.scoring_function = scoring_function
        self.method_name = method_name
        self.thresholds = thresholds
        self.duration = duration

    def to_dataframe(self):
        df = pd.DataFrame(self.metrics)
        columns = list(sorted(df.columns))
        df["shape"] = repr(self.shape)
        df["fragment_source"] = self.fragment_source
        df["discretize_queue"] = self.discretize_queue
        df["scoring_function"] = self.scoring_function
        df["name"] = self.method_name
        df["threshold"] = self.thresholds
        df["duration"] = self.duration
        columns = \
            ["shape", "fragment_source", "discretize_queue", "scoring_function", "name", "threshold", "duration"] \
            + columns
        return df[columns]

    def save_to_dataframe(self):
        from datetime import datetime
        timestr = datetime.utcnow().strftime('%Y%m%d-%H:%M:%S.%f')[:-3]
        df = self.to_dataframe()
        filename = "results_{}_{}.csv".format("tstvol-520-2-h5", timestr)
        df.to_csv(filename)
        print("saved to", filename)
        print(self.duration, "seconds to segment and evaluate", self.method_name)

    def save_to_hdf5(self, filepath):
        rand_merges = self.metrics["V_Rand_merge"]
        rand_split = self.metrics["V_Rand_split"]
        v_info_merge = self.metrics["V_Info_merge"]
        v_info_split = self.metrics["V_Info_split"]
        rand = get_rands(rand_merges, rand_split)
        v_info = v_info_merge + v_info_split
        best_v_rand = max(rand)
        index_of_best_v_rand = rand.index(best_v_rand)
        best_threshold = self.thresholds[index_of_best_v_rand]
        file_inprogress = filepath + '.inprogress'
        segh5 = h5py.File(file_inprogress, 'w')
        segh5.create_dataset('thresh_best', data=best_threshold)
        segh5.create_dataset('thresholds', data=self.thresholds)
        segh5.create_dataset('v_rand', data=rand)
        segh5.create_dataset('v_rand_merge', data=rand_merges)
        segh5.create_dataset('v_rand_split', data=rand_split)
        segh5.create_dataset('v_info', data=v_info)
        segh5.create_dataset('v_info_merge', data=v_info_merge)
        segh5.create_dataset('v_info_split', data=v_info_split)
        segh5.create_dataset('finish_status', shape=(1,), dtype=np.int32, data=1)
        segh5.close()
        os.rename(file_inprogress, filepath)
        print('done saving to segmentation_file_done:', filepath)
