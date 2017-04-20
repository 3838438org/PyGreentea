from __future__ import print_function

import os

import numpy as np
import pandas as pd


import source_datasets
from tasks.segmentation_and_evaluation.legacy_greentea_evaluation_pipeline.postprocessing import output_folders, iters, add_segmentation_to_dset
from tasks.entities.models.descriptions import descriptions
from tasks.entities.models.model_datasets import training_sets, partial_training_sets


rows_big = []
columns = ('model', 'iteration', 'data_name', 'threshold', 'description',
           'is_a_training_eval', 'is_a_partial_training_eval',
           'v_rand', 'v_rand_best', 'v_rand_merge', 'v_rand_split',
           'v_info', 'v_info_best', 'v_info_merge', 'v_info_split',
           )

dset = list(source_datasets.dset)

for d, dataset in enumerate(dset):
    for output_folder in output_folders:
        model_name = os.path.basename(os.path.normpath(output_folder))
        dset[d]['v_rand'][model_name] = dict()
        dset[d]['v_rand_best'][model_name] = dict()
        dset[d]['v_rand_split'][model_name] = dict()
        dset[d]['v_rand_merge'][model_name] = dict()
        dset[d]['v_info'][model_name] = dict()
        dset[d]['v_info_best'][model_name] = dict()
        dset[d]['v_info_split'][model_name] = dict()
        dset[d]['v_info_merge'][model_name] = dict()
        dset[d]["thresholds"][model_name] = dict()
        dset[d]['thresh_best'][model_name] = dict()
        dset[d]['seg_suffix'][model_name] = dict()
        for it in iters:
            added = False
            for seg_suffix in [
                "_seg_432_432_432_waterz_aff_squared_eval_2016-12-12.h5",
                "_seg_162_162_162_waterz_aff_squared_eval_2016-12-12.h5",
                # "_seg_432_432_432_waterz_2016-12-02_eval_2016-12-09.h5",
                # "_seg_162_162_162_waterz_2016-12-02_eval_2016-12-09.h5",
                # "_seg_432_432_432_waterz_2016-12-02.h5",
                # "_seg_162_162_162_waterz_2016-12-02.h5",
                # "_seg_432_432_432_zw_2016-11-17.h5",
                # "_seg_162_162_162_zw_2016-11-17.h5",
                # '_seg_432_432_432_eval0904_zw0.11.h5',
                # '_seg_162_162_162_eval0904_zw0.11.h5',
                # '_seg_432_432_432_eval0904_zw0.11_thick_spark.h5',
                # '_seg_best.h5',
                ]:
                added = add_segmentation_to_dset(dataset, output_folder, it, seg_suffix)
                if added:
                    break
            if added:
                print("Y",
                      model_name.ljust(30),
                      str(it).ljust(6),
                      dataset["name"].ljust(16),
                      "with %d thresholds" % len(dset[d]["thresholds"][model_name][it]),
                      "from", seg_suffix)
            else:
                print("N",
                      model_name.ljust(30),
                      str(it).ljust(6),
                      dataset["name"].ljust(16),
                      "not added")


def is_test_result(model, data_name):
    # training_sets, partial_training_sets
    is_a_training_eval = model in training_sets.get(data_name, [])
    is_a_partial_training_eval = is_a_training_eval or model in partial_training_sets.get(data_name, [])
    return not is_a_partial_training_eval


for dataset_result in dset:
    data_name = dataset_result["name"]
    models = sorted(dataset_result["v_rand"])
    for model in models:
        description = descriptions.get(model, "")
        iterations = dataset_result["v_rand"][model].keys()
        iterations = sorted(iterations)
        is_a_training_eval = model in training_sets.get(data_name, [])
        is_a_partial_training_eval = is_a_training_eval or model in partial_training_sets.get(data_name, [])
        for iteration in iterations:
            v_rand_best = max(dataset_result["v_rand"][model][iteration])
            v_info_best = min(dataset_result["v_info"][model][iteration])
            model_chunk = dict(
                data_name=data_name,
                model=model,
                description=description.replace("\n", " "),
                iteration=iteration,
                is_a_training_eval=is_a_training_eval,
                is_a_partial_training_eval=is_a_partial_training_eval,
                is_test_result=is_test_result(model, data_name),
                threshold=dataset_result["thresholds"][model].get(iteration, np.NaN),
                v_rand=dataset_result["v_rand"][model][iteration],
                v_rand_best=v_rand_best,
                v_rand_merge=dataset_result["v_rand_merge"][model][iteration],
                v_rand_split=dataset_result["v_rand_split"][model][iteration],
                v_info=dataset_result["v_info"][model][iteration],
                v_info_best=v_info_best,
                v_info_merge=dataset_result["v_info_merge"][model][iteration],
                v_info_split=dataset_result["v_info_split"][model][iteration],
            )
            chunk_df_all = pd.DataFrame(model_chunk, columns=columns)
            rows_big.append(chunk_df_all)

df_all_results = pd.concat(rows_big).reset_index(drop=True)
df_all_results.to_csv("df_big_1212_1.csv")
