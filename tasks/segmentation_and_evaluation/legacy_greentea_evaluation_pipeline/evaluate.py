from __future__ import print_function

import argparse
import itertools
import os
from pprint import pprint
import sys
import time

sys.path.append('/groups/turaga/home/grisaitisw/src/zwatershed_v0.11')

import h5py
import malis
import numpy as np
try:
    import zwatershed
except:
    raise ImportError("failed at beginning..." + str(sys.path))

assert zwatershed.__file__ == "/groups/turaga/home/grisaitisw/src/zwatershed_v0.11/"\
                              "zwatershed/__init__.pyc", zwatershed.__file__

from tasks.entities.source_data import group_models, group_datasets, aggregations, \
    group_widths, AffinityEvaluationData

from tasks.entities import source_data
print(source_data.__file__)

parser = argparse.ArgumentParser(description='Evaluate affinities')
parser.add_argument('--start', type=int, default=10000,
                    help='specifies the start iteration value')
parser.add_argument('--step', type=int, default=10000,
                    help="step between iterations, i.e. 'evaluate every X iterations'")
parser.add_argument('--stop', type=int, default=10000,
                    help="step between iterations, i.e. 'evaluate every X iterations'")
parser.add_argument('--affinity-folder', type=str,
                    default='/nrs/turaga/grisaitisw/affinities/',
                    help='directory containing affinity files')
parser.add_argument('--output-folder', type=str, 
                    default='/nrs/turaga/grisaitisw/affinities/',
                    help='where to save segmentations')
parser.add_argument('--n-local-workers', type=int, default=1,
                    help='number of worker processes to perform evaluations on'
                         'the local machine. Irrelevant if using Spark.')
parser.add_argument('--group-name', type=str,
                    default='all',
                    help='name of source data to evaluate')
parser.add_argument('--testing', type=bool,
                    default=False,
                    help='whether or not to use the test/ directory in --affinity-folder')

args = parser.parse_args()

start_iteration = args.start
step_iteration = args.step
stop_iteration = args.stop
output_base_folder = args.output_folder
affinity_base_folder = args.affinity_folder
testing = args.testing
n_local_workers = args.n_local_workers
group = args.group_name


using_pyspark = True
save_all_segs = False

if testing:
    model_names = ['fibsem32_srini_original_affs']
    iteration_values = range(10000, 100001, 10000)
    group = 'trvol-250-1-h5'
    group_model_iterations = list(
        (group, m, i)
        for m in reversed(model_names)
        for i in iteration_values
    )
    using_pyspark = False
else:
    iteration_values = range(start_iteration, stop_iteration + 1, step_iteration)
    if group == "all":
        groups = [
            'trvol-250-1-h5', 'trvol-250-2-h5',
            'tstvol-520-2-h5', 'tstvol-520-1-h5',
            # 'pb2', 'pb', '4440', '4400',
        ]
    else:
        groups = [group]
    group_model_iterations = list(
        (g, m, i)
        for g in groups
        for m in group_models[g]
        for i in iteration_values
    )

pprint(group_model_iterations)


def eval_model_iter(args):
    if using_pyspark:
        sys.path.append('/groups/turaga/home/grisaitisw/src/zwatershed_v0.11')
        try:
            import zwatershed
        except:
            raise ImportError("failed in function..." + str(sys.path))
        assert zwatershed.__file__ == "/groups/turaga/home/grisaitisw/src/zwatershed_v0.11/"\
                                      "zwatershed/__init__.pyc", zwatershed.__file__
    group, model_folder, iteration = args

    # settings
    threshes = []
    threshes.extend([j * 500 for j in range(1, 4)])
    threshes.extend([j * 2000 for j in range(1, 6)])
    threshes.extend([j * 30000 for j in range(2, 30)])
    threshes.extend([j * 30000 for j in range(30, 33)])
    threshes.extend([j * 50000 for j in range(20, 31)])
    threshes.sort()
    # threshes = threshes[0:2]
    nhood = malis.mknhood3d()
    datasets = group_datasets[group]
    # Evaluate
    affinity_path = os.path.join(affinity_base_folder, model_folder, str(iteration), '')
    if not os.path.exists(affinity_path):
        return
    output_path = os.path.join(output_base_folder, model_folder, str(iteration), '')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for d in datasets:
        # print(d)
        affinity_file_name = d['affinity_file_name']
        affinity_file_path = os.path.join(affinity_path, affinity_file_name)
        affinity_file_exists = os.path.exists(affinity_file_path)
        pred_seg_file_name = d['pred_seg_file_name']
        segmentation_file_done = os.path.join(output_path, pred_seg_file_name)
        d['segmentation_file_done'] = segmentation_file_done
        d['affinity_file_exists'] = affinity_file_exists
        predicted_segmentation_file_exists = os.path.exists(segmentation_file_done)
        d['predicted_segmentation_file_exists'] = predicted_segmentation_file_exists
        # print(affinity_file_exists, affinity_file_path)
        # print(predicted_segmentation_file_exists, segmentation_file_done)
        if affinity_file_exists and not predicted_segmentation_file_exists:
            affinity_evaluation_data = AffinityEvaluationData(d, affinity_file_path, nhood)
            components_cropped = affinity_evaluation_data.components_cropped
            aff = affinity_evaluation_data.aff
            print('About to make {segmentation_file_done}'.format(
                segmentation_file_done=segmentation_file_done))
            segmentation_file_inprogress = segmentation_file_done + '.inprogress'
            print('opening segmentation_file_inprogress:',
                  segmentation_file_inprogress)
            print(segmentation_file_inprogress)
            segh5 = h5py.File(segmentation_file_inprogress, 'w')
            if save_all_segs:
                segmentations_group = segh5.create_group("segmentations")
            else:
                segmentations_group = None
            metrics = segment_and_evaluate(aff, components_cropped, threshes, target_for_segmentations=segmentations_group)
            v_rands = metrics['rand']
            best_v_rand = max(v_rands)
            index_of_best_v_rand = v_rands.index(best_v_rand)
            best_threshold = threshes[index_of_best_v_rand]
            # convert to numpy arrays
            best_threshold = np.array([best_threshold], dtype=np.float64)
            threshes = np.array(metrics['thresholds'], dtype=np.float64)
            v_rand = np.array(v_rands, dtype=np.float32)
            v_rand_merge = np.array(metrics['rand_merge'], dtype=np.float32)
            v_rand_split = np.array(metrics['rand_split'], dtype=np.float32)
            v_info = np.array(metrics['voi'], dtype=np.float32)
            v_info_merge = np.array(metrics['voi_merge'], dtype=np.float32)
            v_info_split = np.array(metrics['voi_split'], dtype=np.float32)
            # if (group not in aggregations) and (not save_all_segs):
            #     results_gen = waterz.agglomerate(aff, [best_threshold])
            #     best_segmentation = [segmentation for segmentation in results_gen][0]
            #     segh5.create_dataset('main', data=best_segmentation, chunks=(32, 32, 32), compression="lzf")
            segh5.create_dataset('thresh_best', data=best_threshold)
            segh5.create_dataset('thresholds', data=threshes)
            segh5.create_dataset('v_rand', data=v_rand)
            segh5.create_dataset('v_rand_merge', data=v_rand_merge)
            segh5.create_dataset('v_rand_split', data=v_rand_split)
            segh5.create_dataset('v_info', data=v_info)
            segh5.create_dataset('v_info_merge', data=v_info_merge)
            segh5.create_dataset('v_info_split', data=v_info_split)
            segh5.create_dataset('finish_status', shape=(1,), dtype=np.int32, data=1)
            segh5.close()
            os.rename(segmentation_file_inprogress, segmentation_file_done)
            print('done saving to segmentation_file_done: ',
                  segmentation_file_done)
            d['predicted_segmentation_file_exists'] = True
        d['components'].file.close()
    if group in aggregations:
        new_width = group_widths[group]
        aggregation_file_name = group + '-agg' + '_seg_{0}.h5'.format(new_width)
        print([(d['name'],
                d['predicted_segmentation_file_exists'],
                os.path.exists(d['segmentation_file_done'])) for d in datasets])
        individual_evaluations_exist = all([d['predicted_segmentation_file_exists'] for d in datasets])
        aggregation_file_path = os.path.join(output_path, aggregation_file_name)
        aggregation_doesnt_exist = not os.path.exists(aggregation_file_path)
        if aggregation_doesnt_exist and individual_evaluations_exist:
            aggregation_file_path_inprogress = aggregation_file_path + ".inprogress"
            agg_segh5 = h5py.File(aggregation_file_path_inprogress, 'w')
            for metric_key in ['v_rand', 'v_rand_split', 'v_rand_merge', 'v_info', 'v_info_split', 'v_info_merge']:
                for d in datasets:
                    print(d['pred_seg_file_name'])
                evaluation_file_paths = [os.path.join(output_path, d['pred_seg_file_name']) for d in datasets]
                segh5_files = [h5py.File(path, 'r') for path in evaluation_file_paths]
                list_of_metric_arrays = [segh5[metric_key] for segh5 in segh5_files]
                metric_values = np.array(list_of_metric_arrays)
                mean_of_metric = np.mean(metric_values, axis=0)
                for file in segh5_files: file.close()
                agg_segh5.create_dataset(metric_key, mean_of_metric.shape, np.float32, data=mean_of_metric)
            v_rands = list(agg_segh5['v_rand'])
            print(v_rands)
            best_v_rand = max(v_rands)
            print(best_v_rand)
            index_of_best_v_rand = v_rands.index(best_v_rand)
            best_threshold = threshes[index_of_best_v_rand]
            print(best_threshold)
            d = datasets[2]
            affinity_file_path = os.path.join(affinity_path, d['name'] + '.h5')
            affinity_evaluation_data = AffinityEvaluationData(d, affinity_file_path, nhood)
            # results_gen = waterz.agglomerate(affinity_evaluation_data.aff, [best_threshold])
            # best_segmentation = [segmentation for segmentation in results_gen][0]
            # agg_segh5.create_dataset('main', best_segmentation.shape,
            #                          np.int32, data=best_segmentation)
            agg_segh5.close()
            os.rename(aggregation_file_path_inprogress, aggregation_file_path)
    return True


def segment_and_evaluate(affinities, true_components, thresholds, target_for_segmentations=None):
    results = list()
    for threshold in thresholds:
        try:
            _, zw_metrics = zwatershed.zwatershed_and_metrics(
                true_components, affinities, [threshold], [])
        except Exception, e:
            e.message = e.message + " with threshold {} and zwatershed at {}"\
                .format(threshold, zwatershed.__file__)
            raise e
        zw_metrics["threshold"] = threshold
        results.append(zw_metrics)
        pprint(zw_metrics)
        if zw_metrics["V_Rand_merge"][0] < 0.5:
            break
    metrics = dict()
    metrics['thresholds'] = [r["threshold"] for r in results]
    metrics["rand_merge"] = [r["V_Rand_merge"][0] for r in results]
    metrics["rand_split"] = [r["V_Rand_split"][0] for r in results]
    pprint(metrics)
    metrics["rand"] = [2.0 * vrm * vrs / (vrm + vrs) for vrm, vrs in zip(metrics["rand_merge"], metrics["rand_split"])]
    metrics["voi_merge"] = [r["V_Info_merge"][0] for r in results]
    metrics["voi_split"] = [r["V_Info_split"][0] for r in results]
    metrics["voi"] = [2.0 * vim * vis / (vim + vis) for vim, vis in zip(metrics["voi_merge"], metrics["voi_split"])]
    print(metrics['thresholds'],
          "have v_rand", metrics["rand"],
          "and v_info", metrics["voi"])
    return metrics


if using_pyspark:
    from .executors import SparkExecutor
    spark_app_name = 'eval_0904_zw11_' + group
    spark_executor = SparkExecutor(app_name=spark_app_name)
    map_wrapper = spark_executor.map
else:
    print("Using", n_local_workers, "workers")
    from .executors import LocalExecutor
    if n_local_workers == 1:
        map_wrapper = map
    elif n_local_workers > 1:
        local_executor = LocalExecutor(n_local_workers)
        map_wrapper = local_executor.map
    else:
        raise ValueError("n_local_workers ({}) must be a positive integer".format(n_local_workers))


def run_task(args):
    try:
        return eval_model_iter(args)
    except:
        import traceback
        with open("/groups/turaga/home/grisaitisw/spark_exceptions.txt", 'a') as f:
            f.write("Failure with args {}".format(args))
            traceback.print_exc(file=f)
    return False


if __name__ == '__main__':
    print('calling eval_model_iter with group_model_iterations')
    finish_status = map_wrapper(run_task, group_model_iterations)
    pprint(zip(group_model_iterations, finish_status))

