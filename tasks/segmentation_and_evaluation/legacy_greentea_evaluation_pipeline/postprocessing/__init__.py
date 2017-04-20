from __future__ import print_function

import os
from pprint import pprint

import h5py
import matplotlib
import numpy as np


iters = range(10000, 1000001, 10000)
threshes = []
threshes.extend([j * 500 for j in range(1, 4)])
threshes.extend([j * 2000 for j in range(1, 6)])
threshes.extend([j * 30000 for j in range(2, 30)])

affinity_base_folder = '/nrs/turaga/grisaitisw/affinities/'
segmentation_base_folder = '/nrs/turaga/grisaitisw/affinities/'
model_names = sorted([m for m in os.listdir(segmentation_base_folder)
                      if any([s in m for s in (
                          'run_0712_2', 'run_0719_1', 'run_0723', 'run_09', 'fibsem', 'run_0822', 'run_1'
                        )])
                      ])
output_folders = [os.path.join(affinity_base_folder, m) for m in model_names]

pprint(model_names)

for of in output_folders:
    assert os.path.exists(of), of


def add_segmentation_to_dset(dataset, output_path, it, seg_suffix):
    model = os.path.basename(os.path.normpath(output_path))
    segmentation_file = dataset['name'] + seg_suffix
    fseg = os.path.join(output_path, str(it), segmentation_file)
    fseg_exists = os.path.exists(fseg)
    if fseg_exists:
        with h5py.File(fseg,'r') as fh5seg:
            dataset['seg_suffix'][model][it] = seg_suffix
            dataset['v_rand'][model][it] = np.array(fh5seg['v_rand'])
            dataset['v_rand_best'][model][it] = np.max(fh5seg['v_rand'])
            dataset['v_rand_split'][model][it] = np.array(fh5seg['v_rand_split'])
            dataset['v_rand_merge'][model][it] = np.array(fh5seg['v_rand_merge'])
            thresholds_for_this_model_iteration = fh5seg.get("thresholds", threshes)
            dataset['thresholds'][model][it] = np.array(thresholds_for_this_model_iteration)
            try:
                dataset['v_info'][model][it] = np.array(fh5seg['v_info'])
                dataset['v_info_best'][model][it] = np.min(fh5seg['v_info'])
                dataset['v_info_split'][model][it] = np.array(fh5seg['v_info_split'])
                dataset['v_info_merge'][model][it] = np.array(fh5seg['v_info_merge'])
            except:
                pass
            if len(dataset['thresholds'][model][it]) != len(dataset['v_rand'][model][it]):
                for key in dataset:
                    try:
                        dataset[key][model] = dict()
                    except:
                        pass
        return True
    else:
        return False


cmap = matplotlib.colors.ListedColormap(np.vstack(((0,0,0),np.random.rand(50000,3))))
