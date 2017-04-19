import os
from pprint import pprint

import h5py

from tasks.entities.models import training_sets, partial_training_sets, fibsem17_replicas, fibsem32_replicas


group_models = dict()
for group in ("tstvol-520-1-h5", "pb", "4400",):
    # models = training_sets.get(group, []) + partial_training_sets.get(group, [])
    group_models[group] = []
    group_models[group].extend(reversed(sorted(m for m in training_sets.get(group, []) if m[0:7] in ('2017.03', '2017.04'))))
    group_models[group].extend(["fibsem17_srini_original_affs", "fibsem17_srini_snapshots"])
    group_models[group].extend(["fibsem32_srini_original_affs", "fibsem32_srini_snapshots"])
    group_models[group].extend(reversed(sorted(training_sets.get(group, []))))
    group_models[group].extend(reversed(sorted(partial_training_sets.get(group, []))))
    group_models[group].extend(["run_0712_2"])
    group_models[group].extend(["run_1111_2"])
    group_models[group].extend(reversed(fibsem17_replicas))
    group_models[group].extend(reversed(fibsem32_replicas))
group_models['tstvol-520-2-h5'] = list(group_models["tstvol-520-1-h5"])
group_models['trvol-250-1-h5'] = list(group_models["tstvol-520-1-h5"])
group_models['trvol-250-2-h5'] = list(group_models["tstvol-520-1-h5"])
group_models['pb2'] = list(group_models.get("pb", []))
group_models['4440'] = list(group_models.get("4400", []))

def dedupe(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

for group in group_models:
    group_models[group] = dedupe(group_models[group])

# pprint(group_models["tstvol-520-2-h5"])

group_truth_eroded_file_names = {
    'trvol-250-1-h5': 'groundtruth_seg_thick.h5',
    'trvol-250-2-h5': 'groundtruth_seg_thick.h5',
    'tstvol-520-1-h5': 'groundtruth_seg_thick.h5',
    'tstvol-520-2-h5': 'groundtruth_seg_thick.h5',
    '4400': 'components_eroded_by_1.h5',
    '4440': 'components_eroded_by_1.h5',
    'pb': 'components_eroded_by_1.h5',
    'pb2': 'components_eroded_by_1.h5',
    'SegEM-train': 'groundtruth_seg_thick.h5',
    'SegEM-test': 'groundtruth_seg_thick.h5',
}

group_truth_file_names = group_truth_eroded_file_names

group_true_segmentation_file_paths = {
    'trvol-250-1-h5': '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/trvol-250-1-h5/',
    'trvol-250-2-h5': '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/trvol-250-2-h5/',
    'tstvol-520-1-h5': '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/tstvol-520-1-h5/',
    'tstvol-520-2-h5': '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/tstvol-520-2-h5/',
    '4400': '/nrs/turaga/grisaitisw/data/toufiq_mushroom/4400/',
    '4440': '/nrs/turaga/grisaitisw/data/toufiq_mushroom/4440/',
    'pb': '/nrs/turaga/grisaitisw/data/pb/pb/',
    'pb2': '/nrs/turaga/grisaitisw/data/pb/pb2/',
    'SegEM-test': '/nrs/turaga/grisaitisw/data/augmented/SegEM/SegEM_challenge/volumeTrainingData3/',
}
group_true_segmentation_file_paths['SegEM-train'] = group_true_segmentation_file_paths['SegEM-test']

segem_volumes = os.listdir(group_true_segmentation_file_paths['SegEM-test'])
segem_volumes = sorted(segem_volumes)

dname_volumes = {
    'fibsem-medulla': ['tstvol-520-1-h5', 'tstvol-520-2-h5'],
    'pb': ['pb', 'pb2'],
    'mb': ['4400', '4440'],
    'SegEM': segem_volumes
}


group_volumes = {
    'SegEM-test': [x for x in segem_volumes if x in [str(y) for y in range(1000, 1150)]],
    'SegEM-train': [x for x in segem_volumes if x in [str(y) for y in range(0, 1000)]],
}


group_dnames = {
    'SegEM-test': 'SegEM',
    'SegEM-train': 'SegEM',
}

aggregations = ['SegEM-test', 'SegEM-train']

group_segmentation_h5_keys = {
    'trvol-250-1-h5': 'main',
    'trvol-250-2-h5': 'main',
    'tstvol-520-1-h5': 'main',
    'tstvol-520-2-h5': 'main',
    '4400': 'stack',
    '4440': 'stack',
    'pb': 'stack',
    'pb2': 'stack',
    'SegEM-test': 'y0_x0_xy0_angle000.0',
    'SegEM-train': 'y0_x0_xy0_angle000.0',
}

group_new_shapes = {
    'trvol-250-1-h5': (162,) * 3,
    'trvol-250-2-h5': (162,) * 3,
    'tstvol-520-1-h5': (432,) * 3,
    'tstvol-520-2-h5': (432,) * 3,
    '4400': (162,) * 3,
    '4440': (162,) * 3,
    'pb': (432,) * 3,
    'pb2': (432,) * 3,
    'SegEM-test': (100, 150, 150),
    'SegEM-train': (100, 150, 150),
}


group_widths = dict()
for key, shape in group_new_shapes.iteritems():
    group_widths[key] = "_".join([str(s) for s in shape])

class GroundTruthDatasets(object):
    def __getitem__(self, item):
        return get_ground_truth_datasets(item)

group_datasets = GroundTruthDatasets()

def get_ground_truth_datasets(group):
    datasets = []
    dname = group_dnames.get(group, group)
    # Load source data
    if group in group_volumes:
        subvolumes = group_volumes[group]
        for volume in subvolumes:
            true_seg_file_path = os.path.join(
                group_true_segmentation_file_paths[group],
                volume,
                group_truth_file_names[group])
            if os.path.exists(true_seg_file_path):
                dataset = dict()
                dataset['name'] = "{}-{}".format(dname, volume)
                dataset['seg_file_path'] = true_seg_file_path
                datasets.append(dataset)
    else:
        dataset = dict()
        dataset['name'] = dname
        dataset['seg_file_path'] = os.path.join(
            group_true_segmentation_file_paths[group],
            group_truth_file_names[group])
        datasets.append(dataset)
    for dataset in datasets:
        dataset['dname'] = dname
        dataset['affinity_file_name'] = dataset['name'] + '.h5'
        h5_key = group_segmentation_h5_keys[group]
        new_width = group_widths[group]
        pred_seg_file_name = dataset['name'] + '_seg_{0}_zw0.11_eval_2017-01-04.h5'.format(new_width)
        dataset['pred_seg_file_name'] = pred_seg_file_name
        h5_file = h5py.File(dataset['seg_file_path'], 'r')
        dataset['components'] = h5_file[h5_key]
        dataset['new_shape'] = group_new_shapes[group]
    return datasets


if __name__ == '__main__':
    for group in group_models:
        get_ground_truth_datasets(group)
