from __future__ import print_function

import argparse
import itertools
import os
from os.path import join

import h5py
from tasks.entities.source_data import group_models
from tasks.processing.caffe_model_processing import process_caffe_model

parser = argparse.ArgumentParser(description='Process models')
parser.add_argument('--start', type=int, default=10000,
                    help='specifies the start iteration value')
parser.add_argument('--step', type=int, default=10000,
                    help="step between iterations, i.e. 'process every X iterations'")
parser.add_argument('--model-folder', type=str, 
                    default='/groups/turaga/home/grisaitisw/experiments/',
                    help='directory containing caffemodel files')
parser.add_argument('--output-folder', type=str, 
                    default='/nrs/turaga/grisaitisw/affinities/',
                    help='where to save affinities')
parser.add_argument('--volume-name', type=str,
                    default='tstvol-520-2-h5',
                    help='name of source data to process')

args = parser.parse_args()
start_iteration = args.start
step_iteration = args.step
model_base_folder = args.model_folder
output_base_folder = args.output_folder
dname = args.volume_name

iterations = range(start_iteration, 400001, step_iteration)

data_models = group_models

from pprint import pprint
pprint(data_models)

device = 0
using_in_memory = False

# load dataset
data_file_names = {
    'trvol-250-1-h5': 'im_uint8.h5',
    'trvol-250-2-h5': 'im_uint8.h5',
    'tstvol-520-1-h5': 'im_uint8.h5',
    'tstvol-520-2-h5': 'im_uint8.h5',
    '4400': 'image_from_png_files.h5',
    '4440': 'image_from_png_files.h5',
    'pb': 'image_from_png_files.h5',
    'pb2': 'image_from_png_files.h5',
    'SegEM': 'im_uint8.h5'
}

data_file_paths = {
    'trvol-250-1-h5': '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/trvol-250-1-h5/',
    'trvol-250-2-h5': '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/trvol-250-2-h5/',
    'tstvol-520-1-h5': '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/tstvol-520-1-h5/',
    'tstvol-520-2-h5': '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/tstvol-520-2-h5/',
    '4400': '/nrs/turaga/grisaitisw/data/toufiq_mushroom/4400/',
    '4440': '/nrs/turaga/grisaitisw/data/toufiq_mushroom/4440/',
    'pb': '/nrs/turaga/grisaitisw/data/pb/pb/',
    'pb2': '/nrs/turaga/grisaitisw/data/pb/pb2/',
    'SegEM': '/nrs/turaga/grisaitisw/data/augmented/SegEM/SegEM_challenge/volumeTrainingData3/',
}

segem_volumes = os.listdir(data_file_paths['SegEM'])
segem_volumes = sorted([x for x in segem_volumes if x in [str(y) for y in range(0, 2000)]])

image_scaling_factors = {
    'trvol-250-1-h5': 0.5 ** 8,
    'trvol-250-2-h5': 0.5 ** 8,
    'tstvol-520-1-h5': 0.5 ** 8,
    'tstvol-520-2-h5': 0.5 ** 8,
    '4400': 1,
    '4440': 1,
    'pb': 1,
    'pb2': 1,
    'SegEM': 0.5 ** 8,
}

def get_dataset(dname, volume=None, h5_key='main'):
    dataset = dict()
    data_file_path = data_file_paths[dname]
    data_file_name = data_file_names[dname]
    if volume:
        dataset['data'] = h5py.File(join(data_file_path, volume, data_file_name), 'r')[h5_key]
        dataset['name'] = "{}-{}".format(dname, volume)
    else:
        dataset['data'] = h5py.File(join(data_file_path, data_file_name), 'r')[h5_key]
        dataset['name'] = dname
    dataset['image_scaling_factor'] = image_scaling_factors[dname]
    return dataset

use_net_test_big = {
    'trvol-250-1-h5': True,
    'trvol-250-2-h5': True,
    'tstvol-520-1-h5': True,
    'tstvol-520-2-h5': True,
    '4400': True,
    '4440': True,
    'pb': True,
    'pb2': True,
    'SegEM': True
}
default_use_net_test_big = True

# Process!

if dname == "all":
    dnames_to_process = ["tstvol-520-2-h5", "pb2", "tstvol-520-1-h5", "pb"]
else:
    dnames_to_process = [dname]

def process_most_important_thing():
    jobs = (
        (dname, model, iteration)
        for dname in dnames_to_process
        for model in group_models[dname]
        for iteration in iterations
    )
    for dname, model, iteration in jobs:
        if dname == "SegEM":
            raise Exception("add SegEM volume support")
        modelpath = os.path.join(model_base_folder, model, '')
        outputpath = os.path.join(output_base_folder, model, str(iteration), '')
        aff_file_path = os.path.join(outputpath, dname + '.h5')
        if os.path.exists(aff_file_path):
            continue
        dataset = get_dataset(dname)
        use_big = use_net_test_big.get(dname, default_use_net_test_big)
        # print("Processing", modelpath, "@", iteration, "with dataset", dname)
        try:
            result = process_caffe_model(modelpath, iteration, outputpath, [dataset], use_big_net_proto=use_big,
                                         test_device=device)
        except SystemError, e:
            print(str(e))
            continue
        if result == "done":
            return

while True:
    process_most_important_thing()
