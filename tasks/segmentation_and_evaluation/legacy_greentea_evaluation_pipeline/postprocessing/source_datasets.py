import h5py
import malis
from os.path import join

import matplotlib
import numpy as np

dset = []
nhood = malis.mknhood3d()


volume_names = [
    'tstvol-520-1-h5',
    'tstvol-520-2-h5',
    '4400',
    '4440',
    'pb',
    'pb2',
]


for name in volume_names:
    d = dict()
    d['name'] = name
    d['nhood'] = nhood
    d['v_rand'] = dict()
    d['v_rand_reweighted'] = dict()
    d['v_rand_reweighted_best'] = dict()
    d['v_rand_best'] = dict()
    d['v_rand_split'] = dict()
    d['v_rand_merge'] = dict()
    d['v_info'] = dict()
    d['v_info_reweighted'] = dict()
    d['v_info_reweighted_best'] = dict()
    d['v_info_best'] = dict()
    d['v_info_split'] = dict()
    d['v_info_merge'] = dict()
    d['thresh_best'] = dict()
    d['seg_suffix'] = dict()
    d["thresholds"] = dict()
    dset.append(d)


