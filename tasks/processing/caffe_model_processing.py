from __future__ import print_function

import os

import h5py
import numpy as np

import PyGreentea as pygt


def process_caffe_model(model_path, iteration, output_path, datasets_to_process, use_big_net_proto=True, test_device=0):
    """

    :param model_path: directory containing caffemodels
    :param iteration:
    :param output_path:
    :param datasets_to_process: dictionary containing:
        * "data" array-like with image
        * "name" used for the filename of the output hdf5 file containing the produced affinity map
        * "image_scaling_factor" to ensure image values are in the correct range for your network
    :param use_big_net_proto:
    :param test_device:
    :return:
    """

    protosmall = os.path.join(model_path, 'net_test.prototxt')
    protobig = os.path.join(model_path, 'net_test_big.prototxt')
    if os.path.exists(protobig) and use_big_net_proto:
        net_proto_path = protobig
    elif os.path.exists(protosmall):
        net_proto_path = protosmall
    else:
        print("Error: can't find test proto at model path", model_path)
        return

    caffemodel_filename = 'net_iter_{i}.caffemodel'.format(i=iteration)
    caffemodel_path = os.path.join(model_path, caffemodel_filename)
    if not os.path.exists(caffemodel_path):
        print("Iteration doesn't exist. Skipping", caffemodel_path)
        return

    # paths
    try:
        os.makedirs(output_path)
    except OSError, e:
        print("Output path already exists, probably", e.message)
    h5files = [os.path.join(output_path, d['name'] + '.h5') for d in datasets_to_process]
    alreadyProcessed = map(os.path.exists, h5files)
    if all(alreadyProcessed):
        print('Skipping', h5files)
        return

    # Set devices
    print("Using device: ", test_device)
    pygt.caffe.set_mode_gpu()
    pygt.caffe.set_device(test_device)

    print("Loading proto:", net_proto_path)
    print("Loading model:", caffemodel_path)
    net = pygt.caffe.Net(net_proto_path, caffemodel_path, pygt.caffe.TEST)

    # Process
    print("Processing", len(datasets_to_process), "volumes...")
    for i, dataset_to_process in enumerate(datasets_to_process):
        h5_filename = dataset_to_process['name'] + '.h5'
        h5_filepath = os.path.join(output_path, h5_filename)
        if os.path.exists(h5_filepath):
            print("Skipping because already exists", h5_filepath)
            continue
        file_name = h5_filepath + '.inprogress'
        with h5py.File(file_name, 'w') as outhdf5:
            print("Saving to", h5_filepath)
            affinity_shape = (3,) + dataset_to_process['data'].shape[-3:]
            target_array = outhdf5.create_dataset('main', affinity_shape, np.float32, chunks=(3, 58, 58, 58), compression="gzip")
            print("Processing to", h5_filepath)
            preds = pygt.process(net, [dataset_to_process], target_arrays=[target_array])
        os.rename(file_name, h5_filepath)
    return "done"
