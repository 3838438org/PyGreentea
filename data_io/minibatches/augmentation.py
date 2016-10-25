import malis
import numpy as np


def augment_data_simple(raw_datasets):
    for raw_dataset in raw_datasets:
        for reflectz in range(2):
            for reflecty in range(2):
                for reflectx in range(2):
                    for swapxy in range(2):
                        if reflectz==0 and reflecty==0 and reflectx==0 and swapxy==0:
                            continue
                        new_dataset = reflect_and_swap_dataset(raw_dataset, reflectx, reflecty, reflectz, swapxy)
                        raw_datasets.append(new_dataset)
    return raw_datasets


def reflect_and_swap_dataset(raw_dataset, reflectx, reflecty, reflectz, swapxy):
    new_dataset = dict()
    new_dataset['name'] = raw_dataset['name'] \
                          + '_x' + str(reflectx) \
                          + '_y' + str(reflecty) \
                          + '_z' + str(reflectz) \
                          + '_xy' + str(swapxy)
    new_dataset['reflectz'] = reflectz
    new_dataset['reflecty'] = reflecty
    new_dataset['reflectx'] = reflectx
    new_dataset['swapxy'] = swapxy
    for array_key in ["data", "components", "mask"]:
        if array_key in raw_dataset:
            array_copy = raw_dataset[array_key][:]
            array_copy = array_copy.reshape(array_copy.shape[-3:])
            if reflectz:
                array_copy = array_copy[::-1, :, :]
            if reflecty:
                array_copy = array_copy[:, ::-1, :]
            if reflectx:
                array_copy = array_copy[:, :, ::-1]
            if swapxy:
                array_copy = array_copy.transpose((0, 2, 1))
            new_dataset[array_key] = array_copy
    new_dataset['nhood'] = raw_dataset['nhood']
    new_dataset['label'] = malis.seg_to_affgraph(new_dataset['components'], new_dataset['nhood'])
    for key in raw_dataset:
        if key not in new_dataset:
            # add any other attributes we're not aware of
            new_dataset[key] = raw_dataset[key]
    return new_dataset


def augment_data_elastic(dataset,ncopy_per_dset):
    dsetout = []
    nset = len(dataset)
    for iset in range(nset):
        for icopy in range(ncopy_per_dset):
            reflectz = np.random.rand()>.5
            reflecty = np.random.rand()>.5
            reflectx = np.random.rand()>.5
            swapxy = np.random.rand()>.5

            dataset.append({})
            dataset[-1]['reflectz']=reflectz
            dataset[-1]['reflecty']=reflecty
            dataset[-1]['reflectx']=reflectx
            dataset[-1]['swapxy']=swapxy

            dataset[-1]['name'] = dataset[iset]['name']
            dataset[-1]['nhood'] = dataset[iset]['nhood']
            dataset[-1]['data'] = dataset[iset]['data'][:]
            dataset[-1]['components'] = dataset[iset]['components'][:]

            if reflectz:
                dataset[-1]['data']         = dataset[-1]['data'][::-1,:,:]
                dataset[-1]['components']   = dataset[-1]['components'][::-1,:,:]

            if reflecty:
                dataset[-1]['data']         = dataset[-1]['data'][:,::-1,:]
                dataset[-1]['components']   = dataset[-1]['components'][:,::-1,:]

            if reflectx:
                dataset[-1]['data']         = dataset[-1]['data'][:,:,::-1]
                dataset[-1]['components']   = dataset[-1]['components'][:,:,::-1]

            if swapxy:
                dataset[-1]['data']         = dataset[-1]['data'].transpose((0,2,1))
                dataset[-1]['components']   = dataset[-1]['components'].transpose((0,2,1))

            # elastic deformations

            dataset[-1]['label'] = malis.seg_to_affgraph(dataset[-1]['components'],dataset[-1]['nhood'])

    return dataset