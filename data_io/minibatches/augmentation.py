import malis
import numpy as np


def augment_data_simple(dataset):
    nset = len(dataset)
    for iset in range(nset):
        for reflectz in range(2):
            for reflecty in range(2):
                for reflectx in range(2):
                    for swapxy in range(2):

                        if reflectz==0 and reflecty==0 and reflectx==0 and swapxy==0:
                            continue

                        dataset.append({})
                        dataset[-1]['name'] = dataset[iset]['name']+'_x'+str(reflectx)+'_y'+str(reflecty)+'_z'+str(reflectz)+'_xy'+str(swapxy)



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

                        dataset[-1]['label'] = malis.seg_to_affgraph(dataset[-1]['components'],dataset[-1]['nhood'])

                        dataset[-1]['reflectz']=reflectz
                        dataset[-1]['reflecty']=reflecty
                        dataset[-1]['reflectx']=reflectx
                        dataset[-1]['swapxy']=swapxy
    return dataset


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