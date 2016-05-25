import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

max_component_value = 1

cmap = matplotlib.colors.ListedColormap(np.vstack(((0, 0, 0), np.random.rand(max(10000, max_component_value), 3))))


def showme(dset, z):
    imshape = dset['data'].shape
    predshape = dset['pred'].shape[1:]
    off = [(s[0] - s[1]) / 2 for s in zip(imshape, predshape)]
    slc1 = slice(off[1], predshape[1] - off[1])
    slc2 = slice(off[2], predshape[2] - off[2])
    raw_slc = np.transpose(np.squeeze(dset['data'][z, slc1, slc2]), (1, 0))
    if 'components' in dset:
        gt_slc = np.transpose(np.squeeze(dset['components'][z, slc1, slc2]), (1, 0))
        aff_slc = np.transpose(np.squeeze(dset['label'][:3, z, slc1, slc2]), (2, 1, 0)).astype(np.float)
    test_slc = np.transpose(np.squeeze(dset['pred'][:3, z - off[0], :, :]), (2, 1, 0))
    seg_slc = np.transpose(np.squeeze(dset['predseg'][z - off[0], :, :]), (1, 0))

    f, ax = plt.subplots(2, 3, sharey=True, figsize=(18., 12.))

    a = ax[0, 0]
    a.imshow(raw_slc, cmap=plt.cm.get_cmap('gray'))
    if 'components' in dset:
        a.imshow(gt_slc, cmap=cmap, alpha=0.15)
    a.axis('off')

    a = ax[0, 1]
    if 'components' in dset:
        ax[0, 1].imshow(gt_slc, cmap=cmap)
    a.axis('off')

    a = ax[0, 2]
    if 'components' in dset:
        a.imshow(aff_slc, cmap=plt.cm.get_cmap('gray'))
    a.axis('off')

    a = ax[1, 0]
    a.imshow(raw_slc, cmap=plt.cm.get_cmap('gray'))
    a.imshow(seg_slc, cmap=cmap, alpha=0.15)
    a.axis('off')

    a = ax[1, 1]
    a.imshow(seg_slc, cmap=cmap)
    a.axis('off')

    a = ax[1, 2]
    a.imshow(test_slc, cmap=plt.cm.get_cmap('gray'))
    a.axis('off')
