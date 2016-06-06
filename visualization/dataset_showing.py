from __future__ import print_function

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

max_component_value = 1

random_state = np.random.RandomState(0)

cmap = matplotlib.colors.ListedColormap(np.vstack(((0, 0, 0), (1, 1, 1), random_state.rand(10000, 3))))


def showme(dset, z):
    max_shape = dset['data'].shape
    raw_slices = np.s_[z, :, :]
    raw_slc = np.transpose(np.squeeze(dset['data'][raw_slices]), (1, 0))
    offset_true_label = [0, 0, 0]
    true_label_shape = [0, 0, 0]
    if 'components' in dset:
        true_label_shape = dset['components'].shape
        offset_true_label = [(s[0] - s[1]) / 2 for s in zip(max_shape, true_label_shape)]
        slices = (slice(z - offset_true_label[0], z + 1 - offset_true_label[0]),
                  slice(0, true_label_shape[1]),
                  slice(0, true_label_shape[2]))
        gt_slc = np.transpose(np.squeeze(dset['components'][slices]), (1, 0))
        aff_slc = np.transpose(np.squeeze(dset['label'][(slice(0, 3),) + slices]), (2, 1, 0)).astype(np.float)
    predshape = dset['pred'].shape[-3:]
    offset_predictions = [(s[0] - s[1]) / 2 for s in zip(max_shape, predshape)]
    pred_slices = (slice(z - offset_predictions[0], z + 1 - offset_predictions[0]),
                   slice(0, predshape[1]),
                   slice(0, predshape[2]))
    test_slc = np.transpose(np.squeeze(dset['pred'][(slice(0, 3),) + pred_slices]), (2, 1, 0))
    seg_slc = np.transpose(np.squeeze(dset['predseg'][pred_slices]), (1, 0))

    figure, ax = plt.subplots(3, 3, figsize=(12, 8))
    title_font_size = 10

    image_extent = [0,
                    max_shape[1],
                    0,
                    max_shape[2]]

    true_label_extent = [offset_true_label[1],
                         offset_true_label[1] + true_label_shape[1],
                         offset_true_label[2],
                         offset_true_label[2] + true_label_shape[2]]

    pred_extent = [offset_predictions[1],
                   offset_predictions[1] + predshape[1],
                   offset_predictions[2],
                   offset_predictions[2] + predshape[2]]

    a = ax[0, 0]
    a.set_xlim(0, max_shape[1])
    a.set_ylim(0, max_shape[2])
    a.imshow(raw_slc, cmap=plt.cm.get_cmap('gray'), interpolation='nearest', extent=image_extent)
    if 'components' in dset:
        a.imshow(gt_slc, cmap=cmap, alpha=0.15, interpolation='nearest', extent=true_label_extent)
    a.set_title('data + components:\nimage with true components', fontsize=title_font_size)
    a.axis('off')

    a = ax[0, 1]
    a.set_xlim(0, max_shape[1])
    a.set_ylim(0, max_shape[2])
    if 'components' in dset:
        ax[0, 1].imshow(gt_slc, cmap=cmap, interpolation='nearest', extent=true_label_extent)
    a.set_title('components:\ntrue components', fontsize=title_font_size)
    a.axis('off')

    a = ax[0, 2]
    a.set_xlim(0, max_shape[1])
    a.set_ylim(0, max_shape[2])
    if 'components' in dset:
        a.imshow(aff_slc, cmap=plt.cm.get_cmap('gray'), interpolation='nearest', extent=true_label_extent)
        a.set_title('label:\ntrue affinities', fontsize=title_font_size)
    a.axis('off')

    a = ax[1, 0]
    a.set_xlim(0, max_shape[1])
    a.set_ylim(0, max_shape[2])
    a.imshow(raw_slc, cmap=plt.cm.get_cmap('gray'), interpolation='nearest', extent=image_extent)
    a.imshow(seg_slc, cmap=cmap, alpha=0.15, interpolation='nearest', extent=pred_extent)
    a.set_title('data + predseg:\nimage with predicted components', fontsize=title_font_size)
    a.axis('off')

    a = ax[1, 1]
    a.set_xlim(0, max_shape[1])
    a.set_ylim(0, max_shape[2])
    a.imshow(seg_slc, cmap=cmap, interpolation='nearest', extent=pred_extent)
    a.set_title('predseg:\npredicted components', fontsize=title_font_size)
    a.axis('off')

    a = ax[1, 2]
    a.set_xlim(0, max_shape[1])
    a.set_ylim(0, max_shape[2])
    a.imshow(test_slc, cmap=plt.cm.get_cmap('gray'), interpolation='nearest', extent=pred_extent)
    a.set_title('pred:\npredicted affinities', fontsize=title_font_size)
    a.axis('off')

    a = ax[2, 0]
    a.set_xlim(0, max_shape[1])
    a.set_ylim(0, max_shape[2])
    background = np.ones_like(raw_slc)
    a.imshow(background, cmap=plt.cm.get_cmap('gray'), alpha=0.15, interpolation='nearest', extent=image_extent)
    if 'mask' in dset:
        mask_slices = slices
        mask = np.transpose(np.squeeze(dset['mask'][mask_slices]), (1, 0))
        mask_mean = np.mean(mask)
        a.imshow(mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest', extent=true_label_extent)
        a.set_title('mask (0 means excluded)\nMean value: %07.5f' % mask_mean, fontsize=title_font_size)
    a.axis('off')

    a = ax[2, 1]
    a.set_xlim(0, max_shape[1])
    a.set_ylim(0, max_shape[2])
    if 'components_negative' in dset:
        components_negative = np.transpose(np.squeeze(dset['components_negative'][slices]), (1, 0))
        a.imshow(components_negative, cmap=cmap, interpolation='nearest', extent=true_label_extent)
        a.set_title('components_negative:\ntrue components with zero-moat and 1s', fontsize=title_font_size)
    elif 'error_scale_slice' in dset:
        error_scale_slice = np.transpose(np.squeeze(dset['error_scale_slice'][(slice(0, 3),) + slices]), (2, 1, 0))
        a.imshow(error_scale_slice, cmap=cmap, interpolation='nearest', extent=true_label_extent)
        a.set_title('error_scale_slice', fontsize=title_font_size)
    a.axis('off')

    a = ax[2, 2]
    a.set_xlim(0, max_shape[1])
    a.set_ylim(0, max_shape[2])
    if 'components_positive' in dset:
        components_positive = np.transpose(np.squeeze(dset['components_positive'][slices]), (1, 0))
        a.imshow(components_positive, cmap=cmap, interpolation='nearest', extent=true_label_extent)
        a.set_title('components_positive:\ntrue components with 0s', fontsize=title_font_size)
    a.axis('off')

    return figure
