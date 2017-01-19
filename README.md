# PyGreentea

## Installation:
python PyGreentea.py

## Creating Datasets

A dataset is a list of dictionaries, each containing at least the following keys:

* `data`: the raw data (array-like)
* `components`: the ground-truth labeling (array-like)

Optionally, a dictionary may contain the following keys:

* `mask`: a binary mask for which parts of the `components` to use (array-like)

### Providing Arrays

The arrays `data`, `components`, and `mask` can be provided in different ways.

#### HDF5 Datasets

You can pass HDF5 datasets directly. Do this only if you have a small number of such arrays, as the files will keep open.

```
dataset = {
  'data'      : h5py.File('raw.h5', 'r')['main'],
  'components': h5py.File('gt.h5', 'r')['main'],
  'mask'      : h5py.File('mask.h5', 'r')['main'],
}
```

#### `numpy` Arrays

Using `numpy` arrays will keep all your data in memory. Note that copies of the data will be created for each worker in the data loader (defaults to 10).

```
raw  = np.array(...)
gt   = np.array(...)
mask = np.array(...)

dataset = {
  'data'      : raw,
  'components': gt,
  'mask'      : mask,
}
```

FIXME: When using `numpy` arrays, the dimensions of the arrays are supposed to be different. How? Why?

#### `OutOfCoreArray`s

If you have many datasets, `OutOfCoreArray`s provide a way to open the respective files only on demand and thus avoids many open files as well as keeping everything in memory. There are different _openers_ to read the data from different backend. The following example shows how to use the HDF5 backend:

```
from data_io import OutOfCoreArray
from data_io.out_of_core_arrays import H5PyArrayHandler

raw_opener = H5PyArrayHandler("raw.h5", "main", "raw")
raw = OutOfCoreArray(raw_opener)
gt_opener = H5PyArrayHandler("gt.h5", "main", "gt")
gt = OutOfCoreArray(gt_opener)
mask_opener = H5PyArrayHandler("mask.h5", "main", "mask")
mask = OutOfCoreArray(mask_opener)

dataset = {
  'data'      : raw,
  'components': gt,
  'mask'      : mask,
}
```

Other openers are implemented for `zarr` and `h5pyd`.

#### `OffsettedArray`s

An offsetted array allows you to use only a crop of an array-like.

```
from data_io import OffsettedArray

# create raw, gt, and mask using any of the above

# this selects a 100 cube, with a z offset of 10
offset = (10, 0, 0)
size   = (100, 100, 100)

raw_crop = OffsettedArray(raw, offset, size)
gt_crop = OffsettedArray(gt, offset, size)
raw_crop = OffsettedArray(raw, offset, size)

dataset = {
  'data'      : raw,
  'components': gt,
  'mask'      : mask,
}
```

### Dataset Options

Options are set as additional keys in the dataset dictionaries:

* `transform`
* `mask_dilation_steps`
* `mask_threshold`
* `component_erosion_steps`
* `component_erosion_only_xy`
* `simple_augment`
* `probability_zero`
* `probability_low_contrast`
* `contrast_scalar`

FIXME: complete the list, mention defaults, short documentation
