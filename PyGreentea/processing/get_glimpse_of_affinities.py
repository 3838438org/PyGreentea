import zarr

chunk = (51, 11, 15)
origin = (0,) + tuple(x * 148 for x in chunk)
print(origin)
shape= (3, 148, 148, 148)
slices = tuple(slice(x, x + l) for x, l in zip(origin, shape))
print(slices)

array = zarr.open_array('/scratch/affinities/fibsem17/220000/fib25-e402c09-2016.10.27-22.13.27', 'r')
print(array[slices].mean())
