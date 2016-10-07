
class Processor(object):
    def __init__(self, net_path, caffemodel_path, executor):
        '''
        :param net_path: string path to prototxt file
        :param caffemodel_path: string path to caffemodel file
        :param executor: an Executor-like object that can import PyGreentea
        :return:
        '''
        self.net_path = net_path
        self.caffemodel_path = caffemodel_path
        self.executor = executor
        return

    @staticmethod
    def import_and_process(net_path, caffemodel_path, array_like, target):
        import numpy as np
        def generate_pred():
            import os
            import h5py
            import PyGreentea as pygt
            pygt.caffe.set_mode_gpu()
            pygt.caffe.set_device(0)
            assert os.path.exists(net_path), net_path
            assert os.path.exists(caffemodel_path), caffemodel_path
            net = pygt.caffe.Net(net_path, caffemodel_path, pygt.caffe.TEST)
            if array_like is None:
                input_data = h5py.File("/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/tstvol-520-1-h5/im_uint8.h5")["main"]
                dataset = dict(data=input_data, name="test", image_scaling_factor=0.5 ** 8)
                preds = pygt.process(net, [dataset])
            else:
                input_data = array_like
                dataset = dict(data=input_data, name="test", image_scaling_factor=0.5 ** 8)
                preds = pygt.process(net, [dataset])
            return preds[0]
        
        # pred = generate_pred()
        pred = np.random.RandomState(seed=0).uniform(size=np.prod((3,) + array_like.shape)).reshape((3,) + array_like.shape).astype(np.float32)
        if target:
            target.save(pred, array_like.offset)
        pred_summary = dict(pred_mean=pred.mean(),
                            pred_shape=pred.shape,
                            source_offset=array_like.offset,
                            source_shape=array_like.shape)
        return pred_summary,

    def process(self, source, target=None):
        async_result = self.executor.apply_async(self.import_and_process, self.net_path, self.caffemodel_path, source, target)
        return async_result


# def new_set_item(item):
#     shape = tuple(s.stop - s.start for s in item)
#     if reduce(mul, shape) * dataset.dtype(0).itemsize > 65421312:
#         chunk_shape = (3, 176, 176, 176)
#         chunk_offsets = chunkify_shape(shape, chunk_shape)
#         chunks_slices = [tuple(slice(x, x+l) for x, l in zip(offset, chunk_shape)) for offset in chunk_offsets]
#         for slices in chunk_slices:
# if out_array.nbytes > (3 * 176 ** 3):
#     warnings.warn("This is probably too big...{} with {} bytes".format(out_array.shape, out_array.nbytes))
# target_slices = tuple([slice(o, o + l) for o, l in zip((0,) + array_like.offset, out_array.shape)])
# print("writing to", target_slices)
# with target.open() as a:
#     a[target_slices] = out_array