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
        def initialize_net():
            import os
            import PyGreentea as pygt
            pygt.caffe.set_mode_gpu()
            pygt.caffe.set_device(0)
            assert os.path.exists(net_path), net_path
            assert os.path.exists(caffemodel_path), caffemodel_path
            net = pygt.caffe.Net(net_path, caffemodel_path, pygt.caffe.TEST)
            return net

        def process_array_with_net(net, array_like, target):
            import h5py
            import PyGreentea as pygt
            if array_like is None:
                input_data = h5py.File("/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/tstvol-520-2-h5/im_uint8.h5")["main"]
                dataset = dict(data=input_data, name="test", image_scaling_factor=0.5 ** 8)
                preds = pygt.process(net, [dataset], target_arrays=[target])
            else:
                input_data = array_like
                dataset = dict(data=input_data, name="test", image_scaling_factor=0.5 ** 8, image_is_zero_padded=True, region_offset=array_like.offset)
                preds = pygt.process(net, [dataset], target_arrays=[target])
            return preds[0]

        def generate_fake(array_like):
            import numpy as np
            shape = (3,) + array_like.shape
            x = np.ones(shape=shape, dtype=np.float32)
            return x

        def generate_pred(array_like, target):
            net = initialize_net()
            pred = process_array_with_net(net, array_like, target)
            return pred

        pred = generate_pred(array_like, target)
        pred_summary = dict(pred_shape=pred.shape,
                            source_offset=array_like.offset,
                            source_shape=array_like.shape)
        return pred_summary,

    def process(self, source, target):
        async_result = self.executor.apply_async(self.import_and_process, self.net_path, self.caffemodel_path, source, target)
        return async_result

