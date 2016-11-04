from data_io import OutOfCoreArray
from data_io.out_of_core_arrays import H5PyDArrayHandler


model = "fibsem17"
iteration = "220000"

net_path = "/groups/turaga/home/turagas/research/caffe_v2/pygt_models/{}/net_test.prototxt".format(model)
caffemodel_path = "/groups/turaga/home/turagas/research/caffe_v2/pygt_models/{}/net_iter_{}.caffemodel".format(model, iteration)
net_output_shape = (44,) * 3

dname = "trvol-250-1-h5"
image_opener = H5PyDArrayHandler("FlyEM/fibsem_medulla_7col/{}/im_uint8".format(dname), "main", dname)
image = OutOfCoreArray(image_opener)
