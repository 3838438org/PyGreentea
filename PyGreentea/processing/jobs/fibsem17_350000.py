import dvision
from data_io import OffsettedArray


model = "fibsem17"
iteration = "350000"

net_path = "/groups/turaga/home/turagas/research/caffe_v2/pygt_models/{}/net_test_big.prototxt".format(model)
caffemodel_path = "/groups/turaga/home/turagas/research/caffe_v2/pygt_models/{}/net_iter_{}.caffemodel".format(model, iteration)
net_output_shape = (148,) * 3

dname = "fib25-e402c09"
image = dvision.DVIDDataInstance(
    "slowpoke3",
    32773,
    "e402c09ddd0f45e980d9be6e9fcb9bd0",
    "grayscale"
)

mask = dvision.DVIDRegionOfInterest(
    "slowpoke3",
    32773,
    "e402c09ddd0f45e980d9be6e9fcb9bd0",
    "seven_column_roi"
)
