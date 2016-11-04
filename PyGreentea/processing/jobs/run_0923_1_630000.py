import dvision


model = "run_0923_1"
iteration = "630000"

net_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_test_big.prototxt".format(model)
caffemodel_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_iter_{}.caffemodel".format(model, iteration)
net_output_shape = (116,) * 3

dname = "fib25-e402c09"
image = dvision.DVIDDataInstance(
    "slowpoke3",
    32773,
    "e402c09ddd0f45e980d9be6e9fcb9bd0",
    "grayscale"
)
