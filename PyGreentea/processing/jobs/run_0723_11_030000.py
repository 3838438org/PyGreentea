import dvision


model = "run_0723_11"
iteration = "30000"

net_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_test_big.prototxt".format(model)
caffemodel_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_iter_{}.caffemodel".format(model, iteration)
net_output_shape = (116,) * 3

dname = "mb6-6a5a738"
image = dvision.DVIDDataInstance(
    "slowpoke3",
    32770,
    "6a5a7387b4ce4333aa18d9c8d8647f58",
    "grayscale"
)
