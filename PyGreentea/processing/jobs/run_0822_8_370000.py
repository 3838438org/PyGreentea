import dvision


model = "run_0822_8"
iteration = "370000"

net_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_test_big.prototxt".format(model)
caffemodel_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_iter_{}.caffemodel".format(model, iteration)
net_output_shape = (148,) * 3

dname = "cx-1598d86"
image = dvision.DVIDDataInstance(
    "slowpoke2",
    32768,
    "1598d862fcce47d9b504a67a38b3bfbf",
    "grayscale"
)
