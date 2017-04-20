import dvision

model = "2017.03.30.4"
iteration = "10000"

net_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_test_big.prototxt".format(model)
caffemodel_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_iter_{}.caffemodel".format(model, iteration)
net_output_shape = (140,) * 3

dname = "fib25-e402c09"
image = dvision.DVIDDataInstance(
    "slowpoke3",
    32773,
    "e402c09ddd0f45e980d9be6e9fcb9bd0",
    "grayscale"
)

roi = dvision.DVIDRegionOfInterest(
    "slowpoke3",
    32788,
    "341",
    "seven_column"
)

mask = roi

if __name__ == "__main__":
    print(net_path)
