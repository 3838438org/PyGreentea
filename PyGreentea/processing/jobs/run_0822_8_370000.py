# import data_io
import dvision
# from PyGreentea.processing import Processor, ZarrAffinityHandler as AffinityHandler
# from PyGreentea.processing.array_handlers import AffinitySaver
# from data_io import OutOfCoreArray
# from data_io.out_of_core_arrays import H5PyDArrayHandler, ZarrArrayHandler
# from data_io.util.shape_chunking import chunkify_shape
# from data_io.zero_padded_array import ZeroPaddedArray


model = "run_0822_8"
iteration = "370000"

net_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_test_big.prototxt".format(model)
caffemodel_path = "/groups/turaga/home/grisaitisw/experiments/{}/net_iter_{}.caffemodel".format(model, iteration)
net_output_shape = (148,) * 3

dname = "cx-1598d86"
image = dvision.DVIDDataInstance("slowpoke2", 32770, "1598d862fcce47d9b504a67a38b3bfbf", "grayscale")
