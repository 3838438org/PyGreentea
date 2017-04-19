import os

import malis

MODELS_ROOT = "/groups/turaga/home/grisaitisw/experiments"
OUTPUTS_ROOT = "/nrs/turaga/grisaitisw/affinities"

class Net(object):
    def __init__(self):
        self.fmaps_start = None
        self.fmaps_multiplier = None
        self.input_shape = None
        self.output_shape = None


class Solver(object):
    def __init__(self):
        self.learning_rate = None


class AffinityModel(object):
    def __init__(self,
                 name,
                 iteration,
                 neighborhood=malis.mknhood3d(),
                 padding_requirement=(0, 0, 0),
                 training_root=MODELS_ROOT,
                 outputs_root=OUTPUTS_ROOT,
                 ):
        self.name = name
        self.iteration = iteration
        self.neighborhood = neighborhood
        self.padding_requirement = padding_requirement
        self.training_directory = training_root
        self.outputs_directory = outputs_root
        self.description = os.path.join(self.name, str(self.iteration))
        self.net = Net()
        self.solver = Solver()

    @property
    def caffemodel_path(self):
        filename = "net_iter_{}.caffemodel".format(self.iteration)
        return os.path.join(self.training_directory, self.name, filename)

    @property
    def outputs_path(self):
        return os.path.join(self.outputs_directory, self.name, str(self.iteration))

    def get_affinity_for(self, data_name):
        from data_io import OutOfCoreArray
        from data_io.out_of_core_arrays import H5PyArrayHandler
        path = os.path.join(self.outputs_path, data_name + ".h5")
        return OutOfCoreArray(H5PyArrayHandler(path, "main"))


if __name__ == "__main__":
    m = AffinityModel("run_1208_6", 30000)
    print(m.outputs_directory)
