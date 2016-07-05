import inspect
import os
import sys

import config
import manual_setup
import manual_setup.system_calls


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Determine where PyGreentea is
directory_of_this_module = os.path.normpath(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])))
sys.path.append(directory_of_this_module)

# Determine where PyGreentea gets called from
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)


# If this is called directly, set up everything
if __name__ == "__main__":
    if (directory_of_this_module != current_working_directory):
        os.chdir(directory_of_this_module)

    if (os.geteuid() != 0):
        print(bcolors.WARNING + "PyGreentea setup should probably be executed with root privileges!" + bcolors.ENDC)

    if config.install_packages:
        print(bcolors.HEADER + ("==== PYGT: Installing OS packages ====").ljust(80, "=") + bcolors.ENDC)
        manual_setup.system_calls.install_dependencies()

    print(bcolors.HEADER + ("==== PYGT: Updating Caffe/Greentea repository ====").ljust(80, "=") + bcolors.ENDC)
    manual_setup.system_calls.clone_caffe(config.caffe_path, config.clone_caffe, config.update_caffe)

    print(bcolors.HEADER + ("==== PYGT: Updating Malis repository ====").ljust(80, "=") + bcolors.ENDC)
    manual_setup.system_calls.clone_malis(config.malis_path, config.clone_malis, config.update_malis)

    if config.compile_caffe:
        print(bcolors.HEADER + ("==== PYGT: Compiling Caffe/Greentea ====").ljust(80, "=") + bcolors.ENDC)
        manual_setup.system_calls.compile_caffe(config.caffe_path)

    if config.compile_malis:
        print(bcolors.HEADER + ("==== PYGT: Compiling Malis ====").ljust(80, "=") + bcolors.ENDC)
        manual_setup.system_calls.compile_malis(config.malis_path)

    if (directory_of_this_module != current_working_directory):
        os.chdir(current_working_directory)

    print(bcolors.OKGREEN + ("==== PYGT: Setup finished ====").ljust(80, "=") + bcolors.ENDC)
    sys.exit(0)
