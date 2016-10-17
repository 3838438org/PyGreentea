# Import Caffe

import os
import sys
import warnings

current_file = os.path.abspath(os.path.dirname(__file__))
pygreentea_parent_directory = os.path.dirname(os.path.dirname(current_file))
potential_caffe_path = os.path.join(pygreentea_parent_directory, 'caffe_gt', 'python')
print(potential_caffe_path)
sys.path.append(potential_caffe_path)

try:
    import caffe
except ImportError:
    caffe = None
    warnings.warn("Can't import pycaffe :( Please either add the "
                  "caffe/python/ directory to your PYTHONPATH environment "
                  "variable, or move this repo to be in the same "
                  "directory as where caffe is")
