import h5py
from tasks.processing.caffe_model_processing import process_caffe_model


model_path = "/groups/turaga/home/grisaitisw/experiments/training_template"
iteration = 200000
output_path = "/tmp"
datasets = [dict(
    data=h5py.File("/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col/trvol-250-1-h5/im_uint8.h5", "r")["main"],
    name="trvol-250-1-h5",
    image_scaling_factor=0.5 ** 8,
)]
process_caffe_model(
    model_path,
    iteration,
    output_path,
    datasets
)
