#!/usr/bin/env bash
name=$1
nv_gpu=${2:-0}
start=${3:-10000}
step=${4:-10000}

container_name=processing-$nv_gpu-$start-$step-$name

echo $container_name

nvidia-docker rm $container_name

sudo mount --make-shared /nrs/turaga

NV_GPU=$nv_gpu nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(dirname $(pwd)):/workspace \
    -v /nrs/turaga:/nrs/turaga:shared \
    -v /groups/turaga/home:/groups/turaga/home:ro \
    -v /groups/turaga/home/grisaitisw/src/PyGreentea/tasks:/opt/PyGreentea/tasks:ro \
    --log-opt max-size=1g \
    --name $container_name \
    turagalab/greentea:cudnn5-caffe_gt-pygt-0.9b \
    /bin/bash -c "GLOG_minloglevel=2 python -u processing/process.py --volume-name $name --start $start --step $step"
