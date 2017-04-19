#!/usr/bin/env bash
nv_gpu=${1:-0}

container_name=processing-$nv_gpu-example

echo $container_name

nvidia-docker rm $container_name

NV_GPU=$nv_gpu nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups/turaga/home:/groups/turaga/home:ro \
    -v $(dirname $(dirname $(pwd))):/opt/PyGreentea/tasks:ro \
    -v /tmp:/tmp:rw \
    --log-opt max-size=1g \
    --name $container_name \
    turagalab/greentea:cudnn5-caffe_gt-pygt-0.9b \
    python -u process_a_volume.py
