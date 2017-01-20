nv_gpu=${1:-7}
container_name=greentea-engine-$nv_gpu

echo $container_name

nvidia-docker rm $container_name

sudo mount --make-shared /nrs/turaga

NV_GPU=$nv_gpu \
    nvidia-docker run -d \
    -u `id -u $USER` \
    -v /nrs/turaga:/nrs/turaga:shared \
    -v /scratch/affinities:/scratch/affinities \
    -v /groups/turaga/home:/groups/turaga/home \
    -v ~/src/PyGreentea:/opt/PyGreentea \
    --name $container_name \
    turagalab/greentea:cudnn5-caffe_gt-pygt-0.9d \
    /bin/bash -c "GLOG_minloglevel=2 HOME=/groups/turaga/home/grisaitisw ipengine --profile=greentea"

