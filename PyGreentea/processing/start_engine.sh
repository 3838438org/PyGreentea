nv_gpu=${1:-7}
container_name=greentea-engine-$nv_gpu

echo $container_name

nvidia-docker rm $container_name

sudo mount --make-shared /nobackup/turaga

NV_GPU=$nv_gpu \
    nvidia-docker run --rm \
    -u `id -u $USER` \
    -v /nobackup/turaga:/nobackup/turaga:shared \
    -v /scratch/affinities:/scratch/affinities \
    -v /groups/turaga/home:/groups/turaga/home \
    -v /groups/turaga/home/grisaitisw/src/caffe-dvid/PyGreentea:/opt/PyGreentea \
    --name $container_name \
    turagalab/greentea:cudnn5-caffe_gt-pygt-0.9c \
    /bin/bash -c "GLOG_minloglevel=2 HOME=/groups/turaga/home/grisaitisw ipengine --profile=greentea"

