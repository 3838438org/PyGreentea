container_name=processing-distributed

echo $container_name

nvidia-docker rm $container_name

sudo mount --make-shared /nrs/turaga

NV_GPU=1 \
    nvidia-docker run -it -d \
    -u `id -u $USER` \
    -v $(pwd):/workspace:ro \
    -v /scratch:/scratch \
    -v /groups/turaga/home:/groups/turaga/home \
    -v ~/src/PyGreentea:/opt/PyGreentea:ro \
    --name $container_name \
    turagalab/greentea:cudnn5-caffe_gt-pygt-0.9d \
    /bin/bash -c "HOME=/groups/turaga/home/grisaitisw python -u process_stuff.py"
