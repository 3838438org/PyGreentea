container_name=processing-distributed-run_0923_1-630000-fib25-4

echo $container_name

nvidia-docker rm $container_name

sudo mount --make-shared /nobackup/turaga

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



#    /bin/bash -c "cat /groups/turaga/home/grisaitisw/.ipython/profile_greentea/security/ipcontroller-client.json"
#    /bin/bash -c "HOME=/groups/turaga/home/grisaitisw USER=grisaitisw python -u process_stuff.py"
#    /bin/bash -c "HOME=/groups/turaga/home/grisaitisw USER=grisaitisw ls -la ~"

#    -v /nobackup/turaga:/nobackup/turaga:shared \
