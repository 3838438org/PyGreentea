group=$1
n_workers=${2:-1}  # one worker, if not specified
name=evaluation-$group

docker rm $name

sudo mount --make-shared /nrs/turaga

docker run -d \
    -u `id -u $USER` \
    -v /groups/turaga/home/grisaitisw/src/greentea-evaluation-pipeline/:/workspace \
    -v /nrs/turaga:/nrs/turaga:shared \
    -v /groups/turaga/home:/groups/turaga/home:ro \
    --name $name \
    turagalab/greentea:cudnn5-caffe_gt-pygt-0.9b \
    python -u evaluation/evaluate.py \
        --group-name $group \
        --start 10000 \
        --stop 400000 \
        --n-local-workers $n_workers
