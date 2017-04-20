name=$1

sudo mount --make-shared /nrs/turaga

docker run \
    -u `id -u $USER` \
    -v /groups/turaga/home/grisaitisw/src/greentea-evaluation-pipeline/:/workspace \
    -v /nrs/turaga:/nrs/turaga:shared \
    -v /groups/turaga/home:/groups/turaga/home:ro \
    --name evaluation-test \
    --rm \
    turagalab/greentea:cudnn5-caffe_gt-pygt-0.9b \
    python -u evaluation/evaluate.py \
        --testing True

