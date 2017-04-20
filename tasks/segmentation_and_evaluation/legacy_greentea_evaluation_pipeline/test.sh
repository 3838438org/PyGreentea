docker run \
    --rm \
    -u `id -u $USER` \
    -v ~/src/greentea-evaluation-pipeline:/workspace \
    -v /groups/turaga/home:/groups/turaga/home:ro \
    -v /nrs/turaga/:/nrs/turaga:shared \
    turagalab/greentea:cudnn5-caffe_gt-pygt-0.9 \
    python -u evaluation/evaluate.py \
        --testing True
