docker run \
    --rm \
    -u `id -u $USER` \
    -p 5000:5000 \
    -v /scratch/affinities/fib25:/data \
    --name "h5serv" \
    hdfgroup/h5serv
