docker run \
    -d \
    -u `id -u $USER` \
    -p 5000:5000 \
    -v /groups/turaga/home/grisaitisw/data:/data \
    --name "h5serv" \
    hdfgroup/h5serv
