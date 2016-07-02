FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04

MAINTAINER William Grisaitis <grisaitisw@janelia.hhmi.org>

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        opencl-headers \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopenblas-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        libviennacl-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe

WORKDIR $CAFFE_ROOT

ENV CLONE_TAG=master

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/naibaf7/caffe.git . && \
    for req in $(cat python/requirements.txt) pydot; do pip install $req; done

ADD Makefile.config.turagalab $CAFFE_ROOT/Makefile.config

RUN make pycaffe -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /opt/malis

RUN git clone -b master --depth 1 https://github.com/srinituraga/malis.git . && \
    pip install --upgrade numpy>=1.9 && \
    python setup.py install

WORKDIR /opt/PyGreentea

RUN git clone -b issue-dvid --depth 1 https://github.com/TuragaLab/PyGreentea.git . && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && \
    pip install PyCrypto

ENV PYTHONPATH /opt/PyGreentea:$PYTHONPATH

WORKDIR /opt/zwatershed

RUN git clone -b master --depth 1 https://github.com/TuragaLab/zwatershed.git . && \
    ./make.sh

ENV PYTHONPATH /opt/zwatershed:$PYTHONPATH

WORKDIR /opt/matplotlib

RUN git clone -b v1.5.2rc2 --depth 1 https://github.com/matplotlib/matplotlib.git . && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && \
    python setup.py install && \
    python tests.py

WORKDIR /workspace
