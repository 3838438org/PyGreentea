FROM turagalab/greentea:cudnn5-caffe_gt

MAINTAINER William Grisaitis <grisaitisw@janelia.hhmi.org>

WORKDIR /opt/malis

RUN git clone -b master --depth 1 https://github.com/srinituraga/malis.git . && \
    pip install --upgrade numpy>=1.9 && \
    python setup.py install

WORKDIR /opt/zwatershed

ENV CLONE_TAG=v0.11

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/TuragaLab/zwatershed.git . && \
    ./make.sh

ENV PYTHONPATH /opt/zwatershed:$PYTHONPATH

WORKDIR /opt/matplotlib

RUN git clone -b v1.5.2rc2 --depth 1 https://github.com/matplotlib/matplotlib.git . && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && \
    python setup.py install

WORKDIR /opt/PyGreentea

ENV CLONE_TAG=v0.9

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/TuragaLab/PyGreentea.git . && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && \
    pip install PyCrypto PyPNG

ENV PYTHONPATH /opt/PyGreentea:$PYTHONPATH

WORKDIR /workspace
