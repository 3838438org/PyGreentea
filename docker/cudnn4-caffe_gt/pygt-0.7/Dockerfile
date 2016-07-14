FROM turagalab/greentea:cudnn4-201603


WORKDIR /opt/PyGreentea
ENV CLONE_TAG=v0.7

RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/TuragaLab/PyGreentea.git . && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && \
    pip install PyCrypto PyPNG

ENV PYTHONPATH /opt/PyGreentea:$PYTHONPATH

WORKDIR /workspace
