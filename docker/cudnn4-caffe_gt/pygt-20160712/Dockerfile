FROM turagalab/greentea:cudnn4-201603


WORKDIR /opt/PyGreentea
ENV CLONE_TAG=issue-dvid
ENV COMMIT_SHA1=581fbb7d1d8af35b730f7308ae6568f4f66fa8ce

RUN git clone -b ${CLONE_TAG} https://github.com/TuragaLab/PyGreentea.git . && \
    git checkout ${COMMIT_SHA1} && \
    for req in $(cat requirements.txt) pydot; do pip install $req; done && \
    pip install PyCrypto PyPNG

ENV PYTHONPATH /opt/PyGreentea:$PYTHONPATH

WORKDIR /workspace
