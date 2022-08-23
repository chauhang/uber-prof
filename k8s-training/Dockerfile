ARG BASE_IMAGE=nvcr.io/nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

FROM ${BASE_IMAGE}

RUN apt-get install -y python3-venv

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py

ENV VIRTUAL_ENV=/home/ubuntu/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install -U pip

# RUN pip install torch==1.12.0 torchtext torchdata torchvision torchaudio
RUN pip install torch torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/cu113

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install protobuf==3.19.0

ENTRYPOINT ["/bin/bash", "-l", "-c"]
