FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN apt-get update
RUN pip install pip --upgrade

RUN pip install torch==2.1.0a0+29c30b1
RUN pip install torchaudio==2.0.1

ENV PYTHONPATH /workspace/EEND
WORKDIR /workspace/EEND

RUN apt-get install git-lfs
RUN pip install wandb --upgrade
RUN pip install neptune
