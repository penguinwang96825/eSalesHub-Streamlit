FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04
# FROM anibali/pytorch:1.10.2-cuda11.3-ubuntu20.04

# Set up time zone.
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Set up miniconda environment
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install necessary library
RUN apt-get update \
	&& apt-get install -y htop python3-dev wget git openssh-server

WORKDIR /home/yangwang/esaleshub-streamlit-app

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
	&& mkdir -p /root/.conda \
	&& sh Miniconda3-latest-Linux-x86_64.sh -b \
	&& rm -f Miniconda3-latest-Linux-x86_64.sh

RUN /bin/bash -c "conda update conda -y \
	&& conda create --name esaleshub python=3.8"

RUN conda init bash
RUN /bin/bash -c "source activate esaleshub \
	&& conda install ipykernel ipywidgets -y \
	&& python -m ipykernel install --user --name esaleshub"

RUN mkdir -p /home/yangwang/esaleshub-streamlit-app

COPY . /home/yangwang/esaleshub-streamlit-app

RUN /bin/bash -c "source activate esaleshub \
	&& pip install -r requirements.txt"
