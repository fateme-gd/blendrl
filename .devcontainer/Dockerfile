# Select the base image
#FROM nvcr.io/nvidia/pytorch:21.07-py3
FROM nvcr.io/nvidia/pytorch:21.10-py3

#FROM anibali/pytorch:1.8.0-cuda10.2-ubuntu20.04

ENV TZ=Europe/Berlin
ARG DEBIAN_FRONTEND=noninteractive

# Select the working directory
WORKDIR  /Workspace

# Install system libraries required by OpenCV.
#RUN apt-get update \
#    && apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
#    && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get -y install qt5-default

RUN pip install --upgrade pip
