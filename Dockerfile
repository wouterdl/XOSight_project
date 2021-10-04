# syntax=docker/dockerfile:1

FROM python:3.8
RUN mkdir /app
WORKDIR /app
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install opencv-python
RUN pip3 install pillow
RUN pip3 install scipy
RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install imageio
RUN pip3 install easydict

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y

ENV runtime=nvidia
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all
