# ---------------------------------------------------------
#  Pytorch based container fro development
# ---------------------------------------------------------

FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

# ---------------------------------------------------------
# ---- Supress warnings about the missing frontend
ARG DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------
# ---- Essentials: developer tools, build tools, OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake \
    libopenblas-dev libmysqlclient-dev

# ---------------------------------------------------------
# ---- Python 3.8
RUN apt install python3.8 -y
RUN ln -s /usr/bin/pip3 /usr/bin/pip && \
    ln -s /usr/bin/python3.8 /usr/bin/python
RUN apt-get install python3-pip -y

# ---------------------------------------------------------
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev

# ---------------------------------------------------------
RUN apt-get install -y --no-install-recommends \
    libjpeg8-dev libtiff5-dev libpng-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
    liblapacke-dev checkinstall

RUN apt-get update && apt-get install -y libgdcm-tools

# ---------------------------------------------------------
# ---- JAVA
RUN apt-get install -y --no-install-recommends default-jdk

# ---------------------------------------------------------
RUN pip3 install torch
RUN pip3 install torchvision

# ---------------------------------------------------------
#>>>> FIXME Set the correct path to requirements here
ADD ./requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

RUN mkdir mldev
WORKDIR mldev

# ---------------------------------------------------------
RUN export LC_ALL=en_US.utf-8
RUN export LANG=en_US.utf-8

