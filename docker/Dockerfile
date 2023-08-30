
FROM ros:melodic-ros-base-bionic

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ros1_distro=melodic
###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=user \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:nvgrasp" | chpasswd

## basic tools

RUN apt-get update && apt-get install -y \
    curl \
    lsb-release \
    sudo \
    software-properties-common \
    git \
    cmake \
    wget \
    make \
    vim \
    build-essential \
    libblkid-dev \
    e2fslibs-dev \
    libboost-all-dev \
    libaudit-dev \
    tzdata \
    python-pip

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
RUN pip2 install setuptools==41.0.0

## Dependency installation
COPY ./package_apt.txt /
COPY ./requirement_pip.txt /
RUN apt update && xargs -a /package_apt.txt apt install -y
RUN pip2 install -r /requirement_pip.txt
RUN rm /package_apt.txt /requirement_pip.txt
RUN apt install -y g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
## setting
RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "${USER} ALL=(ALL) ALL" > /etc/sudoers
RUN echo "root ALL=(ALL) ALL" > /etc/sudoers

# GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all


USER ${USER}
WORKDIR ${HOME}
RUN ls
