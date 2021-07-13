FROM nvidia/cuda:11.4.0-devel-ubuntu20.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        apt-transport-https ca-certificates gnupg software-properties-common wget zlib1g-dev\
        build-essential g++-9 make m4 python3-distutils python3-dev

# Install cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && apt-get update && apt-get -y install cmake git

COPY . /app
WORKDIR /app

RUN cmake -B ./build -DCMAKE_BUILD_TYPE=Release && make -C build -j && make -C build install
