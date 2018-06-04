# Use local version of image built from Dockerfile.gpu in /docker/1.6.0/base
#MAINTAINER Amazon AI
FROM nvidia/cuda:9.0-base-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-dev-9-0 \
        cuda-cudart-dev-9-0 \
        cuda-cufft-dev-9-0 \
        cuda-curand-dev-9-0 \
        cuda-cusolver-dev-9-0 \
        cuda-cusparse-dev-9-0 \
        curl \
        git \
        libcudnn7=7.0.5.15-1+cuda9.0 \
        libcudnn7-dev=7.0.5.15-1+cuda9.0 \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        vim \
        nginx \
        iputils-ping \
	libjemalloc-dev \
        && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/cuda-9.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        numpy \
        scipy \
        sklearn \
        pandas \
        h5py \
	psutil \
	memory_profiler \
	opencv-python \
	boto3 \
	awscli


#RUN pip install numpy tensorflow-serving-api==1.5


ARG framework_installable

WORKDIR /root

# Will install from pypi once packages are released there. For now, copy from local file system.
RUN echo "Creating the new docker image"
RUN framework_installable_local=$(basename $framework_installable) && \
    \
    pip install $framework_installable_local --pre && \
    \
    echo "DONE"


