#!/bin/bash

echo "Downloading the build essentials for mxnet build."
sudo apt-get update
sudo apt-get install -y build-essential git
sudo apt-get install -y libopenblas-dev liblapack-dev
sudo apt-get install -y libopencv-dev
sudo apt-get install -y python-dev python-setuptools python-pip libgfortran3

export PERF_HOME=`pwd`


cd ${HOME}
git clone --recursive https://github.com/apache/incubator-mxnet.git incubator-mxnet

export MXNET_HOME=${HOME}/incubator-mxnet
export CPP_INFERENCE_EXAMPLE=${MXNET_HOME}/cpp-package/example/inference

echo "Copying the C++ performance program to ${MXNET_HOME}"
cp ${PERF_HOME}/image_classification.cpp ${CPP_INFERENCE_EXAMPLE}/.
cp ${PERF_HOME}/unit_test_image_classification_cpu.sh ${CPP_INFERENCE_EXAMPLE}/.

echo "Building the mxnet at ${MXNET_HOME}"
cd ${MXNET_HOME}
make USE_CPP_PACKAGE=1 USE_OPENCV=1 USE_CUDA=0 USE_CUDNN=0 USE_LAPACK=0 -j${nproc} 2>&1 | tee buildLog.txt

