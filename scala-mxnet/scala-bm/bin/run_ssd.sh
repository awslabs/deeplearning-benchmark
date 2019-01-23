#!/bin/bash

set -ex

echo $OSTYPE

hw_type=cpu
if [[ $1 = gpu ]]
then
    hw_type=gpu
fi

platform=linux-x86_64

if [[ $OSTYPE = [darwin]* ]]
then
    platform=osx-x86_64
fi


SCALA_VERSION_PROFILE=2.11
MXNET_VERSION="[1.5.0-SNAPSHOT,)"

CURR_DIR=$(cd $(dirname $0)/../; pwd)

echo $CURR_DIR
echo $platform-$hw_type

./mvnw clean install dependency:copy-dependencies package -Dmxnet.profile=$platform-$hw_type -Dmxnet.scalaprofile=$SCALA_VERSION_PROFILE -Dmxnet.version=$MXNET_VERSION

CURR_DIR=$(cd $(dirname $0)/../; pwd)

CLASSPATH=$CLASSPATH:$CURR_DIR/target/*:$CLASSPATH:$CURR_DIR/target/dependency/*

# model dir
MODEL_PATH_PREFIX=$2
# input image
INPUT_IMG=$3

BATCHSIZE=$4

RUNS=$5

java -Xmx8G -Dmxnet.traceLeakedObjects=true -cp $CLASSPATH mxnet.SSDObjectDetection \
--modelPathPrefix $MODEL_PATH_PREFIX \
--inputImagePath $INPUT_IMG \
--batchSize $BATCHSIZE \
--times $RUNS
