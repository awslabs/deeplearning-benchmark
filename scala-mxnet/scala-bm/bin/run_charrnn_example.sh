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

mvn clean install dependency:copy-dependencies package -Dmxnet.profile=$platform-$hw_type -Dmxnet.scalaprofile=$SCALA_VERSION_PROFILE -Dmxnet.version=$MXNET_VERSION

CURR_DIR=$(cd $(dirname $0)/../; pwd)

CLASSPATH=$CLASSPATH:$CURR_DIR/target/*:$CLASSPATH:$CURR_DIR/target/dependency/*

# model dir
MODEL_PATH_PREFIX=$2
# input image
DATA_PATH=$3

#Starter Sentence
STARTER_SENTENCE=$4

RUNS=$5

java -Xmx8G -Dmxnet.traceLeakedObjects=true -cp $CLASSPATH mxnet.CharRnnExample \
--modelPathPrefix $MODEL_PATH_PREFIX \
--data-path $DATA_PATH \
--starter-sentence "$STARTER_SENTENCE" \
--times $RUNS