#!/bin/bash
SCALA_VERSION_PROFILE=2.11
MXNET_VERSION="[1.5.0-SNAPSHOT,)"

# build the project with maven wrapper to avoid avoid dead lock when using the apt install maven but sudo apt update still running on background
./mvnw clean install dependency:copy-dependencies package -Dmxnet.hw_type=$1 -Dmxnet.scalaprofile=$SCALA_VERSION_PROFILE -Dmxnet.version=$MXNET_VERSION
