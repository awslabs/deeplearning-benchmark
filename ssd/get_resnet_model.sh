#!/bin/bash

set -ex


data_path=/tmp/

image_path=$data_path/images/

if [ ! -d "$data_path" ]; then
  mkdir -p "$data_path"
fi

if [ ! -d "$image_path" ]; then
  mkdir -p "$image_path"
fi

if [ ! -f "$data_path" ]; then
  wget https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet50v1.mar -P $data_path
  unzip /tmp/onnx-resnet50v1.mar -d /tmp/onnx-resnet50v1
  cd $image_path
  wget https://s3.amazonaws.com/model-server/inputs/kitten.jpg -O kitten.jpg
fi
