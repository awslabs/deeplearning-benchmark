#!/bin/bash

set -ex


data_path=/tmp/resnet-18/

image_path=$data_path/images/

if [ ! -d "$data_path" ]; then
  mkdir -p "$data_path"
fi

if [ ! -d "$image_path" ]; then
  mkdir -p "$image_path"
fi

if [ ! -f "$data_path" ]; then
  wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/resnet-18-symbol.json -P $data_path
  wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/resnet-18-0000.params -P $data_path
  wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/synset.txt -P $data_path
  wget https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/resnet152/kitten.jpg -P $image_path
fi

max=16
cd /tmp/resnet-18/images/
for i in `seq 2 $max`
do
    cp kitten.jpg $1$i.jpg
done
