#!/bin/bash

set -ex


data_path=/tmp/resnet50_ssd/

image_path=$data_path/images/

if [ ! -d "$data_path" ]; then
  mkdir -p "$data_path"
fi

if [ ! -d "$image_path" ]; then
  mkdir -p "$image_path"
fi

if [ ! -f "$data_path" ]; then
  wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-symbol.json -P $data_path
  wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-0000.params -P $data_path
  wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/synset.txt -P $data_path
  cd $image_path
  wget https://cloud.githubusercontent.com/assets/3307514/20012567/cbb60336-a27d-11e6-93ff-cbc3f09f5c9e.jpg -O dog.jpg
fi
