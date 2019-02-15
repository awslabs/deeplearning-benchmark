#!/bin/bash

set -ex

CURR_DIR=$(cd $(dirname $0)/../; pwd)

if [[ $1 = e2e ]]
then
    model_url="https://s3.us-east-2.amazonaws.com/mxnet-public/end_to_end_models"
    model_name="resnet18_v1_end_to_end"
else
    model_url="https://s3.us-east-2.amazonaws.com/mxnet-public/end_to_end_models"
    model_name="resnet18_v1"
fi
model_path=models/
if [ ! -d "$model_path" ]; then
    mkdir -p "$model_path"
fi

if [ ! -f "$model_path" ]; then
    wget "$model_url/$model_name-symbol.json" -P $model_path
    wget "$model_url/$model_name-0000.params" -P $model_path
fi
