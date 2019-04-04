#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


# Downloading the data and model
export MXNET_HOME=${HOME}/incubator-mxnet
export CPP_INFERENCE_EXAMPLE=image_classification
export CPP_INFERENCE_EXAMPLE_FOLDER=${MXNET_HOME}/cpp-package/example/inference
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MXNET_HOME}/lib

mkdir -p ${CPP_INFERENCE_EXAMPLE_FOLDER}/model
cd ${CPP_INFERENCE_EXAMPLE_FOLDER}/model
wget -nc -O ${CPP_INFERENCE_EXAMPLE_FOLDER}/model/resnet50_ssd_model-symbol.json https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-symbol.json
wget -nc -O ${CPP_INFERENCE_EXAMPLE_FOLDER}/model/resnet50_ssd_model-0000.params https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-0000.params
wget -nc -O ${CPP_INFERENCE_EXAMPLE_FOLDER}/model/synset.txt https://s3.amazonaws.com/model-server/models/resnet50_ssd/synset.txt
wget -nc -O ${CPP_INFERENCE_EXAMPLE_FOLDER}/model/dog.jpg https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/dog.jpg?raw=true
wget -nc -O ${CPP_INFERENCE_EXAMPLE_FOLDER}/model/mean_224.nd https://github.com/dmlc/web-data/raw/master/mxnet/example/feature_extract/mean_224.nd
cd ${CPP_INFERENCE_EXAMPLE_FOLDER}
cp ${MXNET_HOME}/build/cpp-package/example/${CPP_INFERENCE_EXAMPLE} .


# Running the example with dog image.
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MXNET_HOME}/lib ${CPP_INFERENCE_EXAMPLE_FOLDER}/${CPP_INFERENCE_EXAMPLE} --symbol "${CPP_INFERENCE_EXAMPLE_FOLDER}/model/resnet50_ssd_model-symbol.json" --params "${CPP_INFERENCE_EXAMPLE_FOLDER}/model/resnet50_ssd_model-0000.params" --synset "${CPP_INFERENCE_EXAMPLE_FOLDER}/model/synset.txt" --mean "${CPP_INFERENCE_EXAMPLE_FOLDER}/model/mean_224.nd" --image "${CPP_INFERENCE_EXAMPLE_FOLDER}/model/dog.jpg" --warmup 10 --predict 1000 --gpu 
