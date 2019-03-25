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
mkdir -p model
cd model
wget -nc https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-symbol.json
wget -nc https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-0000.params
wget -nc https://s3.amazonaws.com/model-server/models/resnet50_ssd/synset.txt
wget -nc -O dog.jpg https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/python/predict_image/dog.jpg?raw=true
wget -nc -O mean_224.nd https://github.com/dmlc/web-data/raw/master/mxnet/example/feature_extract/mean_224.nd
cd ..


# Running the example with dog image.
if [ "$(uname)" == "Darwin" ]; then
    DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:../../../lib ./image_classification --symbol "./model/resnet50_ssd_model-symbol.json" --params "./model/resnet50_ssd_model-0000.params" --synset "./model/synset.txt" --mean "./model/mean_224.nd" --image "./model/dog.jpg" --warmup 1 --predict 100 2&> image_classification.log
else
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../../lib ./image_classification --symbol "./model/resnet50_ssd_model-symbol.json" --params "./model/resnet50_ssd_model-0000.params" --synset "./model/synset.txt" --mean "./model/mean_224.nd" --image "./model/dog.jpg" --warmup 1 --predict 100 2&> image_classification.log
fi