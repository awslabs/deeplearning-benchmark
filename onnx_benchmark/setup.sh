#!/bin/bash

install_mxnet()
{
    if [ $1 == "cpu" ]; then
        sudo pip uninstall --yes mxnet-cu90
        echo "Installing mxnet cpu ......"
        sudo pip install mxnet --pre
    fi
    if [ $1 == "gpu" ]; then
        sudo pip uninstall --yes mxnet
        echo "Installing mxnet gpu ......"
        sudo pip install mxnet-cu90 --pre
    fi
}

install_onnx()
{
    echo "Installing protobuf ......."
    sudo apt-get -y install protobuf-compiler libprotoc-dev
    echo "Installing ONNX version 1.1.1 ........"
    sudo pip  install protobuf==3.5.2 onnx==1.1.1
}


get_models()
{
    if [ ! -d "models" ]; then
        mkdir models
    fi
    if [ ! -f "models/bvlc_alexnet.tar.gz" ]; then
        curl -o models/bvlc_alexnet.tar.gz   https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_alexnet.tar.gz
    fi
    if [ ! -f "models/bvlc_googlenet.tar.gz" ]; then
        curl -o models/bvlc_googlenet.tar.gz https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_googlenet.tar.gz
    fi
    if [ ! -f "models/bvlc_reference_caffenet.tar.gz" ]; then
        curl -o models/bvlc_reference_caffenet.tar.gz https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_reference_caffenet.tar.gz
    fi
    if [ ! -f "models/bvlc_reference_rcnn_ilsvrc13.tar.gz" ]; then
        curl -o models/bvlc_reference_rcnn_ilsvrc13.tar.gz https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_reference_rcnn_ilsvrc13.tar.gz
    fi
    if [ ! -f "models/densenet121.tar.gz" ]; then
        curl -o models/densenet121.tar.gz https://s3.amazonaws.com/download.onnx/models/opset_3/densenet121.tar.gz
    fi
    if [ ! -f "models/resnet50.tar.gz" ]; then
        curl -o models/resnet50.tar.gz https://s3.amazonaws.com/download.onnx/models/opset_3/resnet50.tar.gz
    fi
    if [ ! -f "models/shufflenet.tar.gz" ]; then
        curl -o models/shufflenet.tar.gz https://s3.amazonaws.com/download.onnx/models/opset_3/shufflenet.tar.gz
    fi
    if [ ! -f "models/squeezenet.tar.gz" ]; then
        curl -o models/squeezenet.tar.gz https://s3.amazonaws.com/download.onnx/models/opset_3/squeezenet.tar.gz
    fi
    if [ ! -f "models/vgg19.tar.gz" ]; then
        curl -o models/vgg19.tar.gz https://s3.amazonaws.com/download.onnx/models/opset_3/vgg19.tar.gz
    fi
    
    for f in models/*.tar.gz; do tar xzf  "$f" -C models/; done
}

main() {
    get_models
    install_onnx
    if [ $1 == "cpu" ]; then 
        install_mxnet "cpu"
    fi
    if [ $1 == "gpu" ]; then
        install_mxnet "gpu"
    fi
}

main $1
