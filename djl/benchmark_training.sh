#!/usr/bin/env bash

set -e

# prepare CUDA 10.1 env
sudo rm -rf /usr/local/cuda
sudo ln -s /usr/local/cuda-10.1 /usr/local/cuda
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}$
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

sudo pip3 install mxnet-cu101mkl

rm -rf /tmp/benchmark
mkdir -p /tmp/benchmark

python3 trainCiFar10.py --epochs $1 --model resnet50_v1 --mode hybrid --lr 0.001 --wd 0.001 --gpus 0 --use_thumbnail
mv image-classification.log /tmp/benchmark/python_res50_cifar10_sym.log
python3 trainCiFar10.py --epochs $1 --model resnet50_v1 --mode imperative --lr 0.001 --wd 0.001 --lr-steps 20,60,90,120,180 --lr-factor 0.31623 --gpus 0 --use_thumbnail
mv image-classification.log /tmp/benchmark/python_res50_cifar10_imp.log

sudo pip3 uninstall -y mxnet-cu101mkl

git clone https://github.com/awslabs/djl.git

cd djl/examples

echo 'Running training Res50...'
./gradlew run -Dmain=ai.djl.examples.training.transferlearning.TrainResnetWithCifar10 --args="-g 1 -b 32 -e ${1} -o logs/" > /tmp/benchmark/djl_res50_cifar10_imp.log 2>&1
echo 'Running training Res50 Symbolic mode...'
./gradlew run -Dmain=ai.djl.examples.training.transferlearning.TrainResnetWithCifar10 --args="-g 1 -b 32 -e ${1} -o logs/ -s" > /tmp/benchmark/djl_res50_cifar10_sym.log 2>&1
echo 'Running training Res50 Symbolic mode with Pretrained Model ...'
./gradlew run -Dmain=ai.djl.examples.training.transferlearning.TrainResnetWithCifar10 --args="-g 1 -b 32 -e ${1} -o logs/ -s -p" > /tmp/benchmark/djl_res50_cifar10_sym_pretrain.log 2>&1

declare -a python_models=("python_res50_cifar10_sym" "python_res50_cifar10_imp")
declare -a djl_models=("djl_res50_cifar10_imp" "djl_res50_cifar10_sym" "djl_res50_cifar10_sym_pretrain")

speed_file="/tmp/benchmark/speed.txt"

{
    printf "Python Training Result\n"
    for model in "${python_models[@]}"; do
        printf "======================================\n"
        # delete speed.txt if it exists
        if [ -f $speed_file ]; then
            rm $speed_file
        fi
        grep 'Speed' /tmp/benchmark/"${model}".log | awk '{ print ($8) }' | sort -g > $speed_file
        line50=$(($(wc -l < $speed_file) / 2))
        printf "%s speed P50: %s\n" "$model" "$(sed -n "${line50}p" $speed_file)"
        printf "%s accuracy: %s\n" "$model" "$(grep -oP "training: accuracy=\K(\d+.\d+)" /tmp/benchmark/"${model}".log | tail -1)"
    done
    printf "\nDJL Training Result\n"
    for djl_model in "${djl_models[@]}"; do
        printf "======================================\n"
        printf "%s speed P50: %s\n" "$djl_model" "$(grep "train P50:" /tmp/benchmark/"${djl_model}".log | awk '{ print 32 / $6 * 1000 }')"
        printf "%s accuracy: %s\n" "$djl_model" "$(grep -oP "train accuracy: \K(\d+.\d+)" /tmp/benchmark/"${djl_model}".log | tail -1)"
    done

} >> /tmp/benchmark/report.txt

cat /tmp/benchmark/report.txt
