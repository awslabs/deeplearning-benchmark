#!/usr/bin/env bash

set -e

# prepare CUDA 10.1 env
sudo rm -rf /usr/local/cuda
sudo ln -s /usr/local/cuda-10.1 /usr/local/cuda
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}$
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

rm -rf djl
git clone https://github.com/awslabs/djl.git

cd djl/examples

# download the images for inference
curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/3dogs.jpg
curl -O https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/pose/soccer.png

rm -rf /tmp/benchmark
mkdir -p /tmp/benchmark

echo "Running inference Res18..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c ${1} -i 3dogs.jpg -r {'layers':'18','flavor':'v1'}" > /tmp/benchmark/djl_res18.log 2>&1
echo "Running inference Res50..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c ${1} -i 3dogs.jpg -r {'layers':'50','flavor':'v2'}" > /tmp/benchmark/djl_res50.log 2>&1
echo "Running inference Res152..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c ${1} -i 3dogs.jpg -r {'layers':'152','flavor':'v1d'}" > /tmp/benchmark/djl_res152.log 2>&1
echo "Running inference Res50Cifar10..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c ${1} -i 3dogs.jpg -r {'layers':'50','flavor':'v1','dataset':'cifar10'}" > /tmp/benchmark/djl_res50_cifar10.log 2>&1
echo "Running inference Res50Cifar10 Imperative..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c ${1} -i 3dogs.jpg -m -r {'layers':'50','flavor':'v1'}" > /tmp/benchmark/djl_res50_cifar10_imp.log 2>&1
echo "Running inference SSD Resnet50..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c ${1} -i soccer.png -n SSD -r {'size':'512','backbone':'resnet50'}" > /tmp/benchmark/djl_ssd_resnet50.log 2>&1
echo "Running inference SSD Vgg16..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c ${1} -i soccer.png -n SSD -r {'size':'512','backbone':'vgg16'}" > /tmp/benchmark/djl_ssd_vgg16.log 2>&1

export MXNET_ENGINE_TYPE=NaiveEngine
echo "Running inference Res18 with NaiveEngine..."
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark -Dcollect-memory=true --args="-c ${1} -i 3dogs.jpg -r {'layers':'18','flavor':'v1'} -t $(($(nproc) / 2))" > /tmp/benchmark/djl_naive_res18.log 2>&1
echo "Running multithread inference Res18..."
./gradlew run -Dmain=ai.djl.examples.inference.MultithreadedBenchmark -Dcollect-memory=true --args="-c ${1} -i 3dogs.jpg -r {'layers':'18','flavor':'v1'} -t $(($(nproc) / 2))" > /tmp/benchmark/djl_multithread_res18.log 2>&1
echo "Running multithread inference Res18 enableed threadsafe..."
./gradlew run -Dmain=ai.djl.examples.inference.MultithreadedBenchmark -Dcollect-memory=true -DMXNET_THREAD_SAFE_INFERENCE=true --args="-c ${1} -i 3dogs.jpg -r {'layers':'18','flavor':'v1'} -t $(($(nproc) / 2))" > /tmp/benchmark/djl_multithread_res18_threadsafe.log 2>&1
echo "Running multithread inference Res18 enableed threadsafe Imperative..."
./gradlew run -Dmain=ai.djl.examples.inference.MultithreadedBenchmark -Dcollect-memory=true --args="-c ${1} -i 3dogs.jpg -m -r {'layers':'50','flavor':'v1'} -t $(($(nproc) / 2))" > /tmp/benchmark/djl_multithread_res18_imp.log 2>&1
unset MXNET_ENGINE_TYPE

if nvidia-smi -L
then
    hw_type=GPU
else
    hw_type=CPU
fi

declare -a models=("djl_res18" "djl_res50" "djl_res152" "djl_res50_cifar10" "djl_res50_cifar10_imp" "djl_ssd_resnet50" "djl_ssd_vgg16")
declare -a multithreading_models=("djl_naive_res18" "djl_multithread_res18" "djl_multithread_res18_threadsafe" "djl_multithread_res18_imp")

{
    printf "DJL Inference Result\n"
    printf "CPU/GPU: %s\n" "$hw_type"
    for model in "${models[@]}"; do
        log="/tmp/benchmark/${model}.log"
        printf "======================================\n"
        printf "%s inference P50: %s\n" "$model" "$(grep "inference P50:" "${log}" | awk '{ print $6 }')"
        printf "%s inference P90: %s\n" "$model" "$(grep "inference P50:" "${log}" | awk '{ print $9 }')"
        printf "%s preprocess P50: %s\n" "$model" "$(grep "preprocess P50:" "${log}" | awk '{ print $6 }')"
        printf "%s preprocess P90: %s\n" "$model" "$(grep "preprocess P50:" "${log}" | awk '{ print $9 }')"
        printf "%s postprocess P50: %s\n" "$model" "$(grep "postprocess P50:" "${log}" | awk '{ print $6 }')"
        printf "%s postprocess P90: %s\n" "$model" "$(grep "postprocess P50:" "${log}" | awk '{ print $9}')"
    done

    for multithreading_model in "${multithreading_models[@]}"; do
        log="/tmp/benchmark/${multithreading_model}.log"
        printf "======================================\n"
        printf "%s inference P50: %s\n" "$multithreading_model" "$(grep "inference P50:" "${log}" | awk '{ print $6 }')"
        printf "%s inference P90: %s\n" "$multithreading_model" "$(grep "inference P50:" "${log}" | awk '{ print $9 }')"
        printf "%s preprocess P50: %s\n" "$multithreading_model" "$(grep "preprocess P50:" "${log}" | awk '{ print $6 }')"
        printf "%s preprocess P90: %s\n" "$multithreading_model" "$(grep "preprocess P50:" "${log}" | awk '{ print $9 }')"
        printf "%s postprocess P50: %s\n" "$multithreading_model" "$(grep "preprocess P50:" "${log}" | awk '{ print $6 }')"
        printf "%s postprocess P90: %s\n" "$multithreading_model" "$(grep "preprocess P50:" "${log}" | awk '{ print $9 }')"
        printf "%s heap P90: %s\n" "$multithreading_model" "$(grep "heap P90:" "${log}" | awk '{ print $NF }')"
        printf "%s nonHeap P90: %s\n" "$multithreading_model" "$(grep "nonHeap P90:" "${log}" | awk '{ print $NF }')"
        printf "%s cpu P90: %s\n" "$multithreading_model" "$(grep "cpu P90:" "${log}" | awk '{ print $NF }')"
        printf "%s rss P90: %s\n" "$multithreading_model" "$(grep "rss P90:" "${log}" | awk '{ print $NF }')"
  done

} >> /tmp/benchmark/report.txt

cat /tmp/benchmark/report.txt
