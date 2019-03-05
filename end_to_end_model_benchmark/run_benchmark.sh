#!/bin/bash
set -e

if [[ $3 = gpu ]]
then
    hw_type=gpu
    use_gpu="--use-gpu"
else
    hw_type=cpu
    use_gpu=""
fi

if [[ $2 = e2e ]]
then
    model_path="../models/resnet18_v1_end_to_end"
    end_to_end="--end-to-end"
else
    model_path="../models/resnet18_v1"
    end_to_end=""
fi
# copy the maven wrapper related script into scala or java folder and
# enter either java or scala folder
if [[ $1 = scala ]]
then
    cp -r ./.mvn scala-bm
    cp -r ./mvnw* scala-bm
    cd scala-bm
else
    cp -r ./.mvn java-bm
    cp -r ./mvnw* java-bm
    cd java-bm
fi
# build the project
bash bin/build.sh $hw_type
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.2/lib64/
CURR_DIR=$(pwd)
CLASSPATH=$CLASSPATH:$CURR_DIR/target/*:$CLASSPATH:$CURR_DIR/target/dependency/*:$CLASSPATH:$CURR_DIR/target/classes/lib/*
# run single inference
output_single=$(java -Xmx8G  \
-Dlog4j.configuration=file:/home/ubuntu/benchmarkai/end_to_end_model_benchmark/log4j.properties \
-cp $CLASSPATH mxnet.EndToEndModelWoPreprocessing \
--model-path-prefix $model_path \
--num-runs $4 \
--batchsize 1 \
--warm-up 5 \
$end_to_end \
$use_gpu)

sum=0.0
# the defualt value is 25 so tha we have enough CPU and GPU memory
num_runs=25
num_iter=$(($4 / $num_runs))
if (( $4 < $num_runs )); then num_runs=$4; fi
if (( $num_iter == 0 )); then num_iter=1; fi
for n in `seq 1 $num_iter`
do
    output_batch=$(java -Xmx8G  \
    -Dlog4j.configuration=file:/home/ubuntu/benchmarkai/end_to_end_model_benchmark/log4j.properties \
    -cp $CLASSPATH mxnet.EndToEndModelWoPreprocessing \
    --model-path-prefix $model_path \
    --num-runs $num_runs \
    --batchsize 25 \
    --warm-up 1 \
    $end_to_end \
    $use_gpu)
    value=$(echo $output_batch | grep -oP '(E2E|Non E2E) (single|batch)_inference_average \K(\d+.\d+)(?=ms)')
    # use awk to support float calculation
    sum=$(awk "BEGIN {print $sum+$value}")
done

metrix=$(echo $output_batch | grep -oE '(single|batch)_inference_average')
echo "$output_single $metrix $(awk "BEGIN {print $sum / $num_iter}")ms"
