#!/bin/bash

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -m|--models)
    MODELS="$2"
    shift # past argument
    ;;
    -h|--hosts_list)
    HOSTS_LIST="$2"
    shift # past argument
    ;;
    -g|--gpu_per_host)
    GPU_PER_HOST="$2"
    shift # past argument
    ;;
    -r|--remote_dir)
    REMOTE_DIR="$2"
    shift # past argument
    ;;
    -x|--max_gpus)
    MAX_GPUS="$2"
    shift # past argument
    ;;
    *)
          # unknown option
    ;;
esac
shift # past argument or value
done

for model in `echo $MODELS | tr ',' ' '`; do
    model=`echo $model | tr ':' ' ' | tr '[:upper:]' '[:lower:]'`
    arr=( $model )
    model_name=${arr[0]}
    batch_size=${arr[1]}
 
    bash runscalabilitytest.sh -h $HOSTS_LIST -m $model_name -g $GPU_PER_HOST -b $batch_size -r $REMOTE_DIR -x $MAX_GPUS
done
