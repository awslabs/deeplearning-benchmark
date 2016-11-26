#!/bin/bash

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -h|--nodes_file)
    NODES_FILE="$2"
    shift # past argument
    ;;
    -m|--model)
    MODEL="$2"
    shift # past argument
    ;;
    -g|--gpu_per_machine)
    MAX_GPU_PER_MACHINE="$2"
    shift # past argument
    ;;
    -b|--batch_size)
    BATCH_SIZE="$2"
    shift # past argument
    ;;
    -r|--remote_dir)
    REMOTE_DIR="$2"
    shift # past argument
    ;;
    -x|--max_gpu)
    MAX_GPUS="$2"
    shift # past argument
    ;;
    *)
          # unknown option
    ;;
esac
shift # past argument or value
done


ngpu=1
gpu_list=""
while [ "$ngpu" -le "$MAX_GPUS" ]; do
    echo "Running with $ngpu GPUs"
    
    num_machines=$(($ngpu/$MAX_GPU_PER_MACHINE))
    if (($ngpu % $MAX_GPU_PER_MACHINE)); then
        num_machines=$((num_machines + 1))
    fi
    
    gpu_per_machine=$MAX_GPU_PER_MACHINE
    if ((ngpu < MAX_GPU_PER_MACHINE)); then
        gpu_per_machine=$ngpu
    fi
    
    bash runtest.sh -m $MODEL -h $NODES_FILE -r $REMOTE_DIR -n $num_machines -g $gpu_per_machine -b $BATCH_SIZE
    
    gpu_list=${gpu_list}${ngpu},
    
    ngpu=$(($ngpu * 2))
done

gpu_list=`echo $gpu_list | rev | cut -c 2- | rev`
csv_file=logs/${MODEL}_b${BATCH_SIZE}.csv
graph_file=logs/${MODEL}_b${BATCH_SIZE}.svg
ideal_csv_file=$csv_file.ideal

bash tabulate.sh -m $MODEL -b $BATCH_SIZE -l logs -g "$gpu_list" -f $csv_file -i

#python plotgraph.py --labels="TensorFlow,Ideal" --csv="${csv_file},${ideal_csv_file}" --file=$graph_file

