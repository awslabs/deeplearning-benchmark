#!/bin/bash

ADD_IDEAL=0

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -m|--model)
    MODEL="$2"
    shift # past argument
    ;;
    -b|--batch_size)
    BATCH_SIZE="$2"
    shift # past argument
    ;;
    -l|--log_dir)
    LOG_DIR="$2"
    shift # past argument
    ;;
    -g|--gpu_list)
    GPU_LIST="$2"
    shift # past argument
    ;;
    -f|--table_file)
    TABLE_FILE="$2"
    shift # past argument
    ;;
    -i|--add_ideal)
    ADD_IDEAL=1
    shift # past argument
    ;;
    *)
          # unknown option
    ;;
esac
shift # past argument or value
done

echo "Writing table to $TABLE_FILE"
IDEAL_FILE=$TABLE_FILE.ideal
rm $TABLE_FILE 2>/dev/null
rm $IDEAL_FILE 2>/dev/null

images_one_gpu=`cat ${LOG_DIR}/${MODEL}_b${BATCH_SIZE}_g1/imagespersec`

for gpu in $(echo $GPU_LIST | sed "s/,/ /g")
do
    images_per_gpu=`cat ${LOG_DIR}/${MODEL}_b${BATCH_SIZE}_g${gpu}/imagespersec`
    images_per_cluster=`echo $images_per_gpu*$gpu | bc`
    echo ${gpu},${images_per_cluster} >> $TABLE_FILE
    ideal=`echo $images_one_gpu*$gpu | bc`
    echo ${gpu},$ideal >> $IDEAL_FILE
done
