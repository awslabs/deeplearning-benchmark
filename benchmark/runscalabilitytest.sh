#!/bin/bash

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -m|--models)
    MODELS="$2"
    shift # past argument
    ;;
    -h|--hosts)
    HOSTS="$2"
    shift # past argument
    ;;
    -n|--hosts_count)
    HOSTS_COUNT="$2"
    shift # past argument
    ;;
    -g|--gpu_per_host)
    GPU_PER_HOST="$2"
    shift # past argument
    ;;
    -t|--ec2_instance_type)
    EC2_INSTANCE_TYPE="$2"
    shift # past argument
    ;;
    -r|--remote_dir)
    REMOTE_DIR="$2"
    shift # past argument
    ;;
    *)
          # unknown option
    ;;
esac
shift # past argument or value
done

instance_type=`curl -m 5 http://169.254.169.254/latest/meta-data/instance-type`
if [[ $instance_type ]]; then
    EC2_INSTANCE_TYPE=$instance_type
fi

if [[ $EC2_INSTANCE_TYPE ]]; then

    echo "EC2 instance type is set to " $EC2_INSTANCE_TYPE

    if [ -z ${GPU_PER_HOST+x} ]; then
        if [ "$EC2_INSTANCE_TYPE" == "p2.16xlarge" ]; then
            GPU_PER_HOST=16
        elif [ "$EC2_INSTANCE_TYPE" == "p2.8xlarge" ]; then
            GPU_PER_HOST=8
        elif [ "$EC2_INSTANCE_TYPE" == "p2.xlarge" ]; then
            GPU_PER_HOST=1
        elif [ "$EC2_INSTANCE_TYPE" == "g2.2xlarge" ]; then
            GPU_PER_HOST=1
        elif [ "$EC2_INSTANCE_TYPE" == "g2.8xlarge" ]; then
            GPU_PER_HOST=4
        else 
            echo "Unknown EC2 instance type."
            exit 1
        fi
    fi


    if [ -z ${HOSTS+x} ]; then 
        cf_hosts_file="/opt/deeplearning/workers"
        if [ -f $default_hosts_file ]; then
            cat /etc/hosts | grep deeplearning-worker > nodes
            HOSTS=nodes
        fi
    fi


    if [ -z ${MODELS+x} ]; then 
        if [[ $EC2_INSTANCE_TYPE == p2* ]]; then
            MODELS="Alexnet:512,Inceptionv3:32"
        elif [[ $EC2_INSTANCE_TYPE == g2* ]]; then
            MODELS="Alexnet:128,Inceptionv3:8"
        fi
    fi

fi


if [ -z ${HOSTS+x} ]; then 
    echo "Hosts not specified"
    exit 1
else
    echo "Using hosts from $HOSTS"
fi

if [ -z ${MODELS+x} ]; then 
    echo "Models not specified"
    exit 1
else
    echo "Running models: $MODELS"
fi

if [ -z ${HOSTS_COUNT+x} ]; then
    HOSTS_COUNT=`cat $HOSTS | grep -v '^$' | wc -l`
fi
echo "Using $HOSTS_COUNT hosts"

if [ -z ${GPU_PER_HOST+x} ]; then
    echo "GPUs per host not specified"
    exit 1
else
    echo "Using $GPU_PER_HOST GPUs per host"
fi

if [ -z ${REMOTE_DIR+x} ]; then
    REMOTE_DIR="/tmp/"
fi
echo "Using $REMOTE_DIR as remote directory"


if [ ! -d "mxnet" ]; then
    echo "Cloning MXNet"
    git clone https://github.com/dmlc/mxnet.git
    cd mxnet && git reset --hard a3a928c21ab91b246a5fab7c9ec135f6e616f899
    git clone https://github.com/dmlc/dmlc-core dmlc-core
    cd dmlc-core && git reset --hard f554de0a6914f8028aab50aea02003a4344e732d
    cd ../..
fi

# Create the hostname list required for MXNet
rm -f hostnames
head -$HOSTS_COUNT $HOSTS |
while read line; do
    if [ -z line ]; then continue; fi
    arr=( $line )
    host_name=${arr[0]}
    echo $host_name >> hostnames
done


echo "Compressing MXNet"
rm -f mxnet.tar.gz
tar -cvzf mxnet.tar.gz ./mxnet > /dev/null 2>&1

echo "Copying MXNet to remote nodes..."
head -$HOSTS_COUNT $HOSTS |
while read line; do
    if [ -z line ]; then continue; fi
    arr=( $line )
    ssh_alias=${arr[1]}

    scp mxnet.tar.gz $ssh_alias:$REMOTE_DIR
    scp hostnames $ssh_alias:$REMOTE_DIR
    ssh $ssh_alias 'cd '${REMOTE_DIR}' && tar -xvzf mxnet.tar.gz > /dev/null 2>&1' &
done

# Construct the models string for MXNet 
mxnet_model_string=$MODELS
mxnet_model_string=`echo $mxnet_model_string | sed "s/Alexnet/alexnet/g"`
mxnet_model_string=`echo $mxnet_model_string | sed "s/Inceptionv3/inception-v3/g"`
mxnet_model_string=`echo $mxnet_model_string | sed "s/Resnet/resnet/g"`
mxnet_model_string=`echo $mxnet_model_string | sed "s/inception-v3:[0-9]*/&:299/g"`
mxnet_model_string=`echo $mxnet_model_string | sed "s/alexnet:[0-9]*/&:224/g"`
mxnet_model_string=`echo $mxnet_model_string | sed "s/resnet:[0-9]*/&:224/g"`
mxnet_model_string=`echo $mxnet_model_string | sed "s/,/' '/g"`
mxnet_model_string="'"${mxnet_model_string}"'"

# Construct the command to run MXNet tests
image_recog_dir="${REMOTE_DIR}/mxnet/example/image-classification/"
mxnet_command="cd $image_recog_dir && python benchmark.py --worker_file ${REMOTE_DIR}/hostnames --worker_count ${HOSTS_COUNT} --gpu_count ${GPU_PER_HOST} --networks ${mxnet_model_string}"
echo $mxnet_command

# Run the MXNet test from the first machine in the hosts list
line=$(head -n 1 $HOSTS)
arr=( $line )
master_host=${arr[1]}
ssh $master_host $mxnet_command
rm -rf csv_mxnet
mkdir csv_mxnet
scp ${master_host}:${REMOTE_DIR}/mxnet/example/image-classification/benchmark/*.csv ./csv_mxnet

# Run TensorFlow
rm -rf csv_tf
mkdir csv_tf
current_dir=$PWD
cp $HOSTS tensorflow/nodes
cd tensorflow
echo bash runall.sh -m $MODELS -h nodes -g $GPU_PER_HOST -r $REMOTE_DIR -x "$((HOSTS_COUNT * GPU_PER_HOST))"
bash runall.sh -m $MODELS -h nodes -g $GPU_PER_HOST -r $REMOTE_DIR -x "$((HOSTS_COUNT * GPU_PER_HOST))"
cp logs/*.csv $current_dir/csv_tf
cd $current_dir

# Remove the header from MXNet CSV files
for file in ./csv_mxnet/*.csv; do
    sed '1d' ${file} > ${file}.bak; mv ${file}.bak ${file}
    echo $file
done


#Plot graph

if [[ $MODELS == *"Alexnet"* ]]
then
    labels=${labels}"Alexnet on MXNet,Alexnet on TensorFlow,"
    csv_files=${csv_files}"csv_mxnet/`ls csv_mxnet | grep -i alexnet`,csv_tf/`ls csv_tf | grep -i alexnet`,"
fi

if [[ $MODELS == *"Inception"* ]]
then
    labels=${labels}"Inception-v3 on MXNet,Inception-v3 on TensorFlow,"
    csv_files=${csv_files}"csv_mxnet/`ls csv_mxnet | grep -i inception`,csv_tf/`ls csv_tf | grep -i inception`,"
fi

if [[ $MODELS == *"Resnet"* ]]
then
    labels=${labels}"Resnet-152 on MXNet,Resnet-152 on TensorFlow,"
    csv_files=${csv_files}"csv_mxnet/`ls csv_mxnet | grep -i resnet`,csv_tf/`ls csv_tf | grep -i resnet`,"
fi

labels=${labels%?}
csv_files=${csv_files%?}

a_csv_file=`ls csv_*/*.csv | xargs echo | tr ' ' '\n' | head -1`
num_lines=`cat ${a_csv_file} | sed '/^\s*$/d' | wc -l | xargs`

max_gpu=$(($HOSTS_COUNT * $GPU_PER_HOST))
python plotgraph.py --labels="${labels}" --csv="${csv_files}" --file=comparison_graph.svg --maxgpu=$max_gpu
