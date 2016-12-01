#!/bin/bash

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
    -h|--hosts_file)
    NODES_FILE="$2"
    shift 
    ;;
    -n|--num_nodes)
    NUM_NODES="$2"
    shift 
    ;;
    -c|--command)
    COMMAND="$2"
    shift 
    ;;
    *)
          # unknown option
    ;;
esac
shift # past argument or value
done

if [ -z "$NUM_NODES" ]; then
    NUM_NODES=`cat $NODES_FILE | grep -v '^$' | wc -l | xargs`
fi

head -$NUM_NODES $NODES_FILE |
while read line; do

    tuple=( $line )
    ssh_alias=${tuple[1]}
    
    ssh -o "StrictHostKeyChecking no" -n $ssh_alias ${COMMAND}

done
