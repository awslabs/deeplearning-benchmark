#!/bin/bash

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h|--hosts_file)
    HOSTS_FILE="$2"
    shift 
    ;;
    --use_ssh_config)
    ssh_opts="-F $2"
    shift
    ;;    
    *)
          # unknown option
    ;;
esac
shift # past argument or value
done

cat $HOSTS_FILE |
while read line; do

    tuple=( $line )
    ssh_alias=${tuple[1]}

    ssh -o "StrictHostKeyChecking no" $ssh_opts -n $ssh_alias "ps -ef | grep 'ps_hosts' | grep -v grep | sed 's/ \+/ /g' | cut -d ' ' -f 2 | xargs kill -9" > /dev/null 2>&1
    ssh -o "StrictHostKeyChecking no" $ssh_opts -n $ssh_alias "ps -ef | grep tail | grep worker | grep -v grep | sed 's/ \+/ /g' | cut -d ' ' -f 2 | xargs kill -9" > /dev/null 2>&1
done
