#!/usr/bin/env bash
set -ex

cd /incubator-mxnet/scala-package/examples/scripts/infer/images/
max=1000
for i in `seq 2 $max`
do
    cp $1.jpg $1$i.jpg
done

echo "Done copying"