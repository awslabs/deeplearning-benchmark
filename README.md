##Comparing scalability of deep learning frameworks

This repository contains scripts that compares the scalability of deep learning frameworks. The scripts train Inception v3 and Alexnet using synchronous SGD. To run the comparison in reasonable time, we run few tens of iterations of SGD and compute the throughput as images processed per second. 

Comparison can be done on clusters created with Amazon CloudFormation using Amazon Deep Learning AMI or on user's own cluster. 

###Running comparison in deep learning cluster created with CloudFormation:

Step 1: Create a deep learning cluster using CloudFormation following the instructions here: https://github.com/dmlc/mxnet/tree/master/tools/cfn

Step 2: Login to master instance using ssh including the -A option to enable ssh agent forwarding. Example: `ssh -A masternode`

Step 3: `git clone https://github.com/awslabs/deeplearning-benchmark.git && cd deeplearning-benchmark/benchmark/ && bash runscalabilitytest.sh`
