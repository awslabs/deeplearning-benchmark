##Comparing scalability of deep learning frameworks

This repository contains scripts that compares the scalability of deep learning frameworks. The scripts train Inception v3 and Alexnet using synchronous Stochastic Gradient Descent. To run the comparison in reasonable time, we run few tens of iterations of SGD and compute the throughput as images processed per second. 

Comparison can be done on clusters created with Amazon CloudFormation using Amazon Deep Learning AMI.

###Running comparison in deep learning cluster created with CloudFormation:

Step 1: Create a deep learning cluster using CloudFormation following the instructions here: https://github.com/dmlc/mxnet/tree/master/tools/cfn

Step 2: Login to master instance using ssh including the -A option to enable ssh agent forwarding. Example: `ssh -A masternode`

Step 3: `git clone https://github.com/awslabs/deeplearning-benchmark.git && cd deeplearning-benchmark/benchmark/ && bash runscalabilitytest.sh`

runscalabilitytest.sh runs scalability tests and records the throughput as images/sec in CSV files under 'csv_*' directories. Each line in the CSV file contains a key-value pair where key is the number of GPUs the test was run on and value is the images processed per second. The script also plots this data in a SVG file named comparison_graph.svg.

Note: The following mini-batch sizes are used by default:

|              | P2 Instance   | G2 Instance  |
|--------------|------|------|
| Inception v3 | 32   | 8    |
| Alexnet      | 512  | 128  |

Mini-batch size can be changed using the --models switch. For example to run Inception-v3 with a batch size of 16 and Alexnet with a batch size of 256, please run `bash runscalabilitytest.sh --models "Inceptionv3:16,Alexnet:256"`.
