# Scalability Comparison Scripts for Deep Learning Frameworks

This repository contains scripts that compares the scalability of deep learning frameworks. 

The scripts train Inception v3 and AlexNet using synchronous stochastic gradient descent (SGD). To run the comparison in reasonable time, we run few tens of iterations of SGD and compute the throughput as images processed per second. 

Comparisons can be done on clusters created with AWS CloudFormation using  the Amazon Deep Learning AMI.

###To run comparisons in a deep learning cluster created with CloudFormation

Step 1: [Create a deep learning cluster using CloudFormation](https://github.com/dmlc/mxnet/tree/master/tools/cfn).

Step 2: Log in to the master instance using SSH, including the -A option to enable SSH agent forwarding. Example: `ssh -A masternode`

Step 3: Run the following command:
`git clone https://github.com/awslabs/deeplearning-benchmark.git && cd deeplearning-benchmark/benchmark/ && bash runscalabilitytest.sh`

The runscalabilitytest.sh script runs scalability tests and records the throughput as images/sec in CSV files under 'csv_*' directories. Each line in the CSV file contains a key-value pair, where the key is the number of GPUs the test was run on and the value is the images processed per second. The script also plots this data in a SVG file named comparison_graph.svg.

Note: The following mini-batch sizes are used by default:

|              | P2 Instance   | G2 Instance  |
|--------------|------|------|
| Inception v3 | 32   | 8    |
| Alexnet      | 512  | 128  |

Mini-batch size can be changed using the --models switch. For example to run Inception-v3 with a batch size of 16 and AlexNet with a batch size of 256, run the following:
`bash runscalabilitytest.sh --models "Inceptionv3:16,Alexnet:256"`.

To run training across multiple machines, the scripts use parameter servers to update parameters. It is possible to get better performance on a single machine by not using the parameter servers. For simplicity, these scripts don't run different code optimized for a single machine for tests that run on single machine, given that we are interested only in distributed performance across multiple machines. This should not affect the results for distributed training.
