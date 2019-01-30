# MXNet Model Server Benchmarking

The benchmarks measure the performance of MMS on various models and benchmarks.
It supports passed in a URL to the .mar file.
It also runs various benchmarks using these models (see benchmarks section below).
MMS is run on the same machine in a docker instance against MMS nightly docker image from docker hub.

## Installation

The benchmarking script requires the following tools to run:
- Docker-ce with the current user added to the docker group
- Nvidia-docker (for GPU)
- ab: Apache bench to run bench mark
- bc: for metric percentile calculation
- aws cli: for upload output to S3 bucket

## Benchmarks

We support several basic benchmarks:
- MMS throughput
- MMS latency P50
- MMS latency P90
- MMS latency P99
- MMS latency mean
- MMS HTTP error rate
- Model latency P50
- Model latency P90
- Model latency P99

## Usage
The benchmarking script will automatically detect GPU/CPU based on docker runtime.
If `nvidia-docker` is installed, the benchmark will be run against GPU instance.

## Examples

Run benchmark test on resnet-50v1 model.
It use kitten.jpg image as input from: https://s3.amazonaws.com/model-server/inputs/kitten.jpg 
```bash
./benchmark.sh -u https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet50v1.mar
```

Run benchmark test on lstm_ptb model with json input
```bash
benchmark.sh -i lstm.json -u https://s3.amazonaws.com/model-server/model_archive_1.0/lstm_ptb.mar
```

By default, the script will use 100 concurrency and run 1000 requests. to change concurrent:
```bash
./benchmark.sh -c 200 -n 2000 -u https://s3.amazonaws.com/model-server/model_archive_1.0/noop-v1.0.mar
```

You can pass `-s` parameter to upload results to S3:
```bash
./benchmark.sh -s -u https://s3.amazonaws.com/model-server/model_archive_1.0/noop-v1.0.mar
```

You can also choose your local docker image to run benchmark
```bash
./benchmark.sh -d mms-cpu-local -u https://s3.amazonaws.com/model-server/model_archive_1.0/noop-v1.0.mar
```
