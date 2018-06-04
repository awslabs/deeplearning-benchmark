# Deep Learning Benchmark Script Driver

This repo holds the benchmark tasks that are executed regularly and also the driver script.
It is designed to be simple and flexible. Basically it is a thin wrapper on top of your benchmark script which provides
basic utilities for managing tasks, profiling gpu and cpu memories, and extract metrics from your logging.

To use the benchmark driver, you just need to look at the task_config_template.cfg to see what benchmark tasks are supported
and use the command `python benchmark_driver.py --task-name [TASK_NAME] --num-gpus [NUM_GPUS]` then the driver will trigger a
task and write the benchmark result to a JSON file called "dlbenchmark_result.json". e.g.

```
{
    'resnet50_cifar10_hybrid.total_training_time': 797.3191379999998,
    'resnet50_cifar10_hybrid.gpu_memory_usage_max': 749.0,
    'resnet50_cifar10_hybrid.cpu_memory_usage': 788,
    'resnet50_cifar10_hybrid.validation_acc': 0.560998,
    'resnet50_cifar10_hybrid.speed': 1268.8090892,
    'resnet50_cifar10_hybrid.gpu_memory_usage_std': 66.27262632335068,
    'resnet50_cifar10_hybrid.training_acc': 0.751256,
    'resnet50_cifar10_hybrid.gpu_memory_usage_mean': 602.6772235225278
}
```

There are also optional flags such as `--framework`, and `--metrics-suffix`, which are used for decorating the metrics names.

To add a new task you just need to put your benchmark script or your benchmark script directory under this repo and
add a section in the `task_config_template.cfg` like the following example:

```
 [resnet50_cifar10_imperative]
 patterns = ['Speed: (\d+\.\d+|\d+) samples/sec', 'training: accuracy=(\d+\.\d+|\d+)', 'validation: accuracy=(\d+\.\d+|\d+)', 'time cost: (\d+\.\d+|\d+)']
 metrics = ['speed', 'training_acc', 'validation_acc', 'total_training_time']
 compute_method = ['average', 'last', 'last', 'total']
 command_to_execute = python image_classification/image_classification.py --model resnet50_v1 --dataset cifar10 --gpus 8 --epochs 20 --log-interval 50
 num_gpus = 8
```

`patterns`, `metrics`, `compute_method` need to be placed in corresponding order so that the metrics map knows the key pair relationship.
Users need to defined the logging extraction rule using python's regular expression.

Typical logging looks like the following example:

```
INFO:root:Epoch[0] Batch [49]	Speed: 829.012392 samples/sec	accuracy=0.172578
INFO:root:Epoch[0] Batch [99]	Speed: 844.157227 samples/sec	accuracy=0.206563
INFO:root:Epoch[0] Batch [149]	Speed: 835.582445 samples/sec	accuracy=0.230781
INFO:root:[Epoch 0] training: accuracy=0.248027
INFO:root:[Epoch 0] time cost: 69.030488
INFO:root:[Epoch 0] validation: accuracy=0.296484
INFO:root:Epoch[1] Batch [49]	Speed: 797.687061 samples/sec	accuracy=0.322344
INFO:root:Epoch[1] Batch [99]	Speed: 821.444257 samples/sec	accuracy=0.329883
INFO:root:Epoch[1] Batch [149]	Speed: 810.339386 samples/sec	accuracy=0.342969
INFO:root:[Epoch 1] training: accuracy=0.351983
INFO:root:[Epoch 1] time cost: 61.266612
INFO:root:[Epoch 1] validation: accuracy=0.393930
```

Here you can see the pattern `Speed: (\d+\.\d+|\d+) samples/sec` correspond to `Speed: 829.012392 samples/sec`,
`Speed: 844.157227 samples/sec`, `Speed: 835.582445 samples/sec`, ... So the driver will extract these parts
into a list and map a number extractor to this list. So in the end we will get `metric=[829.012392, 844.157227,
835.582445,...]`. The `compute_method` is `average`, suppose `average([829.012392, 844.157227,
835.582445,...])=825.25` ,so it will put the pair `speed: 825.25` in the final result file.

The driver will redirect the logging into a logfile and will remove it after the metrics have been successfully
extracted.