from __future__ import print_function
import argparse
import os

from ast import literal_eval
import logging
logging.basicConfig(level=logging.INFO)

try:
    import ConfigParser
    config = ConfigParser.ConfigParser()
except ImportError:
    import configparser
    config = configparser.ConfigParser()

from utils import cfg_process, metrics_manager

CONFIG_TEMPLATE_DIR = './task_config_template.cfg'
CONFIG_DIR = './task_config.cfg'





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a benchmark task.")
    parser.add_argument('--framework', type=str, help='Framework name e.g. mxnet')
    parser.add_argument('--task-name', type=str, help='Task Name e.g. resnet50_cifar10_symbolic.')
    parser.add_argument('--num-gpus', type=int, help='Numbers of gpus. e.g. --num-gpus 8')
    parser.add_argument('--epochs', type=int, help='Numbers of epochs for training. e.g. --epochs 20')
    parser.add_argument('--metrics-suffix', type=str, help='Metrics suffix e.g. --metrics-suffix daily')
    parser.add_argument('--kvstore', type=str, default='device',help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',help='floating point precision to use')
      
    
    
    args = parser.parse_args()

    # modify the template config file and generate the user defined config file.
    cfg_process.generate_cfg(CONFIG_TEMPLATE_DIR, CONFIG_DIR, **vars(args))
    config.read(CONFIG_DIR)

    # the user defined config file should only have one task
    selected_task = config.sections()[0]
    metric_patterns = literal_eval(config.get(selected_task, "patterns"))
    metric_names = literal_eval(config.get(selected_task, "metrics"))
    metric_compute_methods = literal_eval(config.get(selected_task, "compute_method"))
    command_to_execute = config.get(selected_task, "command_to_execute")
    num_gpus = int(config.get(selected_task, "num_gpus"))

    metrics_manager.benchmark(
        command_to_execute=command_to_execute,
        metric_patterns=metric_patterns,
        metric_names=metric_names,
        metric_compute_methods=metric_compute_methods,
        num_gpus=num_gpus,
        task_name=selected_task,
        suffix=args.metrics_suffix,
        framework=args.framework
    )

    # clean up
    os.remove(CONFIG_DIR)
