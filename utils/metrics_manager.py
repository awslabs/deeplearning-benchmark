
import os
import subprocess
import re
import json
import logging



# TODO add detailed error/exception handling in the script
import utils.errors
import utils.cpu_gpu_profiler

NUMERIC_PATTERN = r"(\d+\.\d+|\d+)"
RESULT_FILE_PATH = './dlbenchmark_result.json'

class BenchmarkMetricComputeMethod:
    @staticmethod
    def compute(metric_compute_method, metric):
        numWorkers = int(os.getenv('DEEPLEARNING_WORKERS_COUNT', '0'))

        if metric_compute_method == 'average':
            return 1.0 * sum(metric) / len(metric)
        elif metric_compute_method == 'last':
            return metric[-1]
        elif metric_compute_method == 'total':
            if numWorkers == 0:
                return sum(metric)
            else:
                return 1.0 * sum(metric) / numWorkers
        elif metric_compute_method == 'average_aggregate':
            assert numWorkers != 0
            return (1.0 * sum(metric) / len(metric)) * numWorkers
        else:
            raise utils.errors.MetricComputeMethodError("This metric compute method is not supported!")


class BenchmarkResultManager(object):
    def __init__(self, log_file_location, metric_patterns, metric_names, metric_compute_methods):
        """ Manages holding the map of the result data.
        :param log_file_location: string
            file location
        :param metric_patterns: list
            list of metric patterns
        :param metric_names: list
            list of metric names, in the same order as metric patterns
        :param metric_compute_methods: list
            list of metric computation method, in the same order as metric patterns
        """
        self.metric_map = {}
        if not os.path.isfile(log_file_location):
            raise Exception("log file was missing!")
        with open(log_file_location, 'rb') as f:
            self.log_file = f.read()
        assert isinstance(metric_patterns, list), "metric_patterns is expected to be a list."
        assert isinstance(metric_names, list), "metric_names is expected to be a list."
        assert isinstance(metric_compute_methods, list), "metric_compute_methods is expected to be a list."
        assert len(metric_patterns) == len(metric_names) == len(metric_compute_methods),\
            "metric_patterns, metric_names, metric_compute_methods should have same length."
        self.metric_patterns = metric_patterns
        self.metric_names = metric_names
        self.metric_compute_methods = metric_compute_methods

    @staticmethod
    def __get_float_number(s):
        matches = re.findall(NUMERIC_PATTERN, s)
        if len(matches) == 1:
            return eval(re.findall(NUMERIC_PATTERN, s)[0])
        else:
            raise utils.errors.MetricPatternError("Can not find number in the located metric pattern.")

    @staticmethod
    def uptime():
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
        return uptime_seconds

    

    def parse_log(self):
        for i in range(len(self.metric_patterns)):
            pattern = self.metric_patterns[i]
            name = self.metric_names[i]
            metric = re.findall(pattern, self.log_file)
            if len(metric) == 0:
                raise utils.errors.MetricPatternError("Can not locate provided metric pattern.")
            metric = map(self.__get_float_number, metric)
            metric_result = BenchmarkMetricComputeMethod.compute(
                metric_compute_method=self.metric_compute_methods[i],
                metric=metric
            )
            self.metric_map[name] = metric_result
        self.metric_map['uptime_in_seconds'] = self.uptime()

    def save_to(self, result_file_location):
        if os.path.isfile(result_file_location):
            os.remove(result_file_location)
        with open(result_file_location, 'w') as f:
            f.write(json.dumps(self.metric_map))


    
def benchmark(command_to_execute, metric_patterns,
              metric_names, metric_compute_methods,
              num_gpus, task_name, suffix, framework):
    """ Benchmark Driver Function
    :param command_to_execute: string
        The command line to execute the benchmark job
    :param metric_patterns: list
        list of metric patterns
    :param metric_names: list
        list of metric names, in the same order as metric patterns
    :param metric_compute_methods: list
        list of metric computation method, in the same order as metric patterns
    :param num_gpus: int
        number of gpus to use for training
    :param task_name: str
        task name
    :param suffix: str
        metric suffix in the output
    :param framework: str
        name of the framework
    :return:
    """
    log_file_location = task_name + ".log"
    log_file = open(log_file_location, 'w')
    logging.info("Executing Command: %s", command_to_execute)

    cpu_gpu_memory_usage = {}
    process = subprocess.Popen(
            command_to_execute,
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    # when num_gpus == 0, the cpu_gpu_profiler will only profile cpu usage
    with utils.cpu_gpu_profiler.Profiler(cpu_gpu_memory_usage, num_gpus, process.pid):
        process.communicate()
    log_file.close()

    result = BenchmarkResultManager(
        log_file_location=log_file_location,
        metric_patterns=metric_patterns,
        metric_names=metric_names,
        metric_compute_methods=metric_compute_methods,
    )

    result.metric_map.update(cpu_gpu_memory_usage)
    result.parse_log()

    # prepend task name and append suffix if any
    update_metric_map = {}
    for metric in result.metric_map:
        map_key = task_name + "." + metric
        if suffix:
            map_key += "." + suffix
        if framework:
            map_key = framework + "." + map_key
        update_metric_map[map_key] = result.metric_map[metric]
    logging.info(update_metric_map)
    result.metric_map = update_metric_map
    result.save_to(RESULT_FILE_PATH)
    # clean up
    #os.remove(log_file_location)
    
