import os
import signal
import time
import subprocess
import pandas as pd
from collections import defaultdict
from threading import Event, Thread
import psutil
from .errors import CommandExecutionError


# redirect the GPU memory usage to a file
#GPU_MONITOR = "nvidia-smi --query-gpu=index,memory.used --format=csv -lms 500 -f output.csv"
GPU_MONITOR = "nvidia-smi --query-gpu=index,memory.used --format=csv -l 120 -f output.csv"

def gpu_memory_usage_extract(file_name, ret_dict, num_gpus):
    """Extract GPU usage from the nvidia-smi output file"""
    try:
        gpu_memory_usage = pd.read_csv(file_name)
    except pd.errors.EmptyDataError:
        print("Error! The GPU profiling output is empty!")
        raise

    # filter out break line
    gpu_memory_usage = gpu_memory_usage[~pd.isnull(gpu_memory_usage.iloc[:, 1])]
    gpu_memory_usage = gpu_memory_usage[gpu_memory_usage.iloc[:, 1].str.contains('MiB')]

    # set a default dict to collect gpu mean, var and max usage, default value is an empty list
    record = defaultdict(list)

    # compute the average memory usage on each gpu
    for i in range(num_gpus):
        gpu_usage_i = gpu_memory_usage[gpu_memory_usage.iloc[:,0] == i].iloc[:,1]
        gpu_usage_i = gpu_usage_i.str.extract('(\d+)', expand=False).astype(int)
        mean_use_gpu_i = gpu_usage_i.mean()
        std_use_gpu_i = gpu_usage_i.std()
        max_use_gpu_i = gpu_usage_i.max()
        record['mean_usage'].append(mean_use_gpu_i)
        record['std_usage'].append(std_use_gpu_i)
        record['max_usage'].append(max_use_gpu_i)

    if len(record) == 0:
        return
    ret_dict['gpu_memory_usage_mean'] = float(sum(record['mean_usage'])) / len(record['mean_usage'])
    ret_dict['gpu_memory_usage_std'] = float(sum(record['std_usage'])) / len(record['std_usage'])
    ret_dict['gpu_memory_usage_max'] = float(sum(record['max_usage'])) / len(record['max_usage'])
    os.remove(file_name)


def get_cpu_mem_usage_from_process(pid, cpu_usage):
    """Get CPU memory usage given process id, result append to a mutable list."""
    if not psutil.pid_exists(pid):
        return
    proc = psutil.Process(pid)
    if proc.is_running():
        # The rss, Resident Set Size, is the memory allocated to the process, its unit is KB.
        cpu_usage.append(proc.memory_info().rss / 1024)


class RepeatedQuery:
    """Use another a thread to repeatly execute a given function at a given time interval"""
    def __init__(self, interval, function, *args, **kwargs):
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start = time.time()
        self.event = Event()
        self.thread = Thread(target=self._target)
        self.thread.start()

    def _target(self):
        while not self.event.wait(self._time):
            self.function(*self.args, **self.kwargs)

    @property
    def _time(self):
        return self.interval - ((time.time() - self.start) % self.interval)

    def stop(self):
        self.event.set()
        self.thread.join()


class Profiler(object):
    """The CPU GPU memory profiler"""
    def __init__(self, ret_dict, num_gpus, process_id):
        self.__ret_dict = ret_dict
        self.num_gpus = num_gpus
        self.cpu_usage = []
        self.cpu_mem_repeat_query = RepeatedQuery(
            interval=5,
            function=get_cpu_mem_usage_from_process,
            pid=process_id,
            cpu_usage=self.cpu_usage
        )

    def __enter__(self):
        if self.num_gpus < 1:
            return self
        open("output.csv", 'a').close()
        self.__gpu_monitor_process = subprocess.Popen(
            GPU_MONITOR,
            shell=True,
            preexec_fn=os.setsid
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cpu_mem_repeat_query.stop()
        if len(self.cpu_usage) == 0:
            raise CommandExecutionError
        cpu_usage = sum(self.cpu_usage) / len(self.cpu_usage)
        self.__ret_dict['cpu_memory_usage'] = cpu_usage

        if self.num_gpus < 1:
            return
        os.killpg(os.getpgid(self.__gpu_monitor_process.pid), signal.SIGTERM)
        # to solve race condition
        time.sleep(1)
        gpu_memory_usage_extract(
            file_name="output.csv",
            ret_dict=self.__ret_dict,
            num_gpus=self.num_gpus
        )

