# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# -*- coding: utf-8 -*-
"""
Utilities to fetch benchmark metrics from Benchmark AI.
"""
import boto3
import logging
import yaml

from datetime import datetime, timedelta


class Benchmarks(object):
    """
    This class provides an interface to the Benchmark.AI metrics.
    The boto3 clients require the environment variables to be set accordingly:
    * AWS_ACCESS_KEY_ID
    * AWS_SECRET_ACCESS_KEY
    * AWS_DEFAULT_REGION
    """
    HEADERS = {
        'Inference' : ['Framework', 'Framework Desc', 'Model', 'Benchmark Desc', 'Instance Type',
                       'Latency', 'P50 Latency', 'P90 Latency', 'Throughput', 'CPU Memory',
                       'GPU Memory Mean', 'GPU Memory Max', 'Uptime'],
        'Training CV' : ['Framework', 'Framework Desc', 'Model', 'Benchmark Desc', 'Instance Type',
                         'Precision', 'Top1 val acc', 'Top1 train acc', 'Throughput',
                         'Time to Train', 'CPU Memory', 'GPU Memory Mean', 'GPU Memory Max',
                         'Uptime'],
        'Training NLP' : ['Framework', 'Framework Desc', 'Model', 'Benchmark Desc', 'Instance Type',
                          'Precision', 'Perplexity', 'Throughput', 'Time to Train', 'CPU Memory',
                          'GPU Memory Mean', 'GPU Memory Max', 'Uptime']
    }
    HEADER_UNITS = {
        'Latency' : 'ms',
        'P50 Latency' : 'ms',
        'P90 Latency' : 'ms',
        'Throughput' : '/s',
        'CPU Memory' : 'mb',
        'GPU Memory' : 'mb',
        'GPU Memory Max' : 'mb',
        'GPU Memory Mean' : 'mb',
        'Time to Train' : 's',
        'Uptime' : 's'
    }
    # Some metrics have standard keys, so if the configuration does not contain the metric, we will
    # use the following.
    DEFAULT_METRIC_KEYS = {
        'Throughput' : 'throughput',
        'CPU Memory' : 'cpu_memory',
        'GPU Memory Max' : 'gpu_memory_usage_max',
        'GPU Memory Mean' : 'gpu_memory_usage_mean',
        'Time to Train' : 'time_to_train',
        'Uptime' : 'uptime_in_seconds'
    }
    CATEGORICAL_HEADERS = ['Metric Prefix', 'Metric Suffix', 'Type', 'Test', 'Framework',
                           'Framework Desc', 'Model', 'Benchmark Desc', 'Instance Type',
                           'Num Instances', 'Precision']


    def __init__(self):

        self.cw_ = boto3.client('cloudwatch')
        self._benchmarks = []
        config = yaml.load(open('config/benchmarks.yaml', 'r'))

        for benchmark_keys in config['benchmarks']:
            metric_prefix =  benchmark_keys['Metric Prefix']
            metric_suffix =  benchmark_keys['Metric Suffix']
            headers = Benchmarks.HEADERS[benchmark_keys['Type']]
            benchmark = {}
            for k in ['Type', *headers]:
                # Find a key and value pair that corresponds to a header and metric.
                v = None
                if k in benchmark_keys:
                    v = benchmark_keys[k]  # v may be None
                elif k in Benchmarks.DEFAULT_METRIC_KEYS:
                    v = Benchmarks.DEFAULT_METRIC_KEYS[k]

                if v is None:
                    continue
                elif k in Benchmarks.CATEGORICAL_HEADERS:
                    benchmark[k] = v
                else:
                    metric = "{}.{}.{}".format(metric_prefix, v, metric_suffix)
                    benchmark[k] = self._get_metric(metric)

            benchmark_type = benchmark_keys['Type']
            if benchmark_type not in self.HEADERS:
                logging.error("metric {} with invalid type".format(metric))
                continue

            headers = self.HEADERS[benchmark_type]
            # Fill in any missing headers with blank values.
            for h in headers:
                if h not in benchmark:
                    benchmark[h] = ''

            self._benchmarks.append(benchmark)


    def get_benchmarks(self, type):
        """
        Returns the benchmarks of a specific type.

        Parameters
        ----------
        type : str
            the Training/Inference type (must be one of self.HEADERS.keys())

        Returns
        -------
        a 2-tuple of benchmarks, headers
            Each benchmark is a dict mapping the headers to a measurable or categorical value. The
            headers depend on the type specified. See self.HEADERS[type] for the list of headers.
            Note: 'Type' is included  by default as a header.
        """
        headers = Benchmarks.HEADERS[type]
        return list(filter(lambda x: x['Type'] == type, self._benchmarks)), headers


    def append_header_unit(header):
        """
        Appends a unit to a header.

        Parameters
        ----------
        header : str
            a header in Benchmarks.HEADERS

        Returns
        -------
        str
            the header with an appropriate unit appended; no unit will be added if the header is
            dimensionless
        """
        if header in Benchmarks.HEADER_UNITS:
            return '{} ({})'.format(header, Benchmarks.HEADER_UNITS[header])
        else:
            return header


    def _get_metric(self, metric):
        logging.info("Requesting data for metric {}".format(metric))

        # TODO(vishaalk): Add functionality to fetch other time periods (e.g. last quarter).
        res = self.cw_.get_metric_statistics(Namespace='benchmarkai-metrics-prod',
                                       MetricName=metric,
                                       StartTime=datetime.now() - timedelta(days=7), EndTime=datetime.now(),
                                       Period=86400*7, Statistics=['Average'])
        points = res['Datapoints']
        if points:
            if len(points) > 1:
                logging.warning("More than one datapoint ({}) returned for metric: {}".format(len(points), metric))
            value = points[0]['Average']
            return round(value, 2)
        else:
            logging.warning("metric {} without datapoints".format(metric))
            return ''
