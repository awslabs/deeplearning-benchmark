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
    def __init__(self):
        self.cw_ = boto3.client('cloudwatch')
        self.HEADERS = {
            'Inference' : ['Test', 'Benchmark', 'Benchmark Desc', 'Instance Type', 'Latency',
                           'P50 Latency', 'P90 Latency', 'Throughput', 'CPU Memory',
                           'GPU Memory Mean', 'GPU Memory Max', 'Uptime'],
            'Training CV' : ['Test', 'Benchmark', 'Benchmark Desc', 'Instance Type', 'Model',
                             'Precision', 'Top1 val acc', 'Top1 train acc', 'Throughput',
                             'Time to Train', 'CPU Memory', 'GPU Memory Mean', 'GPU Memory Max',
                             'Uptime'],
            'Training NLP' : ['Test', 'Benchmark', 'Benchmark Desc', 'Instance Type', 'Model',
                              'Precision', 'Perplexity', 'Throughput', 'Time to Train',
                              'CPU Memory', 'GPU Memory Mean', 'GPU Memory Max', 'Uptime']
        }
        non_metric_headers = ['Metric Prefix', 'Metric Suffix', 'Type', 'Test', 'Benchmark',
                              'Benchmark Desc', 'Instance Type', 'Num Instances', 'Model',
                              'Precision', 'Notes']
        self._benchmarks = []
        config = yaml.load(open('config/benchmarks.yaml', 'r'))

        for benchmark_keys in config['benchmarks']:
            metric_prefix =  benchmark_keys['Metric Prefix']
            metric_suffix =  benchmark_keys['Metric Suffix']
            benchmark = {}
            for k, v in benchmark_keys.items():
                if v is None:
                    continue
                elif k in non_metric_headers:
                    benchmark[k] = v
                else:
                    print(metric_prefix, v, metric_suffix)
                    metric = "{}.{}.{}".format(metric_prefix, v, metric_suffix)
                    benchmark[k] = self._get_metric(metric)

            benchmark_type = benchmark['Type']
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
        type : string
            the Training/Inference type (must be one of self.HEADERS.keys())

        Returns
        -------
        a 2-tuple of benchmarks and headers
            Each benchmark is a dict mapping the headers to a measurable or categorical value. The
            headers depend on the type specified. See self.HEADERS[type] for the list of headers.
            Note: 'Type' is included  by default as a header.
        """
        headers = self.HEADERS[type]
        return list(filter(lambda x: x['Type'] == type, self._benchmarks)), headers

    def _get_metric(self, metric):
        logging.info("Requesting data for metric {}".format(metric))

        # TODO(vishaalk): Add functionality to fetch other time periods (e.g. last quarter).
        res = self.cw_.get_metric_statistics(Namespace='benchmarkai-metrics-prod',
                                       MetricName=metric,
                                       StartTime=datetime.now() - timedelta(days=1), EndTime=datetime.now(),
                                       Period=86400, Statistics=['Average'])
        points = res['Datapoints']
        if points:
            if len(points) > 1:
                logging.warning("More than one datapoint ({}) returned for metric: {}".format(len(points), metric))
            value = points[0]['Average']
            return round(value, 2)
        else:
            logging.warning("metric {} without datapoints".format(metric))
            return ''
