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
A CLI to generate benchmark reports for Benchmark.AI.

The boto3 clients require the environment variables to be set accordingly:
* AWS_ACCESS_KEY_ID
* AWS_SECRET_ACCESS_KEY
* AWS_DEFAULT_REGION
"""
import argparse
import logging
import pickle
import sys

from utils.benchmarks import Benchmarks
from utils.reports import generate_report


logging.getLogger().setLevel(logging.INFO)
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a Benchmark report.')
    parser.add_argument('-f', '--report-file', default='report.xlsx', help='the report file name.')
    parser.add_argument('-l', '--load-benchmarks', default='',
                        help='whether to load benchmarks from file (for debugging).')
    parser.add_argument('-s', '--save-benchmarks', default='',
                        help='the file to save the benchmarks (for debugging).')

    args = parser.parse_args()
    if args.load_benchmarks != '':
        benchmarks = pickle.load(open(args.load_benchmarks_file, 'rb'))
    else:
        logging.info("Reading configuration and fetching metrics from Cloudwatch.")
        benchmarks = Benchmarks()
        if args.save_benchmarks != '':
            # For pickling, remove Boto client (We don't need the boto client after this point.)
            benchmarks.cw_ = None
            pickle.dump(benchmarks, open('benchmarks.pkl', 'wb'))


    if (args.report_file == ''):
        logging.error('Report Filename required to write benchmarks.')
        sys.exit(1)
    generate_report(args.report_file, benchmarks)

