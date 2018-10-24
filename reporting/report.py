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
import io
import logging
import pickle
import sys

from utils.benchmarks import Benchmarks
from utils.email import email_report
from utils.reports import HTML_EXTENSION
from utils.reports import generate_report


logging.getLogger().setLevel(logging.INFO)
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a Benchmark report.')
    parser.add_argument('-f', '--report-file', default='report', help='the report file name \
                        minus the extension. An .xlsx and a .html report will be generated.')
    parser.add_argument('-l', '--load-benchmarks', default='',
                        help='whether to load benchmarks from file (for debugging).')
    parser.add_argument('-s', '--save-benchmarks', default='',
                        help='the file to save the benchmarks (for debugging).')
    parser.add_argument('-e', '--email-addr', default='',
                        help='send a report to e-mail address.')
    parser.add_argument('-lm', '--list-all-metrics', default='false',
                        help='gets all Benchmark.AI the metrics (for debugging')

    args = parser.parse_args()
    if args.list_all_metrics == 'true':
        benchmarks = Benchmarks(fetch_metrics=False)
        print('\n'.join(benchmarks.list_all_metrics()))
        sys.exit(0)

    if args.load_benchmarks:
        logging.info("Loading benchmarks from {}".format(args.load_benchmarks))
        benchmarks = pickle.load(open(args.load_benchmarks, 'rb'))
    else:
        logging.info("Reading configuration and fetching metrics from Cloudwatch.")
        benchmarks = Benchmarks()

        if args.save_benchmarks:
            logging.info("Saving benchmarks to {}".format(args.save_benchmarks))
            # For pickling, remove Boto client (we don't need the boto client after this point).
            benchmarks._cw = None
            pickle.dump(benchmarks, open(args.save_benchmarks, 'wb'))

    # TODO(vishaalk): If e-mail is requested and report file name not specified, use a temp file.
    if not args.report_file:
        logging.error('Report filename prefix required to generate report. Skipping report.')
        sys.exit(1)

    logging.info('Generating report.')
    generate_report(args.report_file, benchmarks)
    if args.email_addr:
        report_html = io.open(args.report_file + HTML_EXTENSION, mode='r', encoding='utf-8').read()
        email_report(report_html, args.email_addr)
