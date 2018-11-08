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
The AWS Lambda handler to generate benchmark reports for Benchmark.AI.

The following environment variables will normally be set by AWS Lambda. Boto3 requires them to be
set accordingly.

* AWS_ACCESS_KEY_ID
* AWS_SECRET_ACCESS_KEY
* AWS_DEFAULT_REGION
"""
import io
import json
import logging
import os

from utils.benchmarks import Benchmarks
from utils.email import email_report
from utils.reports import HTML_EXTENSION
from utils.reports import generate_report


logging.getLogger().setLevel(logging.INFO)
logging.getLogger('boto3').setLevel(logging.CRITICAL)
logging.getLogger('botocore').setLevel(logging.CRITICAL)


def lambda_handler(event, context):
    """
    Main entry point for AWS Lambda.
    The EMAIL environment variable is required to be set.
    """
    logging.info("Reading configuration and fetching metrics from Cloudwatch.")
    benchmarks = Benchmarks()

    logging.info('Generating report.')
    REPORT_FILE = '/tmp/bai-report'
    generate_report(REPORT_FILE, benchmarks)
    report_html = io.open(REPORT_FILE + HTML_EXTENSION, mode='r', encoding='utf-8').read()

    logging.info('Emailing report.')
    EMAIL = os.environ['EMAIL']
    email_report(report_html, EMAIL)

    return {
        "statusCode": 200,
        "body": json.dumps('Report successfully generated and e-mailed.')
    }
