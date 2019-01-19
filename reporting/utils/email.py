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
Methods to send a Benchmark.AI e-mail report.
"""
import boto3
import datetime
import logging

REPORT_EMAIL_FROM = 'benchmark-ai@amazon.com'
REPORT_MAILLIST = 'benchmark-ai-reports@amazon.com'

def email_report(report_html, email_addr=REPORT_MAILLIST, footnotes=''):
    """
    Send a Benchmark.AI report to an e-mail address
    TODO(vishaalk): Validate the e-mail address format.

    Parameters
    ----------
    report_html : str
        the report formatted in HTML
    email_addr : str
        the e-mail address to send the report to
    footnotes : str
        any additional information to append to the information (e.g. non-public links). This should
        be formatted with HTML tags.
    """

    html = "{}<br/><br/>{}".format(report_html, footnotes)
    logging.info("Sending e-mail report.")
    ses = boto3.client('ses')
    response = ses.send_email(
        Source=REPORT_EMAIL_FROM,
        Destination={
            'ToAddresses': [email_addr]
        },
        Message={
            'Subject': {
                'Data': 'Benchmark.AI Report'
            },
            'Body': {
                'Html': {
                    'Data': html
                }
            }
        }
    )
    logging.info("Response: {}".format(response))
