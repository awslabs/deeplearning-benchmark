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
Report generation methods for creating xls and other formats.
"""
import logging
import os
import xlsxwriter

from xlsx2html import xlsx2html

HTML_EXTENSION = '.html'
XLSX_EXTENSION = '.xlsx'

def generate_report(filename_prefix, benchmarks):
    """
    Generate a report from benchmark data.

    Parameters
    ----------
    filename_prefix : string
        the output filename minus the extension for the report. Both an xlsx and html file will be
        (over-)written for thei filename_prefix.
    benchmarks : Benchmarks object
        the benchmark data for the report
    """

    html_filename = filename_prefix + HTML_EXTENSION
    xlsx_filename = filename_prefix + XLSX_EXTENSION
    try:
        os.remove(html_filename)
        os.remove(xlsx_filename)
    except OSError as e:
        pass
    assert not os.path.exists(html_filename)
    assert not os.path.exists(xlsx_filename)

    workbook = xlsxwriter.Workbook(xlsx_filename)
    worksheet = workbook.add_worksheet()
    worksheet.set_column('A:A', 20)

    row = 0
    row = add_report(worksheet, row, benchmarks, 'Inference')
    row += 3
    row = add_report(worksheet, row, benchmarks, 'Training CV')
    row += 3
    add_report(worksheet, row, benchmarks, 'Training NLP')

    workbook.close()
    xlsx2html(xlsx_filename, html_filename)


def add_report(worksheet, row, benchmarks, type):
    """
    Adds a report to a worksheet for a task type.

    Parameters
    ----------
    worksheet : an xlswriter worksheet object
        the worksheet to add the report
    row : int
        the row to add the report
    benchmarks : a Benchmarks object
        the benchmark data for the report
    type : string
        the type of benchmarks to capture, e.g. 'Training CV', 'Training NLP', 'Inference' (must be
        one of Benchmarks.HEADERS.keys()).

    Returns
    -------
    int
        the row number following the report
    """
    benchmarks, headers = benchmarks.get_benchmarks(type)
    if benchmarks is None:
        logging.warning('No benchmarks found for type "{}", skipping report'.format(type))
        return row

    worksheet.write(row, 0, 'Type')
    worksheet.write(row + 1, 0, type)

    for j, header in enumerate(headers):
        worksheet.write(row, j + 1, header)
        max_width = len(header)

        for i, benchmark in enumerate(benchmarks):
            val = str(benchmark[header])
            worksheet.write(row + i + 1, j + 1, val)
            max_width = max(max_width, len(val))


        # Set the width of the column to minimally fit all cells.
        width = worksheet.colinfo["%05d" % j][2]
        max_width = max(max_width, width)
        worksheet.set_column(j+1, j+1, max_width)

    return row + len(benchmarks)

