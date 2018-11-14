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

from .benchmarks import Benchmarks
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
    except OSError as e:
        pass
    assert not os.path.exists(html_filename)

    try:
        os.remove(xlsx_filename)
    except OSError as e:
        pass
    assert not os.path.exists(xlsx_filename)

    # Create format references in the workbook.
    workbook = xlsxwriter.Workbook(xlsx_filename)
    formats = {
        'header' : workbook.add_format(
            {
                'align': 'center',
                'bold': True,
                'border': 2,
                'border_color': '#9ea3aa',
                'bg_color': '#dadfe8'
            }
        ),
        'categorical' : workbook.add_format(
            {
                'align': 'center',
                'border': 1,
                'border_color': '#9ea3aa',
                'bg_color': '#f2f6fc'
            }
        ),
        'number' : workbook.add_format(
            {
                'align': 'center',
                # 'num_format': '0.00', Doesn't appear to be working.
                'border': 1,
                'border_color': '#9ea3aa'
            }
        ),
        'number-alarm-state' : workbook.add_format(
            {
                'align': 'center',
                # 'num_format': '0.00', Doesn't appear to be working.
                'bg_color': '#e51b1b',
                'border': 1,
                'border_color': '#9ea3aa'
            }
        ),
        'bold' : workbook.add_format(
            {
                'bold': 'True'
            }
        )
    }

    worksheet = workbook.add_worksheet()
    worksheet.set_column('A:A', 20)

    row = 0
    for benchmark_type in Benchmarks.HEADERS.keys():
        row = _add_report(worksheet, formats, row, benchmarks, benchmark_type)
        row += 3

    REPORT_FOOTNOTE = "Note: Data averaged weekly."
    worksheet.write(row, 0, REPORT_FOOTNOTE, formats['bold'])


    workbook.close()
    xlsx2html(xlsx_filename, html_filename)


def _add_report(worksheet, formats, row, benchmarks, benchmark_type):
    """
    Adds a report to a worksheet for a task type.

    Parameters
    ----------
    worksheet : an xlswriter worksheet object
        the worksheet to add the report
    formats : dict
        a dictionary of worksheet formats
    row : int
        the row to add the report
    benchmarks : a Benchmarks object
        the benchmark data for the report
    benchmark_type : str
        the type of benchmarks to capture, e.g. 'Training CV', 'Training NLP', 'Inference' (must be
        one of Benchmarks.HEADERS.keys()).

    Returns
    -------
    int
        the row number following the report
    """
    benchmarks, headers = benchmarks.get_benchmarks(benchmark_type)

    if benchmarks is None:
        logging.warning('No benchmarks found for type "{}", skipping report'.format(benchmark_type))
        return row

    worksheet.write(row, 0, 'Type', formats['header'])
    worksheet.write(row + 1, 0, benchmark_type, formats['header'])

    for j, header in enumerate(headers):
        worksheet.write(row, j + 1, Benchmarks.append_header_unit(header), formats['header'])
        max_width = len(header)

        for i, benchmark in enumerate(benchmarks):
            values = benchmark[0]
            alarms = benchmark[1]

            value = str(values[header])
            is_alarm_state = header in alarms and alarms[header]
            if header in Benchmarks.CATEGORICAL_HEADERS:
                format = formats['categorical']
            elif is_alarm_state:
                format = formats['number-alarm-state']
            else:
                format = formats['number']

            if header == 'Framework' and 'DashboardUri' in values:
                uri = str(values['DashboardUri'])
                worksheet.write_url(row + i + 1, j + 1, uri, cell_format=format, string=value)
            elif is_alarm_state:
                # We link to the first alarm only.
                worksheet.write_url(row + i + 1, j + 1, alarms[header][0], cell_format=format,
                                    string=value)
            else:
                worksheet.write(row + i + 1, j + 1, value, format)

            max_width = max(max_width, len(value))


        # Set the width of the column to minimally fit all cells.
        width = worksheet.colinfo["%05d" % j][2]
        max_width = max(max_width, width)
        worksheet.set_column(j+1, j+1, max_width)

    return row + len(benchmarks)

