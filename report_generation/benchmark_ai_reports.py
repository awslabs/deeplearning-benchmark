#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to generate HTML and excel reports from BAI"""

__author__ = 'Pedro Larroy'
__version__ = '0.1'

import os
import sys
import subprocess
import boto3
import json
from datetime import datetime,timedelta
import xlsxwriter
from collections import defaultdict
from pprint import pprint
import logging
import pickle
from itertools import tee, filterfalse
from xlsx2html import xlsx2html


#import pandas as pd

events = boto3.client('events')
cw = boto3.client('cloudwatch')

def get_metrics():
    p = cw.get_paginator('list_metrics')
    i = p.paginate(Namespace='benchmarkai-metrics-prod')
    res = i.build_full_result()
    return list(map(lambda x: x['MetricName'], res['Metrics']))

def get_rules():
    rules = events.list_rules()['Rules'][2:]
    return list(filter(lambda x: x != 'TriggerNightlyTestStatusCheck', map(lambda x: x['Name'], rules)))

def rules_to_tasks(rules):
    """Get input from the cloudwatch rule input json data
    :returns: a map of rule name to rule input structure
    """
    res = {}
    for rule in rules:
        target = events.list_targets_by_rule(Rule=rule)['Targets'][0]
        if 'Input' in target:
            j = json.loads(target['Input'])
            #print('{} {} {} {}'.format(rule, j.get('task_name'), j.get('num_gpu'), j.get('framework_name')))
            if 'task_name' in j:
                #res.append(j['task_name'])
                res[rule] = j
    return res

def extract_metric(fullmetric, prefix, suffix):
    """Get the metric name from the full metric name string removing prefix and suffix"""
    tail = fullmetric[len(prefix)+1:]
    end = tail.rfind(suffix)-1
    return tail[:end]

def map_to_columns(metrics):
    """:returns a map of metric to column"""
    metric_set = set(metrics)
    col_map = {}
    col = 0
    for m in sorted(metric_set):
       col_map[m] = col
       col += 1
    return col_map

def fmt_value(x):
    if abs(x) >= 10:
        return '{:.0f}'.format(x)
    else:
        return '{:.2f}'.format(x)

def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = tee(iterable)
    return  list(filter(pred, t2)), list(filterfalse(pred, t1))

def groupn(xs, n):
    assert n > 0
    groups = []
    curgrp = []
    i = 0
    for x in xs:
        if i >= n:
            groups.append(curgrp)
            curgrp = []
            i = 0
        curgrp.append(x)
        i += 1
    if curgrp:
        groups.append(curgrp)
    return groups


def add_report(workbook, worksheet, row, col, benchmarks):
    border = workbook.add_format({"border":1, "border_color": "#000000"})
    number_format = workbook.add_format({"border":1, "border_color": "#000000", "align": "right", "num_format": "0.00"})
    #number_format.set_align('right')
    bold_format = workbook.add_format({'bold': True})
    def get_metrics(metric_data):
        res = []
        for (_,bench) in metric_data.items():
            res.extend(list(bench['metrics'].keys()))
        res.sort()
        return sorted(list(set(res)))

    worksheet.write(row, 0, "benchmark", bold_format)
    worksheet.write(row, 1, "suffix", bold_format)
    metrics = get_metrics(benchmarks)
    groups = groupn(metrics, 10)
    first_row = row
    for group in groups:
        col_map = map_to_columns(group)
        if row > first_row:
            worksheet.write(row, 0, "... continued benchmark")
            worksheet.write(row, 1, "suffix", bold_format)
        col = 2
        for metric in group:
            metric = metric.replace('_', ' ').replace('Average ','').replace('inference','inf.')
            worksheet.write(row, col, metric, border)
            col += 1
        last_column = col
        row += 1
        for bench_name in sorted(benchmarks.keys()):
            bench = benchmarks[bench_name]
            col = 0
            worksheet.write(row, col, bench_name,border)
            col += 1
            worksheet.write(row, col, bench['suffix'], border)
            col += 1
            for metric in group:
                metrics = bench['metrics']
                if metric in metrics:
                    value = metrics[metric]
                    itemcol = col_map[metric] + col
                    worksheet.write(row, itemcol, metric, border)
                    itemcol = col_map[metric] + col
                    #worksheet.write(row, itemcol, fmt_value(value))
                    worksheet.write_number(row, itemcol, value, number_format)
            row += 1
        row += 1
    #worksheet.add_table(first_row, 0, row, last_column)
    return row


def gen_report(path, benchmarks):
    """Generate reports for benchmarks"""
    try:
        os.remove(path)
    except OSError as e:
        pass
    assert not os.path.exists(path)

    (onnx_benchs_keys, rest_keys) = partition(lambda x: x.startswith('onnx_'), benchmarks.keys())
    (mms_benchs_keys, rest_keys) = partition(lambda x: x.startswith('mms_'), rest_keys)
    onnx_benchs_data = dict((x, benchmarks[x]) for x in onnx_benchs_keys)
    mms_benchs_data = dict((x, benchmarks[x]) for x in mms_benchs_keys)
    rest_benchs_data = dict((x, benchmarks[x]) for x in rest_keys)

    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet()
    worksheet.set_column('A:A', 20)
    row = 0

    worksheet.write(row, 0, "Benchmark AI report")
    row += 1
    row = add_report(workbook, worksheet, row, 0, rest_benchs_data)
    row += 1
    row = add_report(workbook, worksheet, row, 0, onnx_benchs_data)
    row += 1
    row = add_report(workbook, worksheet, row, 0, mms_benchs_data)
    row += 1
    workbook.close()


def gather_benchmarks():
    """
    :returns: a dictionary of {'metric': {<metric_name>: value}, ..., 'suffix': } keyed by rule
    """
    benchmarks = {}
    rules = get_rules()
    logging.info("Got {} rules".format(len(rules)))
    metrics = get_metrics()
    logging.info("Got {} metrics".format(len(metrics)))
    rules2tasks = rules_to_tasks(rules)
    #rules = pickle.load(open("rules.pkl", "rb"))
    #metrics = pickle.load(open("metrics.pkl", "rb"))
    #rules2tasks = pickle.load(open("rules2tasks.pkl", "rb"))
    pickle.dump(rules,open("rules.pkl","wb"))
    pickle.dump(metrics,open("metrics.pkl","wb"))
    pickle.dump(rules2tasks, open("rules2tasks.pkl","wb"))
    #print("metrics: {}".format(metrics))
    #print("tasks: {}".format(rules2tasks))
    #print(tasks)
    for (rule, task) in rules2tasks.items():
        metric_prefix = '.'.join((task['framework_name'], task['task_name'])) # task['metrics_suffix']))
        metric_match = list(filter(lambda x: x.startswith(metric_prefix), metrics))
        if metric_match:
            assert rule not in benchmarks
            benchmarks[rule] = {'metrics': defaultdict(dict), 'suffix': task['metrics_suffix']}
            #print("{} {}".format(metric_prefix, task['metrics_suffix']))
            for metric in metric_match:
                metric_name = extract_metric(metric, metric_prefix, task['metrics_suffix'])
                logging.info("request data for metric {}".format(metric))
                res = cw.get_metric_statistics(Namespace='benchmarkai-metrics-prod',
                                               MetricName=metric,
                                               StartTime=datetime.now() - timedelta(days=1), EndTime=datetime.now(),
                                               Period=86400, Statistics=['Average'])
                print(res)
                points = res['Datapoints']
                if points:
                    if len(points) > 1:
                        logging.warn("More than one datapoint ({}) returned for metric: {}".format(len(points), metric))
                    value = points[0]['Average']
                    #print("metric: {} {i".format(metric_name), value)
                    benchmarks[rule]['metrics'][metric_name] = value
                else:
                    #print("metric: {} N/A".format(metric_name))
                    logging.warn("metric %s : %s without datapoints", rule, metric_name)
                    pass
        else:
            logging.warn("task %s doesn't match metrics", rule)
                #print()
    return benchmarks

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    logging.getLogger('botocore').setLevel(logging.CRITICAL)
    logging.info("Gathering metrics")
    benchmarks = gather_benchmarks()
    #pickle.dump(benchmarks, open("benchmarks.pkl", "wb"))
    #benchmarks = pickle.load(open("benchmarks.pkl", "rb"))
    print(pprint(benchmarks))
    gen_report("benchmark_ai.xlsx", benchmarks)
    xlsx2html('benchmark_ai.xlsx','benchmark_ai.html')
    return 0

if __name__ == '__main__':
    sys.exit(main())

