# -*- coding: utf-8 -*-
u'''

This script insert data into a dataset DB in Oracle. Run: Python3 insertData.py --help for usage examples


Created on 3/10/2022

@author: Oriol Ramos Terrades (oriol.ramos@cuab.cat)
@Institution: Computer Science Dept. - Universitat Autònoma de Barcelona
'''
import math
import sys
import numpy as np
import json

import logging

import oracledb

from itertools import product
from src.conn import get_conn, hash_items
from utils import readVectorDataFile
from prettytable import PrettyTable



if __name__ == '__main__':
    db, args = get_conn()

    if args.datasetName is None:
        logging.warning("dataset name is not defined")
        exit()

    Algorithms = {
        'Support Vector Machines': 'SVC',
        'K nearest Neighbor': 'KNN',
        'Random Forest': 'RFC'
    }
    params = {
        'SVC': {'kernel': ("linear", "rbf", "poly"), 'gamma': [1, 5, 10, 20]},
        'KNN': {'n_neighbors': [3, 5, 10, 15]},
        'RFC': {
            'max_depth': [2, 4, 10, None],
            'criterion': ['gini', 'entropy', 'log_loss']}
    }

    datasets = ["Iris", "BreastCancer", "Ionosphere", "letter"]
    results = {}
    for dataset in datasets:
        results[dataset] = {}

    cur = db.cursor()
    all_experiments = PrettyTable()
    all_experiments.field_names = ["Dataset", "Classifier", "Parameters (Truncated)", "Avg Accuracy", "Avg F-Score"]
    for x, y in Algorithms.items():
        # cur.prepare("INSERT INTO CLASSIFICADOR (NOMCURT, NOM) VALUES (:1, :2)")
        # cur.execute(None, [y, x])
        c_params = params[y].keys()
        for idx, values in enumerate(product(*params[y].values())):
            cp = {k: v for k, v in zip(c_params, values) if v is not None}
            hash = hash_items(cp)
            for name in datasets:
                cur.execute(
                    "SELECT STDDEV(accuracy) AS accuracy_dev, AVG(accuracy) AS accuracy_avg, STDDEV(f_score) AS f_score_dev, AVG(f_score) AS f_score_avg FROM EXPERIMENT WHERE PAR_HASH=:1 AND NAMEDATASET =:2",
                    [hash, name])
                data = cur.fetchone()
                if data[0] is None or data == 0:
                    continue
                accuracy_dev, accuracy_avg, f_score_dev, f_score_avg = data
                results[name][hash] = {
                    'algo': y,
                    'params': cp,
                    'accuracy': {
                        'dev': accuracy_dev,
                        'avg': accuracy_avg,
                    },
                    'f_score': {
                        'dev': f_score_dev,
                        'avg': f_score_avg,
                    },
                }
                all_experiments.add_row([name, y, json.dumps(cp), str(round(accuracy_avg, 2)) + " +/- " + str(round(accuracy_dev,2)),
                                         str(round(f_score_avg,2)) + " +/- " + str(round(f_score_dev,2))])

    print(all_experiments)

    best_experiments = PrettyTable()
    best_experiments.field_names = ["Dataset", "Classifier", "Parameters (Truncated)", "Best Accuracy"]
    for dataset in results:
        bestHash = None
        for hash in results[dataset]:
            if bestHash is None or results[dataset][hash]['accuracy']['avg'] > results[dataset][bestHash]['accuracy']['avg']:
                bestHash = hash
        if bestHash is None:
            best_experiments.add_row([dataset, '-', '-', '-'])
        else:
            best_experiments.add_row(
                [dataset, results[dataset][bestHash]['algo'], json.dumps(results[dataset][bestHash]['params']),
                 round(results[dataset][bestHash]['accuracy']['avg'], 4)])

    print(best_experiments)
    db.close()
    sys.exit(0)
