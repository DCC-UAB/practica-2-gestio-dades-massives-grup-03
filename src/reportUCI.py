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


def insertVectorDataset(dbConn, nameDataset, fileName, label_pos, *args, **kwargs):
    """
      Inserts the contents stored in fileName into de DB
      Params:
      :param dbConn: handle to an active (and open) connexion to Oracle DB.
      :param nameDataset: name of the dataset to be inserted into the database.
      :param fileName: full path of the file where the data to be imported is stored.
      :return: Boolean value whether data have properly been inserted into de database
    """
    # Leer los datos del archivo y generar un dataframe
    df, ids = readVectorDataFile(fileName, label_pos=label_pos)

    # Iniciar la conexión con la base de datos

    # Verificar si el dataset ya existe
    cur.execute("SELECT COUNT(*) FROM Dataset WHERE NAME = :1", [nameDataset])
    exist = cur.fetchone()[0] != 0

    if not exist:

        # Calcular tamaño de las características
        feat_size = len(df['features'].iloc[0])

        # Determinar la posición de la etiqueta y calcular el número de clases
        num_classes = len(ids)

        # Preparar y ejecutar la inserción de los datos del dataset
        cur.prepare("INSERT INTO Dataset (NAME, FEAT_SIZE, NUMCLASSES) VALUES (:1, :2, :3)")
        cur.execute(None, [nameDataset, feat_size, num_classes])
        logging.warning(
            f"Insertando dataset {nameDataset} con {feat_size} características y {num_classes} clases")

        # Preparar la consulta para insertar muestras
        cur.prepare("INSERT INTO Samples (NAMEDATASET, ID, FEATURES, LABEL) VALUES (:1, :2, :3, :4)")
        for index, row in df.iterrows():
            blobFeatures = cur.var(oracledb.BLOB)
            features_np_array = np.array(row['features'])
            blobFeatures.setvalue(0, features_np_array.tobytes())

            cur.execute(None, [nameDataset, index, blobFeatures, row['class']])

        confirm = "Datos insertados correctamente en el nuevo dataset"
    else:
        confirm = "El dataset {nameDataset} ya existe, no se realizaron inserciones"

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

    cur.execute("SELECT COUNT(*) FROM CLASSIFICADOR WHERE NOMCURT = :1", [Algorithms['Support Vector Machines']])
    exist = cur.fetchone()[0] != 0
    if exist:
        for x, y in Algorithms.items():
            # cur.prepare("INSERT INTO CLASSIFICADOR (NOMCURT, NOM) VALUES (:1, :2)")
            # cur.execute(None, [y, x])
            c_params = params[y].keys()
            for idx, values in enumerate(product(*params[y].values())):
                cp = {k: v for k, v in zip(c_params, values) if v is not None}
                cur.prepare("INSERT INTO PARAMETRES (NOMCURT, HASH, VALORS) VALUES (:1, :2, :3)")
                valors = cp
                var = cur.var(oracledb.DB_TYPE_JSON)
                var.setvalue(0, valors)
                cur.execute(None, [y, hash_items(cp), var])
        confirm += " y datos de classificador insertados"
    else:
        confirm += " y classificadores ya existen"
    # Intentar hacer commit y manejar excepciones
    try:
        dbConn.commit()
        print(confirm)
        return True
    except Exception as e:
        print(f"Error al hacer commit: {e}")
        return False


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
