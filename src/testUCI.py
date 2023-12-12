import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from datetime import datetime
import socket
from itertools import product
from conn import get_conn, hash_items

import oracledb
from GABDConnect.oracleConnection import oracleConnection as orcl

if __name__ == "__main__":
    """
        Els únics arguments necessaris seran --datasetName, pel nom del dataset on volem fer els experiments
        Totes les opcions:
        --datasetName Iris          
        --datasetName BreastCancer  
        --datasetName Ionosphere    
        --datasetName letter       
    """
    db_write, args = get_conn("DevUCI")
    numIterations = 50
    shortName = {
        'SVC': 'SVC',
        'KNeighborsClassifier': 'KNN',
        'RandomForestClassifier': 'RFC'
    }
    Algorithms = {
        'Support Vector Machines': SVC,
        'K nearest Neighbor': KNN,
        'Random Forest': RFC
    }
    params = {
        'SVC': {'kernel': ("linear", "rbf", "poly"), 'gamma': [1, 5, 10, 20]},
        'KNeighborsClassifier': {'n_neighbors': [3, 5, 10, 15]},
        'RandomForestClassifier': {
            'max_depth': [2, 4, 10, None],
            'criterion': ['gini', 'entropy', 'log_loss']}
    }

    try:
        # Preparar la consulta para leer muestras
        db_read, args = get_conn("TestUCI")
        cur = db_read.cursor()
        cur.execute("SELECT ID, FEATURES, LABEL FROM GestorUCI.Samples WHERE NAMEDATASET = :1",
                    [args.datasetName])
        samples = cur.fetchall()
        cur.execute("SELECT FEAT_SIZE FROM GestorUCI.Dataset WHERE NAME = :1", [args.datasetName])
        dataset_data = cur.fetchone()

        data = {'ID': [], 'FEATURES': [], 'LABEL': []}
        for id, feature_blob, label in samples:
            feature_array = np.frombuffer(feature_blob.read(), dtype=np.float64)
            data['ID'].append(id)
            data['FEATURES'].append(feature_array)
            data['LABEL'].append(label)

        df = pd.DataFrame(data)
        df.set_index('ID', inplace=True)

        Xo = np.array(df['FEATURES'].tolist())
        Yo = df['LABEL'].values

        db_read.close()
        np.random.seed(0)
        current_time = datetime.now().strftime("%d/%m/%Y, %H:%M")

        nameDataset = args.datasetName

        for k in Algorithms:
            classificador = Algorithms[k].__name__
            nomCurt = shortName[classificador]
            c_params = params[classificador].keys()
            for idx, values in enumerate(product(*params[classificador].values())):
                cp = {k: v for k, v in zip(c_params, values) if v is not None}
                clf = Algorithms[k](**cp)
                hash_param = hash_items(cp)
                cur = db_write.cursor()
                cur.execute("DELETE FROM GestorUCI.EXPERIMENT WHERE NAMEDATASET =:1 AND PAR_HASH =:2 AND NOMCURT =:3",
                            [nameDataset, hash_param, classificador])
                cur.prepare(
                    "INSERT INTO GestorUCI.EXPERIMENT (NAMEDATASET, NOMCURT, PAR_HASH, DATA, ACCURACY, F_SCORE) VALUES (:1, :2, :3, TO_DATE(:4, 'DD/MM/YYYY HH24:MI'), :5, :6)")
                for i in range(numIterations):
                    X_train, X_test, y_train, y_test = train_test_split(Xo, Yo, test_size=0.1)

                    clf.fit(X_train, y_train)

                    y_pred = clf.predict(X_test)

                    f_score = f1_score(y_test, y_pred, average='macro')
                    acc = accuracy_score(y_test, y_pred)
                    try:
                        cur.execute(None, [nameDataset, nomCurt, hash_param, current_time, acc, f_score])
                        print("Resultados insertados correctamente.")

                    except Exception as e:
                        print(f"Error al insert resultados: {e}")
                db_write.commit()

    finally:
        if db_write is not None:
            db_write.close()
