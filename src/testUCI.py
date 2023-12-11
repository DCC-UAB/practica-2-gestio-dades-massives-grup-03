import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
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

    db_write, args = get_conn("DevUCI")
    numIterations = 50
    Algorithms = {
        'SVC': SVC,
        'KNN': KNN,
        'RFC': RFC
    }
    params = {
        'SVC': {'kernel': args.kernel, 'gamma': args.gamma},
        'KNN': {'n_neighbors': args.n_neighbors},
        'RFC': {'max_depth': args.max_depth, 'criterion': args.criterion}
    }

    try:
        # Preparar la consulta para leer muestras
        db_read, args = get_conn("TestUCI")
        cur = db_read.cursor()
        cur.execute("SELECT NAMEDATASET, ID, FEATURES, LABEL FROM GestorUCI.Samples WHERE NAMEDATASET = :1",
                    [args.datasetName])
        data = cur.fetchall()
        cur.execute("SELECT FEAT_SIZE FROM GestorUCI.Dataset WHERE NAME = :1", [args.datasetName])
        dataset_data = cur.fetchone()

        Yo = []
        Xo = np.ndarray(shape=(len(data), dataset_data[0]), dtype=np.float64)

        for i, row in enumerate(data):
            Xo[i] = np.frombuffer(row[2].read())
            Yo.append(row[3])
        Yo = np.array(Yo)
        db_read.close()

        n_sample = len(Xo)

        np.random.seed(0)
        current_time = datetime.now().strftime("%d/%m/%Y, %H:%M")

        cp = params[args.alg]
        clf = Algorithms[args.alg](**cp)
        hash_param = hash_items(cp)
        nameDataset = args.datasetName
        cur = db_write.cursor()
        cur.execute("DELETE FROM GestorUCI.REPETICIO WHERE NAMEDATASET =:1", [nameDataset])

        # TODO EN VERDAD SE TENIA QUE HACER CON TODOS LOS ALGORITMOS
        # TODO ALMACENAR CADA
        #
        for i in range(numIterations):
            order = np.random.permutation(n_sample)
            X = Xo[order]
            y = Yo[order]

            X_train = X[: int(0.9 * n_sample)]
            y_train = y[: int(0.9 * n_sample)]
            y_test = y[int(0.9 * n_sample):]
            X_test = X[int(0.9 * n_sample):]

            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            f_score = f1_score(y_test, y_pred, average='macro')
            acc = accuracy_score(y_test, y_pred)
            print('Insert number:', i)
            # def insert_results(db_conn, nameDataset, classifier, parametres, iteration, time, fi_score, accuracy):
            try:
                cur.prepare("INSERT INTO GestorUCI.REPETICIO (NAMEDATASET, NUM) VALUES (:1, :2)")
                cur.execute(None, [nameDataset, i])
                cur.prepare(
                    "INSERT INTO GestorUCI.EXPERIMENT (NAMEDATASET, NOMCURT, PAR_HASH, REP_NUM, DATA, ACCURACY, F_SCORE) VALUES (:1, :2, :3, :4, TO_DATE(:5, 'DD/MM/YYYY HH24:MI'), :6, :7)")
                cur.execute(None, [nameDataset, args.alg, hash_param, i, current_time, acc, f_score])
                db_write.commit()
                print("Resultados insertados correctamente.")

            except Exception as e:
                print(f"Error al insert resultados: {e}")

    finally:
        if db_write is not None:
            db_write.close()
