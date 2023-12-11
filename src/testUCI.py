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
from conn import get_conn

import oracledb
from GABDConnect.oracleConnection import oracleConnection as orcl


def insert_results(db_conn, nameDataset, classifier, parametres, iteration, time, fi_score, accuracy):
    try:
        cur = db_conn.cursor()
        cur.prepare("INSERT INTO GestorUCI.REPETICIO (NAMEDATASET, NUM) VALUES (:1, :2)")
        cur.execute(None, [nameDataset, iteration])
        cur.prepare(
            "INSERT INTO GestorUCI.EXPERIMENT (NAMEDATASET, NOMCURT, PAR_HASH, REP_NUM, DATA, ACCURACY, F_SCORE) VALUES (:1, :2, :3, :4, :5, :6, :7);")
        cp_hashable = tuple(sorted(parametres.items()))
        hash_valors = hash(cp_hashable)
        cur.execute(None, [nameDataset, classifier, hash_valors, iteration, time, accuracy, fi_score])
        db_conn.commit()
        print("Resultados insertados correctamente.")

    except Exception as e:
        print(f"Error al insert resultados: {e}")


if __name__ == "__main__":

    db_write, args = get_conn("DevUCI")
    numIterations = 2
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
        numRepetitions = 0

        for k in Algorithms:
            # todo: only one algo
            f_score = np.zeros(shape=(numIterations,), dtype=np.float64)
            accuracy = np.zeros(shape=(numIterations,), dtype=np.float64)
            classificador = Algorithms[k].__name__
            c_params = params[classificador].keys()
            for idx, values in enumerate(product(*params[classificador].values())):
                cp = {k: v for k, v in zip(c_params, values) if v is not None}
                clf = Algorithms[k](**cp)
                for i in range(numIterations):
                    order = np.random.permutation(n_sample)
                    X = Xo[order]
                    y = Yo[order]

                    X_train = X[: int(0.9 * n_sample)]
                    y_train = y[: int(0.9 * n_sample)]
                    y_test = y[int(0.9 * n_sample):]
                    X_test = X[int(0.9 * n_sample):]

                    # Si no executem el script a la màquina main de la pràctica visualitzem els resultats

                    # fit the model
                    clf.fit(X_train, y_train)

                    y_pred = clf.predict(X_test)

                    f1_s = f1_score(y_test, y_pred, average='macro')
                    acc = accuracy_score(y_test, y_pred)
                    f_score[i] = f1_s
                    accuracy[i] = acc
                    print(
                        "Classificador: {}, Iteracio: {}, paràmetres: {}, time: {}, f-score: {}, accuracy: {}".format(k,
                                                                                                                      i,
                                                                                                                      cp,
                                                                                                                      current_time,
                                                                                                                      f1_s,
                                                                                                                      acc))

            print("f-score: {}, accuracy: {}".format(np.average(f_score), np.average(accuracy)))


    finally:  # lo he canviado putos
        if db_write is not None:
            db_write.close()
