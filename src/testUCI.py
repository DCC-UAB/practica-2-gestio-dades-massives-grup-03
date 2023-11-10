import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from datetime import datetime
import socket
from itertools import product

import oracledb
from GABDConnect.oracleConnection import oracleConnection as orcl


if __name__ == "__main__":

    host = socket.gethostname()
    numIterations = 2
    Algorithms = {
        'Support Vector Machines': SVC,
        'K nearest Neighbor': KNN,
        'Random Forest': RFC
    }
    params = { 'SVC' : {'kernel': ("linear", "rbf", "poly"),
                        'gamma': [1, 5, 10, 20]},
               'KNeighborsClassifier' : { 'n_neighbors' : [3, 5, 10, 15]},
               'RandomForestClassifier' : {'max_depth': [2, 4, 10, None],
                        'criterion': ['gini', 'entropy', 'log_loss']}
               }

    iris = datasets.load_iris()
    Xo = iris.data
    yo = iris.target

    #Xo = Xo[yo != 0, :2]
    #yo = yo[yo != 0]

    #digits = datasets.load_digits()
    #X = digits.data
    #Y = digits.target


    n_sample = len(Xo)

    np.random.seed(0)

    current_time = datetime.now().strftime("%d/%m/%Y, %H:%M")

    for k in Algorithms:
        classificador = Algorithms[k].__name__
        c_params = params[classificador].keys()
        for idx,values in enumerate(product(*params[classificador].values())):
            cp = {k: v for k, v in zip(c_params, values) if v is not None}
            clf = Algorithms[k](**cp)
            for i in range(numIterations):
                order = np.random.permutation(n_sample)
                X = Xo[order]
                y = yo[order].astype(float)

                X_train = X[: int(0.9 * n_sample)]
                y_train = y[: int(0.9 * n_sample)]
                X_test = X[int(0.9 * n_sample) :]
                y_test = y[int(0.9 * n_sample) :]


                # Si no executem el script a la màquina main de la pràctica visualitzem els resultats

                # fit the model
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)

                f1_s = f1_score(y_test, y_pred, average='macro')
                acc = accuracy_score(y_test, y_pred)
                print(
                    "Classificador: {}, Iteracio: {}, paràmetres: {}, time: {}, f-score: {}, accuracy: {}".format(k, i, cp, current_time, f1_s,
                                                                                                acc))