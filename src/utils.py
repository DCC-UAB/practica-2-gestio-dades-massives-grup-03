# -*- coding: utf-8 -*-
u'''
Created on Jun 12, 2018

@author: Oriol Ramos Terrades
@email: oriolrt@cvc.uab.cat
'''


import numpy as np
from collections import namedtuple
import pandas as pd
import random
from scipy.spatial import distance as dist

import numpy as np
import scipy.sparse as sp
import sys
import logging



def readVectorDataFile(fileName, label_pos=-1, **kwargs):
    """
    Anem a llegir el contingut dels fitxers i els guardem en una estructura definida per fila amb camps: id, label, feature
    """
    #fila = namedtuple("fila", "id label feature ")
    df = pd.DataFrame(columns=['id', 'features', 'class'])
    f = open(fileName)
    line = f.readline().rstrip("\n").rstrip("\r")

    id = 0
    ids = {}
    records = []
    while line:
        stringFeatures = line.split(",")
        label = stringFeatures[label_pos]
        stringFeatures.remove(label)
        features = [float(x) for x in stringFeatures]

        if len(features) > 0:
            records.append({'id': id, 'features': features, 'class': label})

            # df = df.append({'id': id, 'features': features, 'class': label}, ignore_index=True)
            if label in ids:
                ids[label].append( id )
            else:
                ids[label] = [id ]
            id = id + 1

        line = f.readline().rstrip("\n").rstrip("\r")
    f.close()
    df = df.from_records(records)
    df = df.set_index('id')
    return df, ids


def loadData(dbms, nameDataset, LoadFeatures=False, **params):
    '''
    Description: load data from the DBMS given the parameters passed by argument

    :param dbms: object with the connection to the DB
    :param nameDataset:
    :param params:
    :return: list of rows with the id of the image and the feature vector as numpy array
    '''


    try:
        features, classIds = dbms.loadData(nameDataset)

        assert len(features) > 0 , "Les caracteristiques de {} No s'han carregat correctament a la BD".format(
              nameDataset)

    except ValueError as err:
        print('Algun valor retornat de la Base de dades no acaba de ser correcte. Revisa el codi')
        raise
    except IndexError as err:
        print('Dades no carregades de la BD. Revisa que les dades existeixen i que tens ben implementat el mètode')
        print("L'execució del programa no pot continuar fins que es solucioni el problema")
        raise


    return features, classIds


def generateHashCode( method, **params ):

    codeList = params.copy()
    codeList["method"] = method

    code = hash(frozenset(codeList.items()))

    return code


