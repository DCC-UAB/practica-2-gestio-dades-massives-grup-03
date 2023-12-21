# -*- coding: utf-8 -*-
u'''

This script insert data into a dataset DB in Oracle. Run: Python3 insertData.py --help for usage examples


Created on 3/10/2022

@author: Oriol Ramos Terrades (oriol.ramos@cuab.cat)
@Institution: Computer Science Dept. - Universitat Autònoma de Barcelona
'''

import sys
import numpy as np

import logging
from argparse import ArgumentParser

import oracledb

from itertools import product
from src.conn import get_conn
from utils import readVectorDataFile
from GABDConnect.oracleConnection import oracleConnection as orcl




if __name__ == '__main__':
    db, args = get_conn()

    if args.datasetName:
        res = insertVectorDataset(db, args.datasetName, args.fileName, args.columnClass)

        if res:
            logging.warning("Dades carregades correctament.")
        else:
            logging.warning("Les Dades no s'han inserit correctament.")

    db.close()
    sys.exit(0)
