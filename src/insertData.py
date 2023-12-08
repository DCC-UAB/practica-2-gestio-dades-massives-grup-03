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
from utils import readVectorDataFile
from GABDConnect.oracleConnection import oracleConnection as orcl


class ImportOptions(ArgumentParser):

    def __init__(self):
        super().__init__(
            description="This script insert data from the UCI repositori."
        )

        super().add_argument("datasetName", type=str, default=None, help="Name of the imported dataset.")
        super().add_argument("fileName", type=str, default=None, help="file where data is stored.")

        super().add_argument("-C", "--columnClass", type=int, default=-1,
                             help="index to denote the column position of class label.")

        super().add_argument("--user", type=str, default="GestorUCI",
                             help="string with the user used to connect to the Oracle DB.")
        super().add_argument("--passwd", type=str, default="33",
                             help="string with the password used to connect to the Oracle DB.")

        super().add_argument("--hostname", type=str, default="oracle-1.grup03.GABD",
                             help="name of the Oracle Server you want to connect")
        super().add_argument("--port", type=str, default="1521", help="Oracle Port connection.")
        super().add_argument("--serviceName", type=str, default="orcl", help="Oracle Service Name")

        super().add_argument("--ssh_tunnel", type=str, default="dcccluster.uab.cat",
                             help="name of the Server you want to create a ssh tunnel")
        super().add_argument("--ssh_user", type=str, default="student", help="SSH user")
        super().add_argument("--ssh_password", type=str, default="TuLLLh8bCiHj.", help="SSH password")
        super().add_argument("--ssh_port", type=str, default="8195", help="SSH port")

    def parse(self):
        return super().parse_args()


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
    cur = dbConn.cursor()

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

        confirm = "Datos insertados correctamente en el nuevo dataset."
    else:
        confirm = f"El dataset {nameDataset} ya existe, no se realizaron inserciones."

    # Intentar hacer commit y manejar excepciones
    try:
        dbConn.commit()
        print(confirm)
        return True
    except Exception as e:
        print(f"Error al hacer commit: {e}")
        return False


if __name__ == '__main__':
    # read commandline arguments, first
    args = ImportOptions().parse()

    # Inicialitzem el diccionari amb les dades de connexió SSH per fer el tunel
    ssh_server = {'ssh': args.ssh_tunnel, 'user': args.ssh_user,
                  'pwd': args.ssh_password, 'port': args.ssh_port} if args.ssh_tunnel is not None else None

    # Cridem el constructor i obrim la connexió
    db = orcl(user=args.user, passwd=args.passwd, hostname=args.hostname, port=args.port,
              serviceName=args.serviceName, ssh=ssh_server)

    conn = db.open()

    if db.testConnection():
        logging.warning("La connexió a {} funciona correctament.".format(args.hostname))

    if args.datasetName:
        res = insertVectorDataset(db, args.datasetName, args.fileName, args.columnClass)

        if res:
            logging.warning("Dades carregades correctament.")
        else:
            logging.warning("Les Dades no s'han inserit correctament.")

    db.close()
    sys.exit(0)
