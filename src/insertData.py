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

    super().add_argument("--user", type=str, default=None, help="string with the user used to connect to the Oracle DB.")
    super().add_argument("--passwd", type=str, default=None,
                         help="string with the password used to connect to the Oracle DB.")
    super().add_argument("--hostname", type=str, default="localhost",
                         help="name of the Oracle Server you want to connect")
    super().add_argument("--port", type=str, default="1521", help="Oracle Port connection.")
    super().add_argument("--serviceName", type=str, default="orcl", help="Oracle Service Name")

    super().add_argument("--ssh_tunnel", type=str, default=None,help="name of the Server you want to create a ssh tunnel")
    super().add_argument("--ssh_user", type=str, default="student",  help="SSH user")
    super().add_argument("--ssh_password", type=str, default=None, help="SSH password")
    super().add_argument("--ssh_port", type=str, default="22", help="SSH port")



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

  df,ids = readVectorDataFile(fileName,  label_pos=label_pos)

  cur = dbConn.cursor()
  # Exemple de manipulacio de JSON
  info = {'description':'Informació del dataset Iris'}
  var = cur.var(oracledb.DB_TYPE_JSON)
  var.setvalue(0, info)
  # Exemple de Blob
  blobFeatures = cur.var(oracledb.BLOB)
  a = np.array([0, 1, 2, 3])
  blobFeatures.setvalue(0, a.tobytes())
  #TODO: Implementeu el codi necessari per fer la inserció de les dades a Oracle

  #FI TODO

  try:
    dbConn.commit()
    return True
  except:
    return False

if __name__ == '__main__':
  # read commandline arguments, first
  args = ImportOptions().parse()

  # Inicialitzem el diccionari amb les dades de connexió SSH per fer el tunel
  ssh_server = {'ssh': args.ssh_tunnel, 'user': args.ssh_user,
                'pwd': args.ssh_password, 'port': args.ssh_port} if args.ssh_tunnel is not None else None

  # Cridem el constructor i obrim la connexió
  db = orcl(user=args.user, passwd=args.passwd, hostname=args.hostname, port=args.port,
            serviceName=args.serviceName,ssh=ssh_server)

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
