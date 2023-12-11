from argparse import ArgumentParser
import oracledb
from GABDConnect.oracleConnection import oracleConnection as orcl
from hashlib import sha256
import logging
import json


class ImportOptions(ArgumentParser):

    def __init__(self, user_fallback:str, password_fallback:str):
        logging.info("fallback: " + user_fallback)
        logging.info("fallback: " + password_fallback)
        super().__init__(
            description="This script insert data from the UCI repositori."
        )

        super().add_argument("datasetName", type=str, default=None, help="Name of the imported dataset.")
        super().add_argument("fileName", type=str, default=None, help="file where data is stored.")

        super().add_argument("-C", "--columnClass", type=int, default=-1,
                             help="index to denote the column position of class label.")

        super().add_argument("--user", type=str, default=user_fallback,
                             help="string with the user used to connect to the Oracle DB.")
        super().add_argument("--passwd", type=str, default=password_fallback,
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

        super().add_argument("--alg", choices=['SVC', 'KNN', 'RFC'], type=str, default="SVC", help="Algo")
        super().add_argument("--kernel", choices=['linear', 'rbf', 'poly'], type=str, default="linear", help="Only valid for SVC")
        super().add_argument("--gamma", choices=[1, 5, 10, 20], type=int, default=1, help="Only valid for SVC")
        super().add_argument("--n_neighbors", choices=[3, 5, 10, 15], type=int, default=3, help="Only valid for KNN")
        super().add_argument("--max_depth", choices=[2, 4, 10, None], type=int, default=None, help="Only valid for RFC")
        super().add_argument("--criterion", choices=['gini', 'entropy', 'log_loss'], type=str, default="gini", help="Only valid for RFC")

    def parse(self):
        return super().parse_args()


def get_conn(user_fallback: str = "GestorUCI", password_fallback: str = "33"):
    args = ImportOptions(user_fallback, password_fallback).parse()

    # Inicialitzem el diccionari amb les dades de connexió SSH per fer el tunel
    ssh_server = {'ssh': args.ssh_tunnel, 'user': args.ssh_user,
                  'pwd': args.ssh_password, 'port': args.ssh_port} if args.ssh_tunnel is not None else None

    # Cridem el constructor i obrim la connexió
    db = orcl(user=args.user,
              passwd=args.passwd,
              hostname=args.hostname,
              port=args.port,
              serviceName=args.serviceName,
              ssh=ssh_server)

    db.open()

    # Comprovem  que la connexió a la base de dades es correcte
    if db.testConnection():
        logging.info("La connexió a {} funciona correctament.".format(args.hostname))
    else:
        logging.error("La connexió a {} **NO** funciona correctament.".format(args.hostname))
        raise SystemExit("error while connecting to the db")

    return db, args


def hash_items(cp) -> str:
    cp_hashable = tuple(sorted(cp.items()))
    return sha256(json.dumps(cp_hashable).encode('utf-8')).hexdigest()