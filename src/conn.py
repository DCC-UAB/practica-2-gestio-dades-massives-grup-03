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

        super().add_argument("--datasetName", type=str, default="Iris", help="Name of the imported dataset.")
        super().add_argument("--fileName", type=str, default="dataset/iris.data.txt", help="file where data is stored.")

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

    def parse(self):
        return super().parse_args()

