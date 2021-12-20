# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 07:55:07 2020

Script to connect to the TfN GIS database. Queries can be sent from a Python
environment. Data will be returned in a pandas Dataframe format.

@author: FraserDouglas
"""

import pyodbc
import random

import pandas as pd


# import pangres as pang
# from sqlalchemy import create_engine


class PostGresConector:
    # CLASS VARIABLES
    user = ""
    password = ""
    postgres_driver = ""
    server = ""
    database = ""
    port = ""
    conn_string = ""
    conn = ""  # pyodbc connection
    cursor = ""  # pyodbc connection cursor

    def __init__(self,
                 user: str,
                 password: str,
                 server: str,
                 database: str,
                 port: str,
                 ):
        """
        Create PostGresConnector class detailing Postgres connection details
        and create cursor object to enable queries to be sent to database.

        Parameters
        ------
        user:
            Username of database user.
        password:
            Password of database user.
        server:
            Server address.
        database:
            Postgres database name.
        port:
            Port.

        Returns
        ------
        print("Connection Successful"):
            Print will show if successfully connected, error will show if
            connection failed.
        """
        self.user = user
        self.password = password
        self.postgres_driver = '{PostgreSQL Unicode}'
        self.server = server
        self.database = database
        self.port = port
        self.conn_string = (
                'driver=' + self.postgres_driver +
                ';server=' + self.server +
                ';database=' + self.database +
                ';uid=' + self.user +
                ';pwd=' + self.password +
                ';port=' + self.port +
                ';sslmode=require'
        )

        self.conn = pyodbc.connect(self.conn_string)
        self.cursor = self.conn.cursor()
        print("Connection Successful")

    def query(self,
              query_file):
        """
        Send query stores in text file to Postgres Database and return result
        in pandas.Dataframe format

        Parameters
        ------
        self:
            Connection details specified in constructor.
        
        query_file:
            File with query stored in it. Example: 

        Returns
        ------
        query_result:
            Requested data in pandas.DataFrame format.
        """
        # query_text = open(
        #     r'' + query_file, 'r')
        # read_text = query_text.read()
        # query_result = pd.read_sql_query(read_text, self.conn)
        query_result = pd.read_sql_query(query_file, self.conn)
        return query_result

    def execute(self,
                query_file):
        """
        Executes query stored in text file to Postgres Database.

        Parameters
        ------
        self:
            Connection details specified in constructor.
        
        query_file:
            File with query stored in it.

        Returns
        ------
        query_result:
            Requested data in pandas.DataFrame format.
        """
        query_text = open(
            r'' + query_file, 'r')
        read_text = query_text.read()
        self.cursor.execute(read_text)
        self.conn.commit()
        return 0

    def update(self,
               query_file,
               dataframe,
               schema,
               staging_table_name=None
               ):
        if staging_table_name is None:
            number = random.randint(100, 1000)
            staging_table_name = "temp_table_" + str(number)

        staging_table_location = schema + "." + staging_table_name
        # configure schema, table_name and engine
        # engine = create_engine("postgresql://" +
        #                        self.user +
        #                        ":" +
        #                        self.password +
        #                        "@" +
        #                        self.server +
        #                        ":" +
        #                        self.port +
        #                        "/" +
        #                        self.database
        #                        )
        # pang.upsert(engine=engine,
        #             df=dataframe,
        #             schema=schema,
        #             table_name=staging_table_name,
        #             if_row_exists="update",
        #             create_schema=False,  # default, creates schema if it does not exist
        #             add_new_columns=True,  # default, adds any columns that are not in the postgres table
        #             adapt_dtype_of_empty_db_columns=True,  # converts data type in postgres for empty columns
        #             # (if we finally have data and it is necessary)
        #             chunksize=10000)  # default, inserts 10000 rows at once

        query_text = open(
            r'' + query_file, 'r')
        read_text = query_text.read()
        self.cursor.execute(read_text)
        self.conn.commit()

        self.cursor.execute("DROP TABLE " + staging_table_location)
        self.conn.commit()
        return 0

    def summary(self,
                query_file):
        """
        Parameters
        ------
        self:
            Connection details specified in constructor.
            
        query_file:
            File with query stored in it.

        Returns
        ------
        query_result:
            Summary Table in pandas.DataFrame format.
        """
        query_text = open(
            r'' + query_file, 'r')
        query_result = pd.read_sql_query(query_text, self.conn)
        return query_result
