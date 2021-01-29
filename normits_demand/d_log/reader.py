# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:27:18 2020

Script to hit the TfN GIS database and download all available data from 2020
D-Log. Supplies both residential and non residential data

@author: genie
"""

# BACKLOG: Expose d_log reader to normits_demand as a function
#  labels: EFS, demand merge

import pyodbc

import pandas as pd

_postgres_driver = '{PostgreSQL Unicode}'
_server = 'tfn-gis-server.postgres.database.azure.com'
_database = 'gis_db'
_u = 'efs_reader@tfn-gis-server'
_pwd = 'dev_read_42'
_port = '5432'

_conn_string = ('driver=' + _postgres_driver +
                ';server=' + _server +
                ';database=' + _database +
                ';uid=' + _u +
                ';pwd=' + _pwd +
                ';port=' + _port +
                ';sslmode=require'
                )

def connect_to_tfn_gis(conn_string):

    """
    Connect to gis given any string.
    
    Parameters
    ------
    conn_string:
        Connection string in globals
    Returns
    ------
    conn:
        Active connection
    cursor:
        Cursor connection pointer to do queries
    """

    conn = pyodbc.connect(conn_string)
    cursor = conn.cursor()
    
    return(conn, cursor)

def unpack_query(cursor):

    """
    Parameters
    ------
    cursor:
        Cursor carrying a query to unpack.
        Will unpack nothing if it's there and return nothing.
    Returns
    ------
    out_dat:
        Requested data in pandas.DataFrame format.
    """

    data = cursor.fetchall()

    # Unpack coldata
    cols = [column[0] for column in cursor.description]

    contents = []
    for row in data:
        contents.append([x for x in row])

    out_dat = pd.DataFrame(contents)
    out_dat.columns = cols
    
    return(out_dat)

def get_dlog_data(cursor):

    """
    Parameters
    ------
    cursor:
        Cursor for d-log connection.
    Returns
    ------
    res_data:
        Table of residential data for d-log.
    non_res_data:
        Table of non-residential data for d-log.
    """

    # Hit residential table
    cursor.execute('select * from gis_data.dlog_residential')
    
    res_data = unpack_query(cursor)

    # Hit non residential table
    cursor.execute('select * from gis_data.dlog_non_residential')
    non_res_data = unpack_query(cursor)

    return(res_data, non_res_data)


if __name__ == '__main__':

    conn, gis = connect_to_tfn_gis(_conn_string)
    
    dlog_res, dlog_non_res = get_dlog_data(gis) 
    
    conn.close()
