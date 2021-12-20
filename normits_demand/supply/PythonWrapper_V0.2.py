# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:24:29 2021

@author: FOUL9363
"""

import psycopg2
from psycopg2 import Error
import random

try:
    # Connect to an existing database
    connection = psycopg2.connect(user="prjt_normits_supply_ed@tfn-gis-server",
                                  password="prjt_normits_supply_ed",
                                  host="10.1.2.6",
                                  port="5432",
                                  database="gis_db",
                                  sslmode='require')

    # Create a cursor to perform database operations
    cursor = connection.cursor()
    # Print PostgreSQL details
    print("PostgreSQL server information")
    print(connection.get_dsn_parameters(), "\n")
    # Executing a SQL query
    selectquery = "SELECT * FROM prjt_normits_supply.skim_master"
    cursor.execute(selectquery) #cursor.execute(myquery)
    #conn.commit() # <- We MUST commit to reflect the inserted data - needed for insert queries??
    
    # Fetch result
    record = cursor.fetchall()
    print("You are connected to - ", record, "\n")

except (Exception, Error) as error:
    print("Error while connecting to PostgreSQL", error)
finally:
    if (connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")