import os
import sys

import sqlite3
import pandas as pd

from logger import logging
from exception import CustomException

## connect to sqllite
connection=sqlite3.connect("dailysummary.db")

##create a cursor object to insert record,create table
cursor=connection.cursor()

## create the table
table_info="""
CREATE TABLE daily_report (
    calendar_date DATE NOT NULL,
    tier VARCHAR(100),
    region_s VARCHAR(100),
    platform_s VARCHAR(100),
    Daily_active_users BIGINT,
    Revenue DECIMAL(20,2),      
    CODA_Revenue DECIMAL(20,2),  
    Payers BIGINT,
    Conversions BIGINT,
    Installs BIGINT,
    Register_Installs BIGINT,
    Registers BIGINT,
    Reactivation BIGINT,
    Session_Hours DECIMAL(20,2), 
    Session_Count BIGINT
);
"""
logging.info("Table is created!")

try:
    cursor.execute(table_info)
    logging.info("SQL Cursor is created")

    df = pd.read_csv(os.path.join(os.getcwd(),"Data\datasource.csv"))
    logging.info("Data source file is loaded in dataframe")

    # Insert data into SQLite table
    df.to_sql("daily_report", connection, if_exists="append", index=False)
    logging.info("Data inserted into the database.")
except Exception as e:
    logging.info(CustomException(e,sys))
    raise CustomException(e,sys)


## Commit your changes in the database
connection.commit()
connection.close()


