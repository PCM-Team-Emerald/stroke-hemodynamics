# Imports
import json, os

import numpy as np
import pandas as pd
pd.set_option('display.max_columns',None)
import sqlite3
import dateutil.parser as parser

# Config file
with open('cfg.json') as json_file:
    cfg = json.load(json_file)

# Setup SQLite DB
data_dir = cfg['WORKING_DATA_DIR'] + '/CCDA_Data'
raw_url = cfg['WORKING_DATA_DIR'] + '/Preprocessed/Working/Raw.db'
processed_url = cfg['WORKING_DATA_DIR'] + '/Preprocessed/Working/Processed.db'
raw_conn = sqlite3.connect(raw_url)
processed_conn = sqlite3.connect(processed_url)

to_run = {
    'ADT':          False,
    'Demographics': False,
    'Dx':           True,
    'Feeding':      True,
    'Flowsheet':    True,
    'IO_Flowsheet': True,
    'Labs':         True,
    'LDA':          True,
    'MAR':          True,
    'Med':          True,
    'Hx':           True,
    'Problem_List': True
}

##### ADT #####
if to_run['ADT']:
    # Read file
    dat = pd.read_sql('SELECT * FROM ADT', raw_conn, parse_dates=True, index_col='index')
    print(dat.head())

    # Processing
    # Unit descriptions
    #TODO: create col for icu vs neuro floor vs other floor?

    # Duration
    dat['duration'] = (dat['out'] - dat['in']) / np.timedelta64(1,'h')

    # Write to processed
    dat.to_sql('ADT', processed_conn, if_exists='replace')

##### Demographics #####
if to_run['Demographics']:
    # Read file
    dat = pd.read_sql('SELECT * FROM DEMOGRAPHICS', raw_conn, parse_dates=True, index_col='index')
    print(dat.head())

    # Processing


    # Charlson
    dat['charlson_comorbidity_index'] = dat['charlson_comorbidity_index'].apply(lambda x: np.nan if type(x)==type(None) else float(x[x.find(':')+1:]))

    print(dat.head())
    # Write to processed
    dat.to_sql('DEMOGRAPHICS', processed_conn, if_exists='replace')

##### Dx #####
if to_run['Dx']:
    pass

##### Feeding #####
if to_run['Feeding']:
    pass

##### Flowsheet #####
if to_run['Flowsheet']:
    # Read file
    dat = pd.read_sql('SELECT * FROM FLOWSHEET', raw_conn, parse_dates=True, index_col='index')
    print(dat.head())

    # Processing
    dat.drop(columns=['template_name'], inplace=True)

    # Pivot
    dat = pd.pivot_table(dat,index=['mrn','csn','recorded_datetime'], column='flowsheet_row_name', value='value')

    print(dat.head())
    # Write to processed
    dat.to_sql('FLOWSHEET', processed_conn, if_exists='replace')

##### IO Flowsheet #####
if to_run['IO_Flowsheet']:
    pass
