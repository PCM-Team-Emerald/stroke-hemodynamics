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
processed_url = cfg['WORKING_DATA_DIR'] + '/Preprocessed/Working/Processed.db'
processed_conn = sqlite3.connect(processed_url)

dat = pd.read_sql_query("SELECT mrn_csn_pair, time_in_hospital_minutes FROM DEMOGRAPHICS", processed_conn, parse_dates=True)

# Primary stroke diagnosis
dx = pd.read_sql_query('SELECT * FROM DX', processed_conn, parse_dates=True)
dat = pd.merge(dat, dx, how='inner', on='mrn_csn_pair')
dat = dat[(dat['hemorrhagic'] == 1) | (dat['ischemic'] == 1)]
print(dat.shape)
print(dat['mrn_csn_pair'].unique().shape)

# Hospital stay >= 24h
dat = dat[dat['time_in_hospital_minutes'] >= 24*60]
print(dat.shape)
print(dat['mrn_csn_pair'].unique().shape)

# Pulse, BP, and pulse_ox w/in first 24h
vitals = pd.read_sql_query(
    'SELECT FLOWSHEET.mrn_csn_pair, FLOWSHEET.recorded_datetime, FLOWSHEET.Name, DEMOGRAPHICS.admission_datetime '
    'FROM FLOWSHEET INNER JOIN DEMOGRAPHICS ON FLOWSHEET.mrn_csn_pair = DEMOGRAPHICS.mrn_csn_pair',
    processed_conn, parse_dates=['recorded_datetime', 'admission_datetime'])
vitals['time'] = (vitals['recorded_datetime'] - vitals['admission_datetime']) / np.timedelta64(1,'m')
vitals = vitals[(vitals['time'] >= 0) & (vitals['time'] <= 24*60)]
vitals = pd.pivot_table(vitals, index=['mrn_csn_pair'], columns='Name', values='time')
vitals = vitals[['pulse', 'dbp', 'sbp', 'pulse_ox', 'temp']]
vitals.dropna(how='any', inplace=True)
vitals.reset_index(drop=False, inplace=True)
vitals.groupby('mrn_csn_pair').agg('mean')

dat = pd.merge(dat, vitals, how='inner', on='mrn_csn_pair')
print(dat['mrn_csn_pair'].unique().shape)