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
processed_url = os.path.join(cfg['WORKING_DATA_DIR'], 'Processed/Processed.db')
processed_conn = sqlite3.connect(processed_url)

dat = pd.read_sql_query("SELECT mrn_csn_pair, time_in_hospital_minutes FROM DEMOGRAPHICS", processed_conn, parse_dates=True)

# Only neuro admits
#dat = pd.read_sql_query("SELECT mrn_csn_pair, time_in_hospital_minutes FROM DEMOGRAPHICS WHERE Neuro=1", processed_conn, parse_dates=True)

print('All',dat['mrn_csn_pair'].unique().shape)
# Primary stroke diagnosis
dx = pd.read_sql_query('SELECT * FROM DX', processed_conn, parse_dates=True)
dat = pd.merge(dat, dx, how='inner', on='mrn_csn_pair')
dat = dat[(dat['hemorrhagic_stroke'] == 1) | (dat['ischemic_stroke'] == 1)]
print( dat.shape)
print('Stroke primary dx',dat['mrn_csn_pair'].unique().shape)

# Hospital stay >= 24h
dat = dat[dat['time_in_hospital_minutes'] >= 24*60]
print(dat.shape)
print('LOS >= 24h',dat['mrn_csn_pair'].unique().shape)

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
print('Have vitals', dat['mrn_csn_pair'].unique().shape)



dat['mrn_csn_pair'].to_sql('mrn_csn_pairs', processed_conn, if_exists='replace', index=False)