# Imports
import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import matplotlib.pyplot as plt
import json
import sqlite3
import os

# Config file
with open('cfg.json') as json_file:
    cfg = json.load(json_file)

# Setup SQLite DB
processed_url = os.path.join(cfg['WORKING_DATA_DIR'], 'Processed/Processed.db')
processed_conn = sqlite3.connect(processed_url)

raw_data = pd.read_sql('SELECT * FROM FLOWSHEET UNION ALL SELECT * FROM NEURO', processed_conn, parse_dates=['recorded_datetime'])
raw_data_demographics = pd.read_sql('SELECT DEMOGRAPHICS.mrn_csn_pair AS mrn_csn_pair, admission_datetime, discharge_datetime, time_in_hospital_minutes FROM DEMOGRAPHICS INNER JOIN mrn_csn_pairs mcp on DEMOGRAPHICS.mrn_csn_pair = mcp.mrn_csn_pair', processed_conn, parse_dates=['discharge_datetime'])

dispo = pd.read_sql('SELECT DISPO.mrn_csn_pair AS mrn_csn_pair, ' \
                    'DISPO.dispo_alf , ' \
                    'DISPO.dispo_deceased, ' \
                    'DISPO.dispo_home, ' \
                    'DISPO.dispo_home_care, ' \
                    'DISPO.dispo_hospice, ' \
                    'DISPO.dispo_hospital, ' \
                    'DISPO.dispo_rehab, ' \
                    'DISPO.dispo_snf ' \
                    'FROM DISPO INNER JOIN mrn_csn_pairs mcp on DISPO.mrn_csn_pair = mcp.mrn_csn_pair', processed_conn)

discharge = raw_data_demographics[['mrn_csn_pair', 'discharge_datetime']]

dat = pd.merge(raw_data, discharge, how='inner', on='mrn_csn_pair').dropna()
dat = pd.merge(dat, dispo, how='inner', on='mrn_csn_pair').dropna()

dat['time_to_discharge'] = (dat['discharge_datetime'] - dat['recorded_datetime']) / np.timedelta64(1,'m')
dat = dat[dat['time_to_discharge'] <= 24*60]

# For each mrn, csn
# Secondary outcomes
indices = []
mrn_csns = dat['mrn_csn_pair'].unique()
for m in mrn_csns:
    for v in ['ampac_mobility_tscore',
              'hlm',
              'ampac_activity_tscore',
              'glasgow_score',
              'glasgow_eye_opening',
              'glasgow_motor_response',
              'glasgow_verbal_response',
              'consciousness',
              'orientation']:
        dat_sub = dat[(dat['mrn_csn_pair'] == m) & (dat['Name'] == v)]
        if dat_sub.shape[0] > 0:
            idx = dat_sub['time_to_discharge'].idxmin()
            indices.append(idx)

dat_last = dat.loc[indices]

dat = dat_last
dat['value'] = dat['value'].astype(float)
dat = pd.pivot_table(dat, index='mrn_csn_pair', columns='Name', values='value')

print(dat.count())

dat.to_sql('secondary_outcomes', processed_conn, if_exists='replace')