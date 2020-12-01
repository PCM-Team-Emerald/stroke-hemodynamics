# Imports
import json, os

import numpy as np
import pandas as pd
pd.set_option('display.max_columns',None)
import sqlite3

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
    'Dx':           False,
    'Feeding':      False,
    'Flowsheet':    True,
    'IO_Flowsheet': False,
    'Labs':         False,
    'LDA':          False,
    'MAR':          False,
    'Med':          False,
    'Hx':           False,
    'Problem_List': False
}

##### Flowsheet #####
if to_run['Flowsheet']:
    # Read file
    dat = pd.read_sql('SELECT * FROM FLOWSHEET', raw_conn, parse_dates=True, index_col='index')

    # Read manual coding
    mc = pd.read_csv('S:/Dehydration_stroke/Team Emerald/Working Data/Preprocessed/Working/Manual_Coding/Annotated/Flowsheet_names.csv')

    dat = pd.merge(dat,mc,how='left',on='flowsheet_row_name')
    dat = dat[dat['Name'].notnull()]
    dat_pivoted = pd.pivot_table(dat,index=['mrn','csn','recorded_datetime'],
                                 columns='Name',values='value',aggfunc='first')
    # Processing
    to_keep = ['bp','pulse_ox','pulse','temp']

    dat_pivoted = dat_pivoted[to_keep]
    dat_pivoted.dropna(how='all',inplace=True)
    bps = dat_pivoted['bp'].str.split('/', n=1, expand=True)
    dat_pivoted['sbp'] = bps[0]
    dat_pivoted['dbp'] = bps[1]
    dat_pivoted.drop(columns=['bp'], inplace=True)
    for c in dat_pivoted.columns:
        dat_pivoted[c] = dat_pivoted[c].astype(float)

    # Convert temp F to C
    dat_pivoted['temp'] = dat_pivoted['temp'].apply(lambda x: x if x < 70 else (x-32)*5/9)

    # Remove extreme values
    dat_pivoted['temp'] = dat_pivoted['temp'].apply(lambda x: np.NaN if x < 30 or x > 45 else x)
    dat_pivoted['pulse_ox'] = dat_pivoted['pulse_ox'].apply(lambda x: np.NaN if x > 100 else x)
    dat_pivoted['pulse'] = dat_pivoted['pulse'].apply(lambda x: np.NaN if x < 30 or x > 300 else x)
    dat_pivoted['sbp'] = dat_pivoted['sbp'].apply(lambda x: np.NaN if x < 50 or x > 300 else x)
    dat_pivoted['dbp'] = dat_pivoted['dbp'].apply(lambda x: np.NaN if x < 10 or x > 200 else x)

    # Dist plots
    import matplotlib.pyplot as plt
    for c in dat_pivoted.columns:
        plt.hist(dat_pivoted[c])
        plt.title(c)
        plt.show()

    # Melt
    dat_melted = dat_pivoted.reset_index(drop=False).melt(id_vars=['mrn','csn','recorded_datetime'])

    # Write to processed
    dat_melted.to_sql('FLOWSHEET', processed_conn, if_exists='replace', index=False)