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
db_url = cfg['WORKING_DATA_DIR'] + '/Preprocessed/Working/Raw.db'
conn = sqlite3.connect(db_url)

# To collect for each file
data_dict = {}

# Datetime parser function
def datetimeColToISO(df, dt_cols):
    for c in dt_cols:
        df[c] = df[c].apply(lambda x: parser.parse(x).isoformat())
    return df

##### ADT #####
# Read file
dat = pd.read_table(data_dir + '/ADT_Transfer.txt')

# Drop unnecesary cols, rename
dat.columns = ['MRN','CSN','UNIT_NAME','IN_DTTM','OUT_DTTM']

# Add to data dict
data_dict.update({'ADT':dat.dtypes.to_dict()})

# Convert datetimes to ISO
datetime_cols = ['IN_DTTM','OUT_DTTM']
dat = datetimeColToISO(dat, datetime_cols)

# Save to db
dat.to_sql('ADT', conn, if_exists='replace')