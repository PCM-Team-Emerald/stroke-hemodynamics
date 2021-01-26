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

cols_to_get = [
    ('SELECT icd9, icd10, diagnosis FROM DX WHERE primary_dx = "Y" OR ed_dx = "Y"','Dx_class'),
    ('SELECT flowsheet_row_name FROM FLOWSHEET','Flowsheet_names'),
    ('SELECT unit FROM ADT', 'Units'),
    ('SELECT admit_service FROM DEMOGRAPHICS','Services'),
    ('SELECT admit_department FROM DEMOGRAPHICS UNION ALL SELECT discharge_department FROM DEMOGRAPHICS', 'Departments'),
    ('SELECT proc_name,display_name FROM FEEDING', 'Feeding'),
    ('SELECT icd10, description FROM HX','Hx'),
    ('SELECT flowsheet_row_name FROM IO_FLOWSHEET','IO_Flowsheet_names'),
    # Labs ('','Lab_names'),
    ('SELECT lda_name from LDA','LDA_names'),
    ('SELECT medication_name FROM MAR UNION SELECT DISTINCT medication_name FROM MED','Med_names'),
    ('SELECT flowsheet_row_name FROM NEURO', 'neuro_names'),
    ('SELECT icd10, description FROM PROBLEM_LIST','Problem_List')
]

cols_to_get = [
    ('SELECT disposition FROM DISPO','Dispo_names'),
    ('SELECT flowsheet_row_name FROM NEURO', 'neuro_names')
]

for c in cols_to_get:
    dat = pd.read_sql(c[0], conn)
    dat.value_counts().to_csv(cfg['WORKING_DATA_DIR'] + '/Preprocessed/Working/Manual_Coding/Unannotated/'+c[1]+'.csv')