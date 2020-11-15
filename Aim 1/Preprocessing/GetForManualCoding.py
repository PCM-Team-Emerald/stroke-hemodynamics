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
    ('SELECT DISTINCT icd9, icd10, diagnosis FROM DX WHERE primary_dx = "Y" OR ed_dx = "Y"','Dx_class'),
    ('SELECT DISTINCT flowsheet_row_name FROM FLOWSHEET','Flowsheet_names'),
    ('SELECT DISTINCT unit FROM ADT', 'Units'),
    ('SELECT DISTINCT admit_service FROM DEMOGRAPHICS','Services'),
    ('SELECT DISTINCT admit_department FROM DEMOGRAPHICS UNION SELECT DISTINCT discharge_department FROM DEMOGRAPHICS', 'Departments'),
    ('SELECT DISTINCT proc_name,display_name FROM FEEDING', 'Feeding'),
    ('SELECT DISTINCT icd10, description FROM HX','Hx'),
    ('SELECT DISTINCT flowsheet_row_name FROM IO_FLOWSHEET','IO_Flowsheet_names'),
    # Labs ('','Lab_names'),
    ('SELECT DISTINCT lda_name from LDA','LDA_names'),
    ('SELECT DISTINCT medication_name FROM MAR UNION SELECT DISTINCT medication_name FROM MED','Med_names'),
    ('SELECT DISTINCT flowsheet_row_name FROM NEURO', 'neuro_names'),
    ('SELECT DISTINCT icd10, description FROM PROBLEM_LIST','Problem_List')



]

for c in cols_to_get:
    dat = pd.read_sql(c[0], conn)
    dat.value_counts().to_csv(cfg['WORKING_DATA_DIR'] + '/Preprocessed/Working/Manual_Coding/Unannotated/'+c[1]+'.csv')