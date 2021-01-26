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
        try:
            df[c] = df[c].apply(lambda x: parser.parse(x).isoformat().replace('T',' '))
        except TypeError:
            pass
    return df

to_run = {
    'ADT':          False,
    'Demographics': False,
    'Dx':           False,
    'Feeding':      False,
    'Flowsheet':    False,
    'IO_Flowsheet': False,
    'Labs':         False,
    'LDA':          False,
    'MAR':          False,
    'Med':          False,
    'Hx':           False,
    'Problem_List': False,
    'Neuro':        False,
    'Dispo':        True
}

##### ADT #####
if to_run['ADT']:
    # Read file
    dat = pd.read_table(data_dir + '/ADT_Transfer.txt')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn','csn','unit','in','out']

    # Convert datetimes to ISO
    datetime_cols = ['in','out']
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'ADT': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('ADT', conn, if_exists='replace')

##### Demographics #####
if to_run['Demographics']:
    # Read file
    dat = pd.read_table(data_dir + '/demographics.txt', sep='|', encoding='ISO-8859-1')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'admission_datetime', 'discharge_datetime', 'age', 'gender', 'race','ed_arrival_datetime', 'admit_service', 'admit_department', 'discharge_department','charlson_comorbidity_index']

    # Convert datetimes to ISO
    datetime_cols = ['admission_datetime','discharge_datetime','ed_arrival_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'Demographics': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('DEMOGRAPHICS', conn, if_exists='replace')

##### Dx #####
if to_run['Dx']:
    # Read file
    dat = pd.read_table(data_dir + '/Dx.txt', sep='|', encoding='ISO-8859-1')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'icd9', 'icd10', 'diagnosis', 'primary_dx', 'ed_dx']

    # Convert datetimes to ISO
    datetime_cols = []
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'Dx': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('DX', conn, if_exists='replace')

##### Feeding #####
if to_run['Feeding']:
    # Read file
    dat = pd.read_table(data_dir + '/Feeding.txt')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'proc_name', 'display_name', 'order_datetime', 'RESULT_TIME', 'result_datetime', 'proc_start_datetime', 'proc_stop_datetime', 'order_id']
    dat = dat[['mrn', 'csn', 'proc_name', 'display_name', 'order_datetime', 'result_datetime', 'proc_start_datetime', 'proc_stop_datetime', 'order_id']]

    # Convert datetimes to ISO
    datetime_cols = ['order_datetime','result_datetime','proc_start_datetime','proc_stop_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'Feeding': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('FEEDING', conn, if_exists='replace')

##### Flowsheet #####
if to_run['Flowsheet']:
    # Read file
    dat = pd.read_table(data_dir + '/flowsheet_11022020.txt',sep='|')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn','recorded_RAW', 'recorded_datetime', 'value', 'template_name', 'flowsheet_row_name']
    dat.drop(columns=['recorded_RAW'])

    # Convert datetimes to ISO
    datetime_cols = ['recorded_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'Flowsheet': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('FLOWSHEET', conn, if_exists='replace')

##### IO Flowsheet #####
if to_run['IO_Flowsheet']:
    # Read file
    dat = pd.read_table(data_dir + '/IO_flowsheet.txt', sep='|')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'recorded_datetime', 'value', 'template_name', 'flowsheet_row_name']

    # Convert datetimes to ISO
    datetime_cols = ['recorded_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'IO_Flowsheet': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('IO_FLOWSHEET', conn, if_exists='replace')

##### Labs #####
if to_run['Labs']:
    # Read file
    dat = pd.read_table(data_dir + '/Labs_1007.txt', sep='|')

    print(dat.columns)

    # Drop unnecesary cols, rename
    dat.columns = ['mrn','csn','RESULT_DATE_TIME','result_datetime',
                   'units','value_numeric','value_text','lab_result_comment','order_id','order_description',
                   'order_display_name','component_name','component_base_name']
    dat = dat[['mrn','csn','result_datetime',
                   'units','value_numeric','value_text','lab_result_comment','order_id','order_description',
                   'order_display_name','component_name','component_base_name']]

    # Convert datetimes to ISO
    datetime_cols = ['result_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'Labs': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('LABS', conn, if_exists='replace')

##### LDA #####
if to_run['LDA']:
    # Read file
    dat = pd.read_table(data_dir + '/LDA.txt', encoding='ISO-8859-1')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'placed_datetime', 'removed_datetime', 'template_name', 'lda_name', 'lda_measurements_and_assessments']

    # Convert datetimes to ISO
    datetime_cols = ['placed_datetime','removed_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'Labs': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('LDA', conn, if_exists='replace')

##### MAR #####
if to_run['MAR']:
    # Read file
    dat = pd.read_table(data_dir + '/MAR.txt')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'medication_name', 'ordered_datetime','order_id','med_admin_start_datetime','med_admin_end_datetime','dosage']

    # Convert datetimes to ISO
    datetime_cols = ['ordered_datetime','med_admin_start_datetime','med_admin_end_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'MAR': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('MAR', conn, if_exists='replace')

##### Med #####
if to_run['Med']:
    # Read file
    dat = pd.read_table(data_dir + '/Med.txt')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'medication_name', 'route', 'dosage', 'max_dose', 'min_discrete_dose', 'order_id', 'ordered_datetime']

    # Convert datetimes to ISO
    datetime_cols = ['ordered_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'Med': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('MED', conn, if_exists='replace')

##### Hx #####
if to_run['Hx']:
    # Read file
    dat = pd.read_table(data_dir + '/Med_HX.txt')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'description', 'resolved_datetime', 'icd10']

    # Convert datetimes to ISO
    datetime_cols = ['resolved_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'Hx': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('HX', conn, if_exists='replace')

##### Problem List #####
if to_run['Problem_List']:
    # Read file
    dat = pd.read_table(data_dir + '/ProblemList.txt', encoding='ISO-8859-1')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'description', 'noted_datetime', 'resolved_datetime', 'icd10']

    # Convert datetimes to ISO
    datetime_cols = ['noted_datetime', 'resolved_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'Problem_List': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('PROBLEM_LIST', conn, if_exists='replace')

##### Neuro flowsheet #####
if to_run['Neuro']:
    # Read file
    dat = pd.read_table(data_dir + '/Neurological_flowsheet.txt', sep='|')

    # Drop unnecesary cols, rename
    # Col names are wrong, overriding here:
    dat.columns = ['mrn', 'csn', 'recorded_datetime_RAW',
       'recorded_datetime', 'value',
       'template_name', 'flowsheet_row_name']
    dat.drop(columns=['recorded_datetime_RAW'], inplace=True)

    # Convert datetimes to ISO
    datetime_cols = ['recorded_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Add to data dict
    data_dict.update({'Neuro': dat.dtypes.to_dict()})

    # Save to db
    print('Neuro',dat.shape)
    dat.to_sql('NEURO', conn, if_exists='replace')

##### Dispo #####
if to_run['Dispo']:
    # Read file
    dat = pd.read_table(data_dir + '/Disch_disposition.txt', sep='|')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'deceased', 'disposition']

    # Add to data dict
    data_dict.update({'Dispo': dat.dtypes.to_dict()})

    # Save to db
    dat.to_sql('DISPO', conn, if_exists='replace')
