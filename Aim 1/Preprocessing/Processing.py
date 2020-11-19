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
    'ADT':          True,
    'Demographics': True,
    'Dx':           True,
    'Feeding':      False,
    'Flowsheet':    False,
    'IO_Flowsheet': False,
    'Labs':         False,
    'LDA':          False,
    'MAR':          False,
    'Med':          False,
    'Hx':           False,
    'Problem_List': False
}

##### ADT #####
if to_run['ADT']:
    # Read file
    df = pd.read_sql_query("SELECT * FROM ADT", raw_conn, parse_dates=True)

    # Create new lable for mrn, csn pair
    df['mrn_csn_pair'] = list(zip(df.mrn,df.csn))
    df = df.sort_values('mrn_csn_pair')
    df['mrn_csn_pair'] = df['mrn_csn_pair'].astype(str)

    # # Calculate difference in admission and discharge time
    # df['in']=pd.to_datetime(df['in'])
    # df['out']=pd.to_datetime(df['out'])
    # df['total_time']=df['out']-df['in']

    # Convert unique unit names to dummie variables
    cols = df['unit'].unique()
    df[cols] = pd.get_dummies(df['unit'], drop_first=False)

    # Drop unused columns
    df = df.drop(['index','mrn','csn','unit','in','out'], axis=1)

    # Identify unique pairs
    unique_pairs = df['mrn_csn_pair'].unique()

    # Iterate through unique pairs to gather units for each individual patient
    d = pd.DataFrame(columns=cols)
    for patient in unique_pairs:
        series = df[df['mrn_csn_pair'] == patient][cols].sum()
        d = d.append(series, ignore_index=True)

    # Insert identifier column to new dataframe
    d.insert(0, 'mrn_csn_pair', unique_pairs)

    # Write to processed
    d.to_sql('ADT', processed_conn, if_exists='replace')

##### Demographics #####
if to_run['Demographics']:
    # Read file
    df_d = pd.read_sql('SELECT * FROM DEMOGRAPHICS', raw_conn, parse_dates=True)
    print(dat.head())

    # Processing

    # Create single label combining mrn and csn
    df_d['mrn_csn_pair'] = list(zip(df_d.mrn,df_d.csn))
    df_d['mrn_csn_pair'] = df_d['mrn_csn_pair'].astype(str)

    # Calculate difference in admission and discharge time
    df_d['admission_datetime']=pd.to_datetime(df_d['admission_datetime'])
    df_d['discharge_datetime']=pd.to_datetime(df_d['discharge_datetime'])
    df_d['time_in_hospital_minutes']=df_d['discharge_datetime']-df_d['admission_datetime']

    # Function to convert time delta into minutes
    def timedelta_to_min(time):
        days = time.days
        minutes = time.seconds//60
        minutes += days*60*24
        return minutes

    # Convert the total hospital stay to minutes
    df_d['time_in_hospital_minutes']=df_d['time_in_hospital_minutes'].apply(timedelta_to_min)

    # Create dummy fariable for gender
    df_d['male'] = pd.get_dummies(df_d['gender'], drop_first=True)

    # Group all patients with ages over 90
    df_d['age'] = df_d['age'].apply(lambda x:'90' if x == '90+' else x)
    df_d['age'] = pd.to_numeric(df_d['age'])

    # Function to extract CCI score from string
    def cci(s):
        if str(s) == 'None':
            return None
        val = s.split(':')
        return int(val[-1])

    # Create new feature to return CCI score
    df_d['cci'] = df_d['charlson_comorbidity_index'].apply(cci)

    # Drop all unused features
    df_d = df_d.drop(['admission_datetime', 'discharge_datetime', 'ed_arrival_datetime', 'index',
                    'charlson_comorbidity_index', 'male', 'mrn', 'csn'], axis=1)

    print(df_d.head())

    # Write to processed
    df_d.to_sql('DEMOGRAPHICS', processed_conn, if_exists='replace')

##### Dx #####
if to_run['Dx']:
    df = pd.read_sql_query("SELECT * FROM DX", raw_conn, parse_dates=True) 

    # Create new lable for mrn, csn pair
    df['mrn_csn_pair'] = list(zip(df.mrn,df.csn))
    df = df.sort_values('mrn_csn_pair')
    df['mrn_csn_pair'] = df['mrn_csn_pair'].astype(str)

    # Load in csv data categorizing diagnoses into stroke types
    path = "S:\Dehydration_stroke\Team Emerald\Working Data\Preprocessed\Working\Dx_class_annotated.csv"
    df_stroke_classes = pd.read_csv(path)

    # Create dictionary to reference stroke types
    a = pd.Series(df_stroke_classes['diagnosis'])
    b = pd.Series(df_stroke_classes['class'])
    stroke_dict = dict(zip(a.values,b.values))

    # Only include patients with a primary or an ED diagnosis
    df = df[(df['primary_dx'] == 'Y') | (df['ed_dx'] == 'Y')]

    # Create new stroke class column categorizing diagnosis based on dictionary
    df['stroke_class'] = df['diagnosis'].apply(lambda x: stroke_dict[x])

    # Drop unnecessary columns
    df = df.reset_index()

    df = df.drop(['level_0', 'mrn', 'csn', 'index', 'icd9', 'icd10', 
                'diagnosis', 'primary_dx', 'ed_dx'], axis=1)

    print(df.head())

    # Write to processed
    df.to_sql('DX', processed_conn, if_exists='replace')

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
