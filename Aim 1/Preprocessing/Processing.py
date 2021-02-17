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
    'LDA':          True,
    'MAR':          False,
    'Med':          False,
    'Hx':           True,
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

    # Load in manual coding to categorize unit classes
    path = "S:\\Dehydration_stroke\\Team Emerald\\Working Data\\Preprocessed\\Working\\Manual_Coding\\Annotated\\Units_ann.csv"
    units_classes = pd.read_csv(path)

    # Create dictionary to reference stroke types
    a = pd.Series(units_classes['unit'])
    b = pd.Series(units_classes['Unit'])
    unit_dict = dict(zip(a.values,b.values))

    # Create new feature column categorizing units
    df['Unit'] = df['unit'].apply(lambda x: unit_dict[x])
    df['Unit'] = df['Unit'].apply(lambda x: 'Other' if str(x) == 'nan' else x)

    # Convert unique unit names to dummie variables
    cols = pd.get_dummies(df['Unit'], drop_first=False).columns
    df[cols] = pd.get_dummies(df['Unit'], drop_first=False)

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

    print('\nAdt: \n', d.head())

    # Write to processed
    d.to_sql('ADT', processed_conn, if_exists='replace')





##### Demographics #####
# Function to extract CCI score from string
def cci(s):
    if str(s) == 'None':
        return None
    val = s.split(':')
    return int(val[-1])

# Function to convert time delta into minutes
def timedelta_to_min(time):
    days = time.days
    minutes = time.seconds//60
    minutes += days*60*24
    return minutes

if to_run['Demographics']:
    # Read file
    df = pd.read_sql_query("SELECT * FROM DEMOGRAPHICS", raw_conn, parse_dates=True)

    # Create single label combining mrn and csn
    df['mrn_csn_pair'] = list(zip(df.mrn,df.csn))
    df['mrn_csn_pair'] = df['mrn_csn_pair'].astype(str)

    # Calculate difference in admission and discharge time
    df['admission_datetime']=pd.to_datetime(df['admission_datetime'])
    df['discharge_datetime']=pd.to_datetime(df['discharge_datetime'])
    df['time_in_hospital_minutes']=df['discharge_datetime']-df['admission_datetime']

    # Convert the total hospital stay to minutes
    df['time_in_hospital_minutes']=df['time_in_hospital_minutes'].apply(timedelta_to_min)

    # Create dummy fariable for gender
    df['male'] = pd.get_dummies(df['gender'], drop_first=True)

    # Group all patients with ages over 90
    df['age'] = df['age'].apply(lambda x:'90' if x == '90+' else x)
    df['age'] = pd.to_numeric(df['age'])

    # Create new feature to return CCI score
    df['cci'] = df['charlson_comorbidity_index'].apply(cci)

    # Load in units manual coding file
    path = "S:\\Dehydration_stroke\\Team Emerald\\Working Data\\Preprocessed\\Working\\Manual_Coding\\Annotated\\Units_ann.csv"
    units_classes = pd.read_csv(path)

    # Create dictionary to reference department units
    a = pd.Series(units_classes['unit'])
    b = pd.Series(units_classes['Unit'])
    unit_dict = dict(zip(a.values,b.values))
    unit_dict['BVBCP BURTON 2'] = 'Floor'

    # Create new feature column categorizing units
    df['admit_unit'] = df['admit_department'].apply(lambda x: unit_dict[x])
    df['discharge_unit'] = df['discharge_department'].apply(lambda x: unit_dict[x])
    
    # Convert unique unit names to dummie variables
    admit_names = ['admit_floor', 'admit_ICU', 'admit_stroke_unit']
    discharge_names = ['discharge_floor', 'discharge_ICU', 'discharge_stroke_unit']
    df[admit_names] = pd.get_dummies(df['admit_unit'], drop_first=False)
    df[discharge_names] = pd.get_dummies(df['discharge_unit'], drop_first=False)

    # Load in services manual coding file
    services_path = "S:\\Dehydration_stroke\\Team Emerald\\Working Data\\Preprocessed\\Working\\Manual_Coding\\Annotated\\Services_ann.csv"
    services_classes = pd.read_csv(services_path)

    a = pd.Series(services_classes['admit_service'])
    b = pd.Series(services_classes['Service'])
    service_dict = dict(zip(a.values,b.values))

    df['admit_service'] = df['admit_service'].apply(lambda x: service_dict[x])
    df['Neuro'] = pd.get_dummies(df['admit_service'])

    # Create dummy variables to categorize race
    cols = ['White or Caucasian', 'Black or African American', 'Other', 'Declined', 'Unknown',
           'American Indian or Alaskan Native', 'Asian', 'Native Hawaiian or Other Pacific Islander',
           ]
    for i in range(len(cols)):
        df[cols[i]] = df['race'].apply(lambda x: 1 if cols[i] in x else 0)
        
    # Drop all other unused features
    df = df.drop(['ed_arrival_datetime', 'index', 'charlson_comorbidity_index', 
                     'gender', 'mrn', 'csn', 'admit_department',
                     'discharge_department', 'admit_unit', 'discharge_unit',
                     'admit_service', 'race'], axis=1)
    
    print('\nDemographics: \n', df.head())

    # Write to processed
    df.to_sql('DEMOGRAPHICS', processed_conn, if_exists='replace')





##### Dx #####
if to_run['Dx']:
    # Read file
    df = pd.read_sql_query("SELECT * FROM DX", raw_conn, parse_dates=True) 

    # Create new lable for mrn, csn pair
    df['mrn_csn_pair'] = list(zip(df.mrn,df.csn))
    df = df.sort_values('mrn_csn_pair')
    df['mrn_csn_pair'] = df['mrn_csn_pair'].astype(str)

    # Load in csv data categorizing diagnoses into stroke types
    path = "S:\Dehydration_stroke\Team Emerald\Working Data\Preprocessed\Working\Manual_Coding\Annotated\Dx_class_ann.csv"
    df_stroke_classes = pd.read_csv(path)

    # Create dictionary to reference stroke types
    a = pd.Series(df_stroke_classes['diagnosis'])
    b = pd.Series(df_stroke_classes['stroke_type'])
    stroke_dict = dict(zip(a.values,b.values))

    # Only include patients with a primary or an ED diagnosis
    df = df[(df['primary_dx'] == 'Y') | (df['ed_dx'] == 'Y')]

    # Create new stroke class column categorizing diagnosis based on dictionary
    df['stroke_class'] = df['diagnosis'].apply(lambda x: stroke_dict[x])

    # Convert stroke classes to dummy variables
    dummies = pd.get_dummies(df['stroke_class'])
    df[['hemorrhagic', 'ischemic', 'no_stroke', 'probable']] = dummies

    # Drop unnecessary columns
    df = df.reset_index()
    df = df.drop(['level_0', 'mrn', 'csn', 'index', 'icd9', 'icd10', 
                'diagnosis', 'primary_dx', 'ed_dx', 'stroke_class'], axis=1)

    print('\nDx: \n',df.head())

    # Write to processed
    df.to_sql('DX', processed_conn, if_exists='replace')





##### Hx #####
if to_run['Hx']:
    # Read file
    df = pd.read_sql_query("SELECT * FROM HX", raw_conn, parse_dates=True)

    # Create new lable for mrn, csn pair
    df['mrn_csn_pair'] = list(zip(df.mrn,df.csn))
    df = df.sort_values('mrn_csn_pair')
    df['mrn_csn_pair'] = df['mrn_csn_pair'].astype(str)

    # Load in csv data categorizing descriptions into categories
    path = "S:\Dehydration_stroke\Team Emerald\Working Data\Preprocessed\Working\Manual_Coding\Annotated\Hx_ann.csv"
    hx_categories = pd.read_csv(path)

    # Create dictionary to map descriptions to categories
    a = pd.Series(hx_categories['description'])
    b = pd.Series(hx_categories['Comorbidity'])
    hx_dict = dict(zip(a.values,b.values))
    hx_dict['Tuberculin test reaction']=float('NaN')

    # Create new feature with hx class
    df['hx_class'] = df['description'].apply(lambda x: str(hx_dict[x]))

    # Convert hx class feature to dummies
    cols = pd.get_dummies(df['hx_class']).columns
    df[cols] = pd.get_dummies(df['hx_class'])
    df = df.drop(['index','mrn','csn','description','resolved_datetime',
                'icd10','hx_class'], axis=1)

    # Identify unique pairs
    unique_pairs = df['mrn_csn_pair'].unique()

    # Iterate through unique pairs to gather units for each individual patient
    d = pd.DataFrame(columns=cols)
    for patient in unique_pairs:
        series = df[df['mrn_csn_pair'] == patient][cols].sum()
        d = d.append(series, ignore_index=True)
    d = d.rename(columns = {'nan':'other diseases'})

    # Insert identifier column to new dataframe
    d.insert(0, 'mrn_csn_pair', unique_pairs)

    print('\nHx: \n', d.head())

    # Write to processed
    d.to_sql('HX', processed_conn, if_exists='replace')





##### LDA #####
if to_run['LDA']:
    # Read file
    df = pd.read_sql_query("SELECT * FROM LDA", raw_conn, parse_dates=True)

    # Create new lable for mrn, csn pair
    df['mrn_csn_pair'] = list(zip(df.mrn,df.csn))
    df = df.sort_values('mrn_csn_pair')
    df['mrn_csn_pair'] = df['mrn_csn_pair'].astype(str)

    # Drop rows that do not have a raw LDA Name
    df = df.drop(df[df['lda_name'].isnull()].index)

    # Load in csv data categorizing descriptions into categories
    path = "S:\Dehydration_stroke\Team Emerald\Working Data\Preprocessed\Working\Manual_Coding\Annotated\LDA_names.csv"
    lda_names = pd.read_csv(path)

    # Create dictionary to map descriptions to categories
    a = pd.Series(lda_names['lda_name'])
    b = pd.Series(lda_names['Unnamed: 2'])
    lda_dict = dict(zip(a.values,b.values))

    # Create new feature with lda class
    df['lda_names'] = df['lda_name'].apply(lambda x: str(lda_dict[x]))

    # Convert hx class feature to dummies
    cols = pd.get_dummies(df['lda_names']).columns
    df[cols] = pd.get_dummies(df['lda_names'])
    df = df.drop(['index','mrn','csn','placed_datetime','removed_datetime',
                'template_name','lda_name', 'lda_measurements_and_assessments',
                'lda_names'], axis=1)

    # Identify unique pairs
    unique_pairs = df['mrn_csn_pair'].unique()

    # Iterate through unique pairs to gather units for each individual patient
    d = pd.DataFrame(columns=cols)
    for patient in unique_pairs:
        series = df[df['mrn_csn_pair'] == patient][cols].sum()
        d = d.append(series, ignore_index=True)

    # Insert identifier column to new dataframe
    d.insert(0, 'mrn_csn_pair', unique_pairs)

    print('\nLDA: \n', d.head())

    # Write to processed
    d.to_sql('LDA', processed_conn, if_exists='replace')





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


raw_conn.close()
processed_conn.close()