# Imports
import json
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
import sqlite3
import dateutil.parser as parser
import os

# Config file
with open('cfg.json') as json_file:
    cfg = json.load(json_file)
    
# Datetime parser function
def datetimeColToISO(df, dt_cols):
    for c in dt_cols:
        try:
            df[c] = df[c].apply(lambda x: parser.parse(x).isoformat().replace('T', ' '))
        except TypeError:
            pass
    return df

# Setup paths and SQLite DB
data_dir = os.path.join(cfg['WORKING_DATA_DIR'], 'Raw/')
processed_dir = os.path.join(cfg['WORKING_DATA_DIR'], 'Processed/Processed.db')
annotated_dir = os.path.join(cfg['WORKING_DATA_DIR'], 'Processed/Manual_Coding')
vars_to_include_dir = os.path.join(cfg['WORKING_DATA_DIR'], 'Processed/Vars_to_Keep')
processed_conn = sqlite3.connect(processed_dir)

to_run = {
    'ADT':          True,
    'Demographics': True,
    'Dx':           True,
    'Flowsheet':    True,
    'IO_Flowsheet': True,
    'Labs':         True,
    'LDA':          True,
    'MAR':          True,
    'Hx':           True,
    'Problem_List': True,
    'Neuro':        True,
    'Dispo':        True
}

##### Flowsheet #####
if to_run['Flowsheet']:
    # Read file
    dat = pd.read_table(os.path.join(data_dir, 'flowsheet.txt'), sep='|')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'recorded_RAW', 'recorded_datetime', 'value', 'template_name', 'flowsheet_row_name']
    dat.drop(columns=['recorded_RAW'])

    # Convert datetimes to ISO
    datetime_cols = ['recorded_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Save to db
    dat.drop_duplicates(inplace=True, ignore_index=True)

    # Read manual coding
    mc = pd.read_csv(os.path.join(annotated_dir, 'Flowsheet_names.csv'))
    dat = pd.merge(dat,mc,how='left',on='flowsheet_row_name')
    dat_pivoted = pd.pivot_table(dat,index=['mrn','csn','recorded_datetime'],
                                 columns='Name',values='value',aggfunc='first')

    # Processing
    with open(os.path.join(vars_to_include_dir, 'Flowsheet.txt')) as f:
        to_keep = f.read().splitlines()

    dat_pivoted = dat_pivoted[to_keep]

    # BPs
    bps = dat_pivoted['bp'].str.split('/', n=1, expand=True)
    dat_pivoted['sbp'] = bps[0]
    dat_pivoted['dbp'] = bps[1]
    dat_pivoted.drop(columns=['bp'], inplace=True)


    # HLM
    to_replace = ['Lying in bed (1)',
                  'Turn self in bed/Bed activity/Dependent transfer (2)',
                  'Sit on edge of bed (3)',
                  'Transfer to chair (4)',
                  'Stand for 1 minute (5)',
                  'Walk 10+ steps (6)',
                  'Walk 25+ feet (7)',
                  'Walk 250+ feet (8)']
    replacing = [1,2,3,4,5,6,7,8]

    dat_pivoted['hlm'] = dat_pivoted['hlm'].replace(to_replace,replacing)
    dat_pivoted['hlm'] = dat_pivoted['hlm'].apply(lambda x: x if x != x else x if type(x) == int else np.NaN)

    for c in dat_pivoted.columns:
        dat_pivoted[c] = dat_pivoted[c].astype(float)

    # Convert temp F to C
    dat_pivoted['temp'] = dat_pivoted['temp'].apply(lambda x: x if x < 70 else (x-32)*5/9)

    # Remove extreme values
    dat_pivoted['temp'] = dat_pivoted['temp'].apply(lambda x: np.NaN if x < 35 or x > 42.2 else x)
    dat_pivoted['pulse_ox'] = dat_pivoted['pulse_ox'].apply(lambda x: np.NaN if x < 60 or x > 100 else x)
    dat_pivoted['pulse'] = dat_pivoted['pulse'].apply(lambda x: np.NaN if x < 30 or x > 200 else x)
    dat_pivoted['sbp'] = dat_pivoted['sbp'].apply(lambda x: np.NaN if x < 50 or x > 300 else x)
    dat_pivoted['dbp'] = dat_pivoted['dbp'].apply(lambda x: np.NaN if x < 10 or x > 200 else x)

    # Melt
    dat_melted = dat_pivoted.reset_index(drop=False).melt(id_vars=['mrn','csn','recorded_datetime'])

    # MRN, CSN Pairs
    dat_melted['mrn_csn_pair'] = dat_melted.apply(lambda x: '({}, {})'.format(x['mrn'],x['csn']), axis=1)

    # Write to processed
    dat_melted.dropna(subset=['value'], inplace=True)
    dat_melted.to_sql('FLOWSHEET', processed_conn, if_exists='replace', index=False)

##### Neuro #####
if to_run['Neuro']:
    # Read file
    dat = pd.read_table(os.path.join(data_dir, 'neuro.txt'), sep='|')

    # Drop unnecesary cols, rename
    # Col names are wrong, overriding here:
    dat.columns = ['mrn', 'csn', 'recorded_datetime_RAW',
                   'recorded_datetime', 'value',
                   'template_name', 'flowsheet_row_name']
    dat.drop(columns=['recorded_datetime_RAW'], inplace=True)

    # Convert datetimes to ISO
    datetime_cols = ['recorded_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Save to db
    dat.drop_duplicates(inplace=True, ignore_index=True)

    # Read manual coding
    mc = pd.read_csv(os.path.join(annotated_dir, 'neuro_names.csv'))
    dat = pd.merge(dat, mc, how='left', on='flowsheet_row_name')

    dat = dat[dat['Name'].notnull()]
    dat_pivoted = pd.pivot_table(dat, index=['mrn', 'csn', 'recorded_datetime'],
                                 columns='Name', values='value', aggfunc='first')
    # Processing
    with open(os.path.join(vars_to_include_dir, 'Neuro.txt')) as f:
        to_keep = f.read().splitlines()

    dat_pivoted = dat_pivoted[to_keep]

    # Orientation
    orientation = pd.read_csv(os.path.join(annotated_dir, 'vars/orientation.csv'))
    dat_pivoted['orientation'] = dat_pivoted['orientation'].replace(orientation['value'].tolist(), orientation['code'].tolist())

    # Consciousness
    consciousness = pd.read_csv(os.path.join(annotated_dir, 'vars/consciousness.csv'))
    dat_pivoted['consciousness'] = dat_pivoted['consciousness'].replace(consciousness['value'].tolist(),
                                                                    consciousness['code'].tolist())

    dat_pivoted.dropna(how='all', inplace=True)

    # Melt
    dat_melted = dat_pivoted.reset_index(drop=False).melt(id_vars=['mrn', 'csn', 'recorded_datetime'])

    # MRN, CSN Pairs
    dat_melted['mrn_csn_pair'] = dat_melted.apply(lambda x: '({}, {})'.format(x['mrn'], x['csn']), axis=1)

    # Write to processed
    dat_melted.dropna(subset=['value'], inplace=True)
    dat_melted.to_sql('NEURO', processed_conn, if_exists='replace', index=False)

##### Dispo #####
if to_run['Dispo']:
    # Read file
    dat = pd.read_table(os.path.join(data_dir, 'dispo.txt'), sep='|')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'deceased', 'disposition']

    # Save to db
    dat.drop_duplicates(inplace=True, ignore_index=True)

    # Read manual coding
    mc = pd.read_csv(os.path.join(annotated_dir, 'Dispo_names.csv'))

    dat = pd.merge(dat, mc, how='left', on='disposition')
    dat = dat[dat['Name'].notnull()]

    # MRN, CSN Pairs
    dat['mrn_csn_pair'] = dat.apply(lambda x: '({}, {})'.format(x['mrn'], x['csn']), axis=1)
    dat = dat[['mrn_csn_pair', 'Name']]
    mrn_csns = dat['mrn_csn_pair']
    dat = pd.get_dummies(dat['Name'], prefix='dispo')
    dat.index = mrn_csns
    # Write to processed
    dat.reset_index(drop=False, inplace=True)
    dat.to_sql('DISPO', processed_conn, if_exists='replace', index=False)

##### Meds #####
if to_run['MAR']:
    # Read file
    dat = pd.read_table(os.path.join(data_dir, 'mar.txt'))

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'medication_name', 'ordered_datetime', 'order_id', 'med_admin_start_datetime',
                   'med_admin_end_datetime', 'dosage']

    # Convert datetimes to ISO
    datetime_cols = ['ordered_datetime', 'med_admin_start_datetime', 'med_admin_end_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Save to db
    dat.drop_duplicates(inplace=True, ignore_index=True)

    # Read manual coding
    mc = pd.read_csv(os.path.join(annotated_dir, 'med_names.csv'))

    dat = pd.merge(dat, mc, how='left', on='medication_name')

    # Processing
    with open(os.path.join(vars_to_include_dir, 'MAR.txt')) as f:
        meds_to_keep = f.read().splitlines()

    dat = dat[dat['class'].isin(meds_to_keep)]
    dat.dropna(how='all', inplace=True)

    dat_list = dat[['class','medication_name']].drop_duplicates(ignore_index=True).sort_values(by='medication_name').groupby('class').agg(lambda x: '\n'.join(x)).reset_index(drop=False)
    with open(os.path.join(annotated_dir, 'Meds.txt'), 'w') as f:
        for i, r in dat_list.iterrows():
            f.write(r['class']+'\n')
            f.write(r['medication_name']+'\n')
            f.write('\n')

    # MRN, CSN Pairs
    dat['mrn_csn_pair'] = dat.apply(lambda x: '({}, {})'.format(x['mrn'], x['csn']), axis=1)
    dat = dat[['mrn_csn_pair', 'class', 'med_admin_start_datetime', 'med_admin_end_datetime', 'dosage']]
    # Write to processed
    dat.to_sql('MAR', processed_conn, if_exists='replace', index=False)


##### ADT #####
if to_run['ADT']:
    # Read file
    dat = pd.read_table(os.path.join(data_dir, 'adt.txt'))

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'unit', 'in', 'out']

    # Convert datetimes to ISO
    datetime_cols = ['in', 'out']
    dat = datetimeColToISO(dat, datetime_cols)

    # Save to db
    dat.drop_duplicates(inplace=True, ignore_index=True)

    # Create new lable for mrn, csn pair
    dat['mrn_csn_pair'] = list(zip(dat.mrn,dat.csn))
    dat = dat.sort_values('mrn_csn_pair')
    dat['mrn_csn_pair'] = dat['mrn_csn_pair'].astype(str)

    # Load in manual coding to categorize unit classes
    units_classes = pd.read_csv(os.path.join(annotated_dir, 'Units_ann.csv'))

    dat = pd.merge(dat, units_classes, how='inner', on='unit')
    dat = dat[['mrn_csn_pair', 'in', 'out', 'Unit']]
    dat.dropna(how='any', inplace=True)

    # Write to processed
    dat.to_sql('ADT', processed_conn, if_exists='replace', index=False)

##### Demographics #####
# Function to extract CCI score from string
def cci(s):
    if str(s) == 'None':
        return None
    if type(s) == float:
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
    dat = pd.read_table(os.path.join(data_dir, 'demographics.txt'), sep='|', encoding='ISO-8859-1')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'admission_datetime', 'discharge_datetime', 'age', 'gender', 'race',
                   'ed_arrival_datetime', 'admit_service', 'admit_department', 'discharge_department',
                   'charlson_comorbidity_index']

    # Convert datetimes to ISO
    datetime_cols = ['admission_datetime', 'discharge_datetime', 'ed_arrival_datetime']
    df = datetimeColToISO(dat, datetime_cols)

    # Save to db
    df.drop_duplicates(inplace=True, ignore_index=True)

    # Create single label combining mrn and csn
    df['mrn_csn_pair'] = list(zip(df.mrn,df.csn))
    df['mrn_csn_pair'] = df['mrn_csn_pair'].astype(str)

    # Calculate difference in admission and discharge time
    df['admission_datetime']=pd.to_datetime(df['admission_datetime'])
    df['discharge_datetime']=pd.to_datetime(df['discharge_datetime'])
    df['time_in_hospital_minutes']=df['discharge_datetime']-df['admission_datetime']

    # Convert the total hospital stay to minutes
    df['time_in_hospital_minutes']=df['time_in_hospital_minutes'].apply(timedelta_to_min)
    df['los_ge_7'] = df['time_in_hospital_minutes'].apply(lambda x: 1 if x >= 60*24*7 else 0)

    # Admission TOD
    df['admit_tod'] = df['admission_datetime'].apply(
        lambda x: x.hour + x.minute/60 + x.second/3600)
    df['admit_tod'] = (df['admit_tod'] - 7).apply(lambda x: x if x > 0 else x + 24)
    df['admit_tod'] = df['admit_tod']/24

    # Create dummy fariable for gender
    df['male'] = pd.get_dummies(df['gender'], drop_first=True)

    # Group all patients with ages over 90
    df['age'] = df['age'].apply(lambda x:'90' if x == '90+' else x)
    df['age'] = pd.to_numeric(df['age'])

    # Create new feature to return CCI score
    df['cci'] = df['charlson_comorbidity_index'].apply(cci)

    # Load in units manual coding file
    units_classes = pd.read_csv(os.path.join(annotated_dir, 'Units_ann.csv'))

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
    services_classes = pd.read_csv(os.path.join(annotated_dir, 'Services_ann.csv'))

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
    df = df.drop(['ed_arrival_datetime', 'charlson_comorbidity_index',
                     'gender', 'mrn', 'csn', 'admit_department',
                     'discharge_department', 'admit_unit', 'discharge_unit',
                     'admit_service', 'race'], axis=1)
    
    print('\nDemographics: \n', df.head())

    # Write to processed
    df.to_sql('DEMOGRAPHICS', processed_conn, if_exists='replace', index=False)

##### Dx #####
if to_run['Dx']:
    # Read file
    df = pd.read_table(os.path.join(data_dir, 'dx.txt'), sep='|', encoding='ISO-8859-1')

    # Drop unnecesary cols, rename
    df.columns = ['mrn', 'csn', 'icd9', 'icd10', 'diagnosis', 'primary_dx', 'ed_dx']

    # Save to db
    df.drop_duplicates(inplace=True, ignore_index=True) 

    # Create new lable for mrn, csn pair
    df['mrn_csn_pair'] = list(zip(df.mrn,df.csn))
    df = df.sort_values('mrn_csn_pair')
    df['mrn_csn_pair'] = df['mrn_csn_pair'].astype(str)

    # Load in csv data categorizing diagnoses into stroke types
    df_stroke_classes = pd.read_csv(os.path.join(annotated_dir, 'Dx_class_ann_updated.csv'))

    dat = pd.merge(df, df_stroke_classes, how='left', on=['icd10', 'diagnosis'])
    dat = dat[(dat['primary_dx'] == 'Y') | (dat['ed_dx'] == 'Y')]

    dat['hemorrhagic_stroke'] = dat['stroke_type'] == 'H'
    dat['ischemic_stroke'] = dat['stroke_type'] == 'I'
    dat = dat[['mrn_csn_pair', 'hemorrhagic_stroke', 'ischemic_stroke']].groupby('mrn_csn_pair').agg('sum')
    dat['hemorrhagic_stroke'] = dat['hemorrhagic_stroke'].apply(lambda x: 0 if x == 0 else 1)
    dat['ischemic_stroke'] = dat['ischemic_stroke'].apply(lambda x: 0 if x == 0 else 1)
    dat.reset_index(drop=False, inplace=True)
    # Write to processed
    dat.to_sql('DX', processed_conn, if_exists='replace', index=False)

##### Hx #####
if to_run['Hx']:
    # Read file
    df = pd.read_table(os.path.join(data_dir, 'hx.txt'))

    # Drop unnecesary cols, rename
    df.columns = ['mrn', 'csn', 'description', 'resolved_datetime', 'icd10']

    # Convert datetimes to ISO
    datetime_cols = ['resolved_datetime']
    df = datetimeColToISO(df, datetime_cols)

    # Save to db
    dat.drop_duplicates(inplace=True, ignore_index=True)

    # Create new lable for mrn, csn pair
    df['mrn_csn_pair'] = list(zip(df.mrn,df.csn))
    df = df.sort_values('mrn_csn_pair')
    df['mrn_csn_pair'] = df['mrn_csn_pair'].astype(str)

    # Load in csv data categorizing descriptions into categories
    hx_categories = pd.read_csv(os.path.join(annotated_dir, 'Hx_ann.csv'))

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
    df = df.drop(['mrn','csn','description','resolved_datetime',
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
    d.to_sql('HX', processed_conn, if_exists='replace', index=False)

##### LDA #####
if to_run['LDA']:
    # Read file
    df = pd.read_table(os.path.join(data_dir, 'lda.txt'), encoding='ISO-8859-1')

    # Drop unnecesary cols, rename
    df.columns = ['mrn', 'csn', 'placed_datetime', 'removed_datetime', 'template_name', 'lda_name',
                   'lda_measurements_and_assessments']

    # Convert datetimes to ISO
    datetime_cols = ['placed_datetime', 'removed_datetime']
    df = datetimeColToISO(df, datetime_cols)

    # Save to db
    df.drop_duplicates(inplace=True, ignore_index=True)

    # Create new lable for mrn, csn pair
    df['mrn_csn_pair'] = list(zip(df.mrn,df.csn))
    df = df.sort_values('mrn_csn_pair')
    df['mrn_csn_pair'] = df['mrn_csn_pair'].astype(str)

    # Drop rows that do not have a raw LDA Name
    df = df.drop(df[df['lda_name'].isnull()].index)

    # Load in csv data categorizing descriptions into categories
    lda_names = pd.read_csv(os.path.join(annotated_dir, 'LDA_names.csv'))

    dat = pd.merge(df, lda_names, how='inner', on='lda_name')
    dat = dat[['mrn_csn_pair', 'placed_datetime', 'removed_datetime', 'Name']]

    dat.dropna(how='any', inplace=True)
    dat.drop_duplicates(inplace=True)

    # Write to processed
    dat.to_sql('LDA', processed_conn, if_exists='replace', index=False)

##### IO Flowsheet #####
if to_run['IO_Flowsheet']:
    # Read file
    dat = pd.read_table(os.path.join(data_dir, 'io_flowsheet.txt'), sep='|')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'recorded_datetime', 'value', 'template_name', 'flowsheet_row_name']

    # Convert datetimes to ISO
    datetime_cols = ['recorded_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Save to db
    dat.drop_duplicates(inplace=True, ignore_index=True)

    # Read manual coding
    mc = pd.read_csv(os.path.join(annotated_dir, 'IO_Flowsheet_names.csv'))
    dat = pd.merge(dat, mc, how='left', on='flowsheet_row_name')
    dat_pivoted = pd.pivot_table(dat, index=['mrn', 'csn', 'recorded_datetime'],
                                 columns='Name', values='value', aggfunc='first')

    # Processing
    with open(os.path.join(vars_to_include_dir, 'IO_Flowsheet.txt')) as f:
        to_keep = f.read().splitlines()

    dat_pivoted = dat_pivoted[to_keep]

    dat_pivoted.dropna(how='all', inplace=True)

    # Melt
    dat_melted = dat_pivoted.reset_index(drop=False).melt(id_vars=['mrn', 'csn', 'recorded_datetime'])

    # MRN, CSN Pairs
    dat_melted.dropna(subset=['value'], inplace=True)
    dat_melted['mrn_csn_pair'] = dat_melted.apply(lambda x: '({}, {})'.format(x['mrn'], x['csn']), axis=1)

    # Write to processed
    dat_melted.to_sql('IO_FLOWSHEET', processed_conn, if_exists='replace', index=False)

##### Labs #####
if to_run['Labs']:
    # Read file
    dat = pd.read_table(os.path.join(data_dir, 'labs.txt'), sep='|')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'result_datetime',
                   'units', 'value_numeric', 'value_text', 'lab_result_comment', 'order_id', 'order_description',
                   'order_display_name', 'component_name', 'component_base_name']

    dat = dat[['mrn', 'csn', 'result_datetime',
               'units', 'value_numeric', 'value_text', 'lab_result_comment', 'order_id', 'order_description',
               'order_display_name', 'component_name', 'component_base_name']]

    # Convert datetimes to ISO
    datetime_cols = ['result_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Save to db
    dat.drop_duplicates(inplace=True, ignore_index=True)

    # Read manual coding
    mc = pd.read_csv(os.path.join(annotated_dir, 'Lab_names.csv'))
    for c in ['order_description', 'component_name', 'component_base_name']:
        mc[c] = mc[c].str.upper()
        dat[c] = dat[c].str.upper()
    dat = pd.merge(dat, mc, how='left', on=['order_description', 'component_name', 'component_base_name'])
    dat_pivoted = pd.pivot_table(dat, index=['mrn', 'csn', 'result_datetime'],
                                 columns='Name', values='value_numeric', aggfunc='first')

    # Processing
    with open(os.path.join(vars_to_include_dir, 'Labs.txt')) as f:
        to_keep = f.read().splitlines()

    dat_pivoted = dat_pivoted[to_keep]
    dat_pivoted.dropna(how='all', inplace=True)

    # Melt
    dat_melted = dat_pivoted.reset_index(drop=False).melt(id_vars=['mrn', 'csn', 'result_datetime'])

    # MRN, CSN Pairs
    dat_melted['mrn_csn_pair'] = dat_melted.apply(lambda x: '({}, {})'.format(x['mrn'], x['csn']), axis=1)

    # Write to processed
    dat_melted.dropna(subset=['value'], inplace=True)
    dat_melted.to_sql('LABS', processed_conn, if_exists='replace', index=False)

##### Problem List #####
if to_run['Problem_List']:
    # Read file
    dat = pd.read_table(os.path.join(data_dir, 'problem_list.txt'), encoding='ISO-8859-1')

    # Drop unnecesary cols, rename
    dat.columns = ['mrn', 'csn', 'description', 'noted_datetime', 'resolved_datetime', 'icd10']

    # Convert datetimes to ISO
    datetime_cols = ['noted_datetime', 'resolved_datetime']
    dat = datetimeColToISO(dat, datetime_cols)

    # Save to db
    dat.drop_duplicates(inplace=True, ignore_index=True)

    # Read manual coding
    mc = pd.read_csv(os.path.join(annotated_dir, 'Hx_ann.csv'))
    dat = pd.merge(dat, mc, how='left', on=['icd10', 'description'])
    dat = dat[dat['Comorbidity'].notnull()]
    one_hot = pd.get_dummies(dat['Comorbidity'], prefix='problem_list')
    dat = dat.join(one_hot)

    # MRN, CSN Pairs
    dat['mrn_csn_pair'] = dat.apply(lambda x: '({}, {})'.format(x['mrn'], x['csn']), axis=1)
    
    dat.drop(columns=['mrn','csn','description','noted_datetime','resolved_datetime','icd10','0','Comorbidity'], inplace=True)
    dat = dat.groupby('mrn_csn_pair').agg('max')
    dat.reset_index(drop=False, inplace=True)

    # Write to processed
    dat.to_sql('PROBLEM_LIST', processed_conn, if_exists='replace', index=False)