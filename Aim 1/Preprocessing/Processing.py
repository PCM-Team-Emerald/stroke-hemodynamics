# Imports
import json, os

import numpy as np
import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',10000)
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

##### Flowsheet #####
if to_run['Flowsheet']:
    # Read file
    dat = pd.read_sql('SELECT * FROM FLOWSHEET', raw_conn, parse_dates=True, index_col='index')

    # Read manual coding
    mc = pd.read_csv('S:/Dehydration_stroke/Team Emerald/Working Data/Preprocessed/Working/Manual_Coding/Annotated/Flowsheet_names.csv')
    dat = pd.merge(dat,mc,how='left',on='flowsheet_row_name')
    dat_pivoted = pd.pivot_table(dat,index=['mrn','csn','recorded_datetime'],
                                 columns='Name',values='value',aggfunc='first')

    # Processing
    to_keep = ['bp','pulse_ox','pulse','temp', 'ampac_mobility_tscore','ampac_activity_tscore','hlm']

    dat_pivoted = dat_pivoted[to_keep]
    dat_pivoted.dropna(how='all',inplace=True)

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

    dat_pivoted['hlm'] = dat_pivoted['hlm'].replace(to_replace,replacing).str.strip()
    dat_pivoted['hlm'] = dat_pivoted['hlm'].apply(lambda x: x if x != x else x if len(x)==1 else np.NaN)

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

    # Dist plots
    #import matplotlib.pyplot as plt
    #for c in dat_pivoted.columns:
    #    plt.hist(dat_pivoted[c])
    #    plt.title(c)
    #    plt.show()

    # Melt
    dat_melted = dat_pivoted.reset_index(drop=False).melt(id_vars=['mrn','csn','recorded_datetime'])

    # MRN, CSN Pairs
    dat_melted['mrn_csn_pair'] = dat_melted.apply(lambda x: '({}, {})'.format(x['mrn'],x['csn']), axis=1)

    # Write to processed
    dat_melted.to_sql('FLOWSHEET', processed_conn, if_exists='replace', index=False)

##### Neuro #####
if to_run['Neuro']:
    # Read file
    dat = pd.read_sql('SELECT * FROM NEURO', raw_conn, parse_dates=True, index_col='index')

    # Read manual coding
    mc = pd.read_csv(
        'S:/Dehydration_stroke/Team Emerald/Working Data/Preprocessed/Working/Manual_Coding/Annotated/neuro_names.csv')

    dat = pd.merge(dat, mc, how='left', on='flowsheet_row_name')
    dat = dat[dat['Name'].notnull()]
    dat_pivoted = pd.pivot_table(dat, index=['mrn', 'csn', 'recorded_datetime'],
                                 columns='Name', values='value', aggfunc='first')
    # Processing
    to_keep = ['glasgow_score',
               'glasgow_eye_opening',
               'glasgow_verbal_response',
               'glasgow_motor_response',
               'cam_icu',
               'consciousness',
               'orientation']

    dat_pivoted = dat_pivoted[to_keep]
    dat_pivoted.dropna(how='all', inplace=True)

    # Dist plots
    # import matplotlib.pyplot as plt
    # for c in dat_pivoted.columns:
    #    plt.hist(dat_pivoted[c])
    #    plt.title(c)
    #    plt.show()

    # Melt
    dat_melted = dat_pivoted.reset_index(drop=False).melt(id_vars=['mrn', 'csn', 'recorded_datetime'])

    # MRN, CSN Pairs
    dat_melted['mrn_csn_pair'] = dat_melted.apply(lambda x: '({}, {})'.format(x['mrn'], x['csn']), axis=1)

    # Write to processed
    dat_melted.to_sql('NEURO', processed_conn, if_exists='replace', index=False)

##### Dispo #####
if to_run['Dispo']:
    # Read file
    dat = pd.read_sql('SELECT * FROM DISPO', raw_conn, parse_dates=True, index_col='index')

    # Read manual coding
    mc = pd.read_csv(
        'S:/Dehydration_stroke/Team Emerald/Working Data/Preprocessed/Working/Manual_Coding/Annotated/Dispo_names.csv')

    dat = pd.merge(dat, mc, how='left', on='disposition')
    dat = dat[dat['Name'].notnull()]

    # MRN, CSN Pairs
    dat['mrn_csn_pair'] = dat.apply(lambda x: '({}, {})'.format(x['mrn'], x['csn']), axis=1)
    dat = dat[['mrn_csn_pair', 'Name']]
    # Write to processed
    dat.to_sql('DISPO', processed_conn, if_exists='replace', index=False)

##### Meds #####
if to_run['MAR']:
    # Read file
    dat = pd.read_sql('SELECT * FROM MAR', raw_conn, parse_dates=True, index_col='index')

    # Read manual coding
    mc = pd.read_csv(
        'S:/Dehydration_stroke/Team Emerald/Working Data/Preprocessed/Working/Manual_Coding/Annotated/med_names.csv')

    dat = pd.merge(dat, mc, how='left', on='medication_name')

    # Processing
    meds_to_keep = [
        '5asa_der',
        'acei',
        'acei_diuretic',
        'adenosine',
        'adh',
        'adh_analog',
        'adrenergic_agonist',
        'albumin',
        'alpha_agonist',
        'alpha_beta_agonist',
        'alpha_blocker',
        'alpha2_agonist',
        'anti_anginal',
        'antiarrhythmic',
        'anticholinergic',
        'anticholinergic_beta_agonist',
        'antiplatelet',
        'antithyroid',
        'anxiolytic',
        'arb',
        'arb_ccb',
        'arb_diuretic',
        'arb_neprilysin_inh',
        'bb_diuretic',
        'beta_agonist',
        'beta_agonist_corticosteroid',
        'beta_blocker',
        'beta3_agonist',
        'ccb',
        'ccb_acei',
        'ccb_arb',
        'ccb_statin',
        'cholinergic',
        'corticosteroid',
        'crystalloid',
        'colloid',
        'direct_renin_inh',
        'diuretic',
        'doac',
        'GP_IIb_IIIa_inh',
        'heparin',
        'hypertonic_saline',
        'hypertonic_saline_kcl',
        'inotrope',
        'k_channel_blocker',
        'laxative',
        'lmwh',
        'nmba',
        'nmba_receptor_antagonist',
        'nsaid',
        'acetaminophen',
        'opioid',
        'opioid_acetaminophen',
        'osmotic',
        'p2y12_inh',
        'pde_inh',
        'pde5_inh',
        'peritoneal_dialysis_soln',
        'sodium_chloride',
        'somatostatin_analog',
        'stimulant',
        'stool_softener',
        'thrombolytic',
        'thyroid_hormone',
        'txa',
        'vasodilator'
    ]

    dat = dat[dat['class'].isin(meds_to_keep)]
    dat.dropna(how='all', inplace=True)

    # MRN, CSN Pairs
    dat['mrn_csn_pair'] = dat.apply(lambda x: '({}, {})'.format(x['mrn'], x['csn']), axis=1)
    dat = dat[['mrn_csn_pair', 'class', 'route', 'med_admin_start_datetime', 'med_admin_end_datetime']]
    # Write to processed
    print(dat.dtypes)
    dat.to_sql('MAR', processed_conn, if_exists='replace', index=False)