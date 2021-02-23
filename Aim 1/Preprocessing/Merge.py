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
processed_dir = os.path.join(cfg['WORKING_DATA_DIR'], 'Preprocessed/Working/Processed.db')
processed_conn = sqlite3.connect(processed_dir)

merged_dir = os.path.join(cfg['WORKING_DATA_DIR'], 'Preprocessed/Working/Merged.db')
merged_conn = sqlite3.connect(merged_dir)


# Static predictors
query = 'SELECT mrn_csn_pairs.mrn_csn_pair AS mrn_csn_pair, ' \
        'DEMOGRAPHICS.age as age, ' \
        'DEMOGRAPHICS.male as male, ' \
        'DEMOGRAPHICS.admit_floor as admit_floor, ' \
        'DEMOGRAPHICS.admit_ICU as admit_icu, ' \
        'DEMOGRAPHICS.admit_stroke_unit as admit_stroke_unit, ' \
        'DEMOGRAPHICS.admit_tod as admit_tod, ' \
        'DEMOGRAPHICS.Neuro as admit_neuro, ' \
        'DEMOGRAPHICS."White or Caucasian" as race_white, ' \
        'DEMOGRAPHICS."Black or African American" as race_aa, ' \
        'DEMOGRAPHICS.Other as race_other, ' \
        'DEMOGRAPHICS."American Indian or Alaskan Native" as race_aian, ' \
        'DEMOGRAPHICS.Asian as race_asian, ' \
        'DEMOGRAPHICS."Native Hawaiian or Other Pacific Islander" as race_nhopi, ' \
        'DX.hemorrhagic_stroke as hemorrhagic_stroke, ' \
        'DX.ischemic_stroke as ischemic_stroke, ' \
        'HX."Atrial fibrillation" as hx_afib, ' \
        'HX.Cancer as hx_cancer, ' \
        'HX.Diabetes as hx_diabetes, ' \
        'HX."Heart failure" as hx_heart_failure, ' \
        'HX.Hypertension as hx_htn, ' \
        'HX."Kidney disease" as hx_kidney_disease, ' \
        'HX."other diseases" as hx_other, ' \
        'PROBLEM_LIST."problem_list_Atrial fibrillation" AS pl_afib, ' \
        'PROBLEM_LIST.problem_list_Cancer AS pl_cancer, ' \
        'PROBLEM_LIST.problem_list_Diabetes AS pl_diabetes, ' \
        'PROBLEM_LIST."problem_list_Heart failure" AS pl_heart_failure, ' \
        'PROBLEM_LIST.problem_list_Hypertension AS pl_htn, ' \
        'PROBLEM_LIST."problem_list_Kidney disease" AS pl_kidney_disease ' \
        'FROM mrn_csn_pairs ' \
        'LEFT JOIN DEMOGRAPHICS ON mrn_csn_pairs.mrn_csn_pair = DEMOGRAPHICS.mrn_csn_pair ' \
        'LEFT JOIN ADT_static ON mrn_csn_pairs.mrn_csn_pair = ADT_static.mrn_csn_pair ' \
        'LEFT JOIN DX on mrn_csn_pairs.mrn_csn_pair = DX.mrn_csn_pair ' \
        'LEFT JOIN HX on mrn_csn_pairs.mrn_csn_pair = HX.mrn_csn_pair ' \
        'LEFT JOIN LDA_static on mrn_csn_pairs.mrn_csn_pair = LDA_static.mrn_csn_pair ' \
        'LEFT JOIN PROBLEM_LIST on mrn_csn_pairs.mrn_csn_pair = PROBLEM_LIST.mrn_csn_pair'
dat = pd.read_sql(query, processed_conn, parse_dates=True)


# Fill NaNs
dat.drop_duplicates(inplace=True)
for c in dat.columns:
        if ('hx_' in c) or ('pl_' in c):
                dat[c].fillna(0, inplace=True)

print(dat.describe())

dat.to_sql('static_predictors', merged_conn, if_exists='replace', index=False)

# Outcomes
query = 'SELECT mrn_csn_pairs.mrn_csn_pair AS mrn_csn_pair, ' \
        'DISPO.dispo_alf, ' \
        'DISPO.dispo_deceased, ' \
        'DISPO.dispo_home, ' \
        'DISPO.dispo_home_care, ' \
        'DISPO.dispo_hospice, ' \
        'DISPO.dispo_hospital, ' \
        'DISPO.dispo_rehab, ' \
        'DISPO.dispo_snf, ' \
        'DEMOGRAPHICS.time_in_hospital_minutes as time_in_hospital, ' \
        'DEMOGRAPHICS.los_ge_7 as los_ge_7d, ' \
        'secondary_outcomes.ampac_activity_tscore AS ampac_activity_score, ' \
        'secondary_outcomes.ampac_mobility_tscore AS ampac_mobility_tscore, ' \
        'secondary_outcomes.glasgow_score AS gcs, ' \
        'secondary_outcomes.glasgow_eye_opening AS gcs_eye, ' \
        'secondary_outcomes.glasgow_motor_response AS gcs_motor, ' \
        'secondary_outcomes.glasgow_verbal_response AS gcs_verbal, ' \
        'secondary_outcomes.hlm AS hlm, ' \
        'secondary_outcomes.consciousness AS consciousness, ' \
        'secondary_outcomes.orientation AS orientation ' \
        'FROM mrn_csn_pairs ' \
        'LEFT JOIN DEMOGRAPHICS ON DEMOGRAPHICS.mrn_csn_pair = mrn_csn_pairs.mrn_csn_pair ' \
        'LEFT JOIN DISPO on mrn_csn_pairs.mrn_csn_pair = DISPO.mrn_csn_pair ' \
        'LEFT JOIN secondary_outcomes on mrn_csn_pairs.mrn_csn_pair = secondary_outcomes.mrn_csn_pair'
dat = pd.read_sql(query, processed_conn, parse_dates=True)
dat.drop_duplicates(inplace=True)

dat['los_binned'] = dat['time_in_hospital'].apply(lambda x: 2 if x >= 7*1440 else 1 if x >= 3*1440 else 0)
dat.to_sql('outcomes', merged_conn, if_exists='replace', index=False)


# Instantaneous timeseries
query = 'SELECT mrn_csn_pairs.mrn_csn_pair AS mrn_csn_pair, ' \
        'DEMOGRAPHICS.admission_datetime AS admit_datetime, ' \
        'FLOWSHEET.recorded_datetime AS timestamp, ' \
        'FLOWSHEET.Name AS measure, ' \
        'FLOWSHEET.value AS value ' \
        'FROM mrn_csn_pairs ' \
        'INNER JOIN FLOWSHEET on mrn_csn_pairs.mrn_csn_pair = FLOWSHEET.mrn_csn_pair ' \
        'INNER JOIN DEMOGRAPHICS on mrn_csn_pairs.mrn_csn_pair = DEMOGRAPHICS.mrn_csn_pair ' \
        'UNION ALL ' \
        'SELECT mrn_csn_pairs.mrn_csn_pair AS mrn_csn_pair, ' \
        'DEMOGRAPHICS.admission_datetime AS admit_datetime, ' \
        'IO_FLOWSHEET.recorded_datetime AS timestamp, ' \
        'IO_FLOWSHEET.Name AS measure, ' \
        'IO_FLOWSHEET.value AS value ' \
        'FROM mrn_csn_pairs ' \
        'INNER JOIN IO_FLOWSHEET on mrn_csn_pairs.mrn_csn_pair = IO_FLOWSHEET.mrn_csn_pair ' \
        'INNER JOIN DEMOGRAPHICS on mrn_csn_pairs.mrn_csn_pair = DEMOGRAPHICS.mrn_csn_pair ' \
        'UNION ALL ' \
        'SELECT mrn_csn_pairs.mrn_csn_pair AS mrn_csn_pair, ' \
        'DEMOGRAPHICS.admission_datetime AS admit_datetime, ' \
        'LABS.result_datetime AS timestamp, ' \
        'LABS.Name AS measure, ' \
        'LABS.value AS value ' \
        'FROM mrn_csn_pairs ' \
        'INNER JOIN LABS on mrn_csn_pairs.mrn_csn_pair = LABS.mrn_csn_pair ' \
        'INNER JOIN DEMOGRAPHICS on mrn_csn_pairs.mrn_csn_pair = DEMOGRAPHICS.mrn_csn_pair ' \
        'UNION ALL ' \
        'SELECT mrn_csn_pairs.mrn_csn_pair AS mrn_csn_pair, ' \
        'DEMOGRAPHICS.admission_datetime AS admit_datetime, ' \
        'NEURO.recorded_datetime AS timestamp, ' \
        'NEURO.Name AS measure, ' \
        'NEURO.value AS value ' \
        'FROM mrn_csn_pairs ' \
        'INNER JOIN NEURO on mrn_csn_pairs.mrn_csn_pair = NEURO.mrn_csn_pair ' \
        'INNER JOIN DEMOGRAPHICS on mrn_csn_pairs.mrn_csn_pair = DEMOGRAPHICS.mrn_csn_pair '


dat = pd.read_sql(query, processed_conn, parse_dates=['admit_datetime', 'timestamp'])
dat['timestamp'] = (dat['timestamp'] - dat['admit_datetime'])/pd.Timedelta(minutes=1)
dat.drop(columns=['admit_datetime'], inplace=True)
dat['value'] = dat['value'].astype(float)
dat.to_sql('timeseries_instantaneous', merged_conn, if_exists='replace', index=False)

# Start-stop timeseries

query = 'SELECT mrn_csn_pairs.mrn_csn_pair AS mrn_csn_pair, ' \
        'DEMOGRAPHICS.admission_datetime AS admit_datetime, ' \
        'LDA.placed_datetime AS start, ' \
        'LDA.removed_datetime AS stop, ' \
        'LDA.Name AS measure ' \
        'FROM mrn_csn_pairs ' \
        'INNER JOIN LDA on mrn_csn_pairs.mrn_csn_pair = LDA.mrn_csn_pair ' \
        'INNER JOIN DEMOGRAPHICS on mrn_csn_pairs.mrn_csn_pair = DEMOGRAPHICS.mrn_csn_pair ' \
        'UNION ALL ' \
        'SELECT mrn_csn_pairs.mrn_csn_pair AS mrn_csn_pair, ' \
        'DEMOGRAPHICS.admission_datetime AS admit_datetime, ' \
        'ADT."in" as start, ' \
        'ADT."out" as stop, ' \
        'ADT.Unit as measure ' \
        'FROM mrn_csn_pairs ' \
        'INNER JOIN ADT on mrn_csn_pairs.mrn_csn_pair = ADT.mrn_csn_pair ' \
        'INNER JOIN DEMOGRAPHICS on mrn_csn_pairs.mrn_csn_pair = DEMOGRAPHICS.mrn_csn_pair ' \
        'UNION ALL ' \
        'SELECT mrn_csn_pairs.mrn_csn_pair AS mrn_csn_pair, ' \
        'DEMOGRAPHICS.admission_datetime AS admit_datetime, ' \
        'MAR.med_admin_start_datetime AS start, ' \
        "DATETIME(MAR.med_admin_start_datetime, '+1 minutes') AS stop, " \
        'MAR.class AS measure ' \
        'FROM mrn_csn_pairs ' \
        'INNER JOIN MAR on mrn_csn_pairs.mrn_csn_pair = MAR.mrn_csn_pair ' \
        'INNER JOIN DEMOGRAPHICS on mrn_csn_pairs.mrn_csn_pair = DEMOGRAPHICS.mrn_csn_pair '

dat = pd.read_sql(query, processed_conn, parse_dates=['admit_datetime', 'start', 'stop'])

print(dat[dat['measure']=='diuretic'].head())

dat['start'] = (dat['start'] - dat['admit_datetime'])/pd.Timedelta(minutes=1)
dat['stop'] = (dat['stop'] - dat['admit_datetime'])/pd.Timedelta(minutes=1)



dat_ts = []
for mrn_csn in dat['mrn_csn_pair'].unique():
        for measure in dat['measure'].unique():
                dat_sub = dat[(dat['mrn_csn_pair'] == mrn_csn) & (dat['measure'] == measure)].sort_values('start')
                running_count = 0
                for i, r in dat_sub.iterrows():
                        dat_ts.append([mrn_csn, measure, r['start'], running_count])
                        running_count += r['stop'] - r['start']
                        dat_ts.append([mrn_csn, measure, r['stop'], running_count])
dat_ts = pd.DataFrame(dat_ts)
dat_ts.columns = ['mrn_csn_pair', 'measure', 'timestamp', 'value']
dat_ts.to_sql('timeseries_startstop', merged_conn, if_exists='replace', index=False)