#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[27]:


raw_data = pd.read_csv('S:\\Dehydration_stroke\\CCDA_Data\\flowsheet_11022020.txt',delimiter='|')
raw_data_demographics = pd.read_csv('S:\\Dehydration_stroke\\CCDA_Data\\demographics.txt',delimiter='|')


# In[28]:


#add discharge data to raw_data from flowsheet
discharge = raw_data_demographics[['mrn_deidentified','CSN_DEIDENTIFIED','DISCHARGE_DATE_DEIDENTIFIED']]


# In[5]:


#isolate pulse data
raw_data_pulse=raw_data.drop(raw_data[raw_data.FLOWSHEET_ROW_NAME != 'PULSE'].index)
raw_data_pulse.head()
raw_data_379 = raw_data.drop(raw_data[raw_data.MRN_DEIDENTIFIED != 379].index)
#raw_data_379['FLOWSHEET_ROW_NAME'].value_counts()


# In[6]:


#isolate pulse data specific to patient 379
raw_data_pulse_379=raw_data_pulse.drop(raw_data_pulse[raw_data_pulse.MRN_DEIDENTIFIED != 379].index)
#find time since admission
raw_data_pulse_379['RECORDED_DATE_TIME_DEIDENTIFIED'] = pd.to_datetime(raw_data_pulse_379['RECORDED_DATE_TIME_DEIDENTIFIED'])
def timeinmin(time):
    days=time.days
    minutes=time.seconds//60
    minutes += days*60*24
    return minutes
#find admission time and subtract others from that time
t0 = raw_data_pulse_379['RECORDED_DATE_TIME_DEIDENTIFIED'].min()
raw_data_pulse_379['RECORDED_DATE_TIME_DEIDENTIFIED'] = raw_data_pulse_379['RECORDED_DATE_TIME_DEIDENTIFIED'] - t0
raw_data_pulse_379['RECORDED_DATE_TIME_DEIDENTIFIED'] = raw_data_pulse_379['RECORDED_DATE_TIME_DEIDENTIFIED'].apply(timeinmin)
#drop NaN and nonsensical time values
raw_data_pulse_379 = raw_data_pulse_379.dropna()
raw_data_pulse_379=raw_data_pulse_379.drop(raw_data_pulse_379[raw_data_pulse_379.RECORDED_DATE_TIME_DEIDENTIFIED < 0].index)

#parse all data points to int so they can be plotted
raw_data_pulse_379.RECORDED_DATE_TIME_DEIDENTIFIED = pd.to_numeric(raw_data_pulse_379.RECORDED_DATE_TIME_DEIDENTIFIED)
raw_data_pulse_379.MEAS_VALUE = pd.to_numeric(raw_data_pulse_379.MEAS_VALUE)
raw_data_pulse_379 = raw_data_pulse_379.sort_values('RECORDED_DATE_TIME_DEIDENTIFIED',ascending=True)

#remove nonsencial pulse values
raw_data_pulse_379=raw_data_pulse_379.drop(raw_data_pulse_379[raw_data_pulse_379.MEAS_VALUE == 0].index)

#change minutes to days
raw_data_pulse_379.RECORDED_DATE_TIME_DEIDENTIFIED = raw_data_pulse_379['RECORDED_DATE_TIME_DEIDENTIFIED']/1440


#plot figure
figure=plt.figure(figsize=(8,8))
raw_data_pulse_379.plot(x='RECORDED_DATE_TIME_DEIDENTIFIED',y='MEAS_VALUE',color='g')
plt.xlabel('Time Since Admission (Days)')
plt.ylabel('Pulse(bpm)')
plt.title('Pulse Time Series Data for a Sample Patient')
plt.legend([])
plt.savefig('S:\Dehydration_stroke\Team Emerald\scripts\Michael Working Scripts\Michael Figures\PulseTimeSeries.png')


# In[9]:


#Find TIME Distribution
time = raw_data_pulse_379['RECORDED_DATE_TIME_DEIDENTIFIED']
time = np.asarray(time)
time = time[1:100]
y = np.ones(len(time))*5
plt.scatter(time,y)
plt.xlabel('Time(days)')
plt.legend(['Measurement'])


# In[10]:


# PLOTTING BP TIME SERIES DATA
raw_data_bp=raw_data.drop(raw_data[raw_data.FLOWSHEET_ROW_NAME != 'BLOOD PRESSURE'].index)
raw_data_bp_379=raw_data_bp.drop(raw_data_bp[raw_data_bp.MRN_DEIDENTIFIED != 379].index)
raw_data_bp_379['RECORDED_DATE_TIME_DEIDENTIFIED'] = pd.to_datetime(raw_data_bp_379['RECORDED_DATE_TIME_DEIDENTIFIED'])
raw_data_bp_379['MEAS_VALUE'] = raw_data_bp_379['MEAS_VALUE'].astype(str)
raw_data_bp_379 = raw_data_bp_379[raw_data_bp_379.MEAS_VALUE.str.contains('/',na=False)]

#print(raw_data_bp_379.dtypes)

def systolic(string):
    systolic =  list(map(int, string.split('/')))
    systolicbp = systolic[0]
    return systolicbp
def diastolic(string):
    diastolic =  list(map(int, string.split('/')))
    diastolicbp = diastolic[1]
    return diastolicbp

raw_data_bp_379['Systolic_BP'] = raw_data_bp_379['MEAS_VALUE'].apply(systolic)
raw_data_bp_379['Diastolic_BP'] = raw_data_bp_379['MEAS_VALUE'].apply(diastolic)

t0 = raw_data_bp_379['RECORDED_DATE_TIME_DEIDENTIFIED'].min()
raw_data_bp_379['RECORDED_DATE_TIME_DEIDENTIFIED'] = raw_data_bp_379['RECORDED_DATE_TIME_DEIDENTIFIED'] - t0
raw_data_bp_379['RECORDED_DATE_TIME_DEIDENTIFIED'] = raw_data_bp_379['RECORDED_DATE_TIME_DEIDENTIFIED'].apply(timeinmin)
raw_data_bp_379=raw_data_bp_379.drop(raw_data_bp_379[raw_data_bp_379.RECORDED_DATE_TIME_DEIDENTIFIED < 0].index)
raw_data_bp_379.RECORDED_DATE_TIME_DEIDENTIFIED = pd.to_numeric(raw_data_bp_379.RECORDED_DATE_TIME_DEIDENTIFIED)
raw_data_bp_379.Systolic_BP = pd.to_numeric(raw_data_bp_379.Systolic_BP)
raw_data_bp_379.Diastolic_BP = pd.to_numeric(raw_data_bp_379.Diastolic_BP)
raw_data_bp_379 = raw_data_bp_379.sort_values('RECORDED_DATE_TIME_DEIDENTIFIED',ascending=True)
raw_data_bp_379.RECORDED_DATE_TIME_DEIDENTIFIED = raw_data_bp_379['RECORDED_DATE_TIME_DEIDENTIFIED']/1440
raw_data_bp_379 = raw_data_bp_379.dropna()

figure=plt.figure(figsize=(8,8))
raw_data_bp_379.plot(x='RECORDED_DATE_TIME_DEIDENTIFIED',y=['Systolic_BP','Diastolic_BP'],color=['g','r'])
plt.xlabel('Time Since Admission (Days)')
plt.ylabel('Blood Pressure (mmHg)')
plt.title('Blood Pressure Time Series Data for a Sample Patient')
plt.legend(['Systolic BP','Diastolic BP'])
plt.savefig('S:\Dehydration_stroke\Team Emerald\scripts\Michael Working Scripts\Michael Figures\BPTimeSeries.png')

plt.show()



# In[25]:


# Inclusion/Exclusion How many have over 24hr of vitals data?
raw_data_vitals = raw_data.drop(raw_data[(raw_data.FLOWSHEET_ROW_NAME != 'PULSE') & (raw_data.FLOWSHEET_ROW_NAME != 'BLOOD PRESSURE') & (raw_data.FLOWSHEET_ROW_NAME != 'PULSE OXIMETRY')].index)
raw_data_vitals = raw_data_vitals.sort_values('MRN_DEIDENTIFIED',ascending=True)
raw_data_vitals = raw_data_vitals.sort_values('CSN_DEIDENTIFIED',ascending=True)
raw_data_vitals = raw_data_vitals.drop(raw_data_vitals[raw_data_vitals.FLOWSHEET_ROW_NAME != 'PULSE'].index)
raw_data_vitals['mrn_csn_pair'] = list(zip(raw_data_vitals.MRN_DEIDENTIFIED,raw_data_vitals.CSN_DEIDENTIFIED))
mrn_csn = raw_data_vitals.mrn_csn_pair.unique()


pat_mrn_csn = []
count=0


for i in mrn_csn:
    count+=1
    #print(count)
    raw_data_vitals_i = raw_data_vitals.drop(raw_data_vitals[(raw_data_vitals.MRN_DEIDENTIFIED != i[0]) & (raw_data_vitals.CSN_DEIDENTIFIED != i[1])].index)
    raw_data_vitals_i['RECORDED_DATE_TIME_DEIDENTIFIED'] = pd.to_datetime(raw_data_vitals_i['RECORDED_DATE_TIME_DEIDENTIFIED'])
    t0 = raw_data_vitals_i['RECORDED_DATE_TIME_DEIDENTIFIED'].min()
    tmax = raw_data_vitals_i['RECORDED_DATE_TIME_DEIDENTIFIED'].max()
    timedelta = tmax-t0
    timedelta = timeinmin(timedelta)
    if timedelta >= 1440:
        pat_mrn_csn.append(i)
    


# In[33]:


#Find number of patients with mobility AMPAC data within 24 hours of discharge
def num_pats_feature(feature):
    raw_data_feature = raw_data[raw_data.FLOWSHEET_ROW_NAME.str.contains(feature,na=False)]
    raw_data_feature = raw_data_feature.sort_values('MRN_DEIDENTIFIED',ascending=True)
    raw_data_feature['mrn_csn_pair'] = list(zip(raw_data_feature.MRN_DEIDENTIFIED,raw_data_feature.CSN_DEIDENTIFIED))
    mrn_csn = raw_data_feature.mrn_csn_pair.unique()
    count=0;
    discharge['DISCHARGE_DATE_DEIDENTIFIED'] = pd.to_datetime(discharge['DISCHARGE_DATE_DEIDENTIFIED'])
    mrn_csn_feature = [];


    for i in mrn_csn:
        count+=1
        #print(count)
        raw_data_feature_i = raw_data_feature.drop(raw_data_feature[raw_data_feature.MRN_DEIDENTIFIED != i[0]].index)
        raw_data_feature_i = raw_data_feature_i.drop(raw_data_feature_i[raw_data_feature_i.CSN_DEIDENTIFIED != i[1]].index)
        raw_data_feature_i['RECORDED_DATE_TIME_DEIDENTIFIED'] = pd.to_datetime(raw_data_feature_i['RECORDED_DATE_TIME_DEIDENTIFIED'])
        raw_data_feature_i = raw_data_feature_i.sort_values('RECORDED_DATE_TIME_DEIDENTIFIED',ascending=True)
    
        discharge_i = discharge.drop(discharge[discharge.mrn_deidentified != i[0]].index)
        discharge_i = discharge_i.drop(discharge_i[discharge_i.CSN_DEIDENTIFIED != i[1]].index)
        discharge_i = discharge_i.reset_index(drop=True)
        timedischarge = discharge_i.at[0,'DISCHARGE_DATE_DEIDENTIFIED']
        timelast = raw_data_feature_i['RECORDED_DATE_TIME_DEIDENTIFIED'].max()
        timedelta = timedischarge-timelast
        timedelta = timeinmin(timedelta)
        if((timedelta <= 1440) & (timedelta > 0) & (timedelta != 'nan')):
            mrn_csn_feature.append(i)
    return mrn_csn_feature

mrn_csn_ampacm = num_pats_feature('R JHM PT AMPAC MOBILITY INPT T SCORE')
mrn_csn_ampaca = num_pats_feature('R JHM OT AMPAC DAILY ACTIVITY INPT T SCORE')


# In[34]:


print(len(mrn_csn_ampacm))
print(len(mrn_csn_ampaca))


# In[32]:


#similar extraction for GCS data
gcs_data = pd.read_csv('S:\\Dehydration_stroke\\Team Emerald\\Working Data\\CCDA_Data\\flowsheet_neu_am_11132020.txt',delimiter='|')
gcs_data.head()
raw_data_gcs = gcs_data[gcs_data.RESULT_VALUE_TEXT.str.contains('R CPN GLASGOW COMA SCALE SCORE',na=False)]
raw_data_gcs = raw_data_gcs.sort_values('MRN_DEIDENTIFIED',ascending=True)
raw_data_gcs['mrn_csn_pair'] = list(zip(raw_data_gcs.MRN_DEIDENTIFIED,raw_data_gcs.CSN_DEIDENTIFIED))
mrn_csn = raw_data_gcs.mrn_csn_pair.unique()
count=0;
mrn_csn_gcs = [];

for i in mrn_csn:
    count+=1
    #print(count)
    raw_data_gcs_i = raw_data_gcs.drop(raw_data_gcs[raw_data_gcs.MRN_DEIDENTIFIED != i[0]].index)
    raw_data_gcs_i = raw_data_gcs_i.drop(raw_data_gcs_i[raw_data_gcs_i.CSN_DEIDENTIFIED != i[1]].index)
    raw_data_gcs_i['RESULT_DATE_TIME_DEIDENTIFIED'] = pd.to_datetime(raw_data_gcs_i['RESULT_DATE_TIME_DEIDENTIFIED'])
    raw_data_gcs_i = raw_data_gcs_i.sort_values('RESULT_DATE_TIME_DEIDENTIFIED',ascending=True)


    discharge_i = discharge.drop(discharge[discharge.mrn_deidentified != i[0]].index)
    discharge_i = discharge_i.drop(discharge_i[discharge_i.CSN_DEIDENTIFIED != i[1]].index)
    discharge_i = discharge_i.reset_index(drop=True)
    timedischarge = discharge_i.at[0,'DISCHARGE_DATE_DEIDENTIFIED']
    timelastgcs = raw_data_gcs_i['RESULT_DATE_TIME_DEIDENTIFIED'].max()
    timedelta = timedischarge-timelastgcs
    timedelta = timeinmin(timedelta)
    if((timedelta <= 1440) & (timedelta > 0) & (timedelta != 'nan')):
        mrn_csn_gcs.append(i)

    


# In[34]:


print(len(mrn_csn))
print(len(mrn_csn_gcs))


# In[20]:


#Time hisstogram, using pulse as representative
def average_times(feature):
    raw_data_feature=raw_data.drop(raw_data[raw_data.FLOWSHEET_ROW_NAME != feature].index)
    raw_data_feature = raw_data_feature.sort_values('MRN_DEIDENTIFIED',ascending=True)
    raw_data_feature['mrn_csn_pair'] = list(zip(raw_data_feature.MRN_DEIDENTIFIED,raw_data_feature.CSN_DEIDENTIFIED))
    mrn_csn = raw_data_feature.mrn_csn_pair.unique()
    count=0;
    avg_time = [];
    #median_time=[];
    #mrn_csn_abnormalmedian=[];
    mrn_csn_abnormalmean=[];

    for i in mrn_csn:
        raw_data_pulse_i = raw_data_pulse.drop(raw_data_pulse[raw_data_pulse.MRN_DEIDENTIFIED != i[0]].index)
        raw_data_pulse_i = raw_data_pulse_i.drop(raw_data_pulse_i[raw_data_pulse_i.CSN_DEIDENTIFIED != i[1]].index)
        raw_data_pulse_i['RECORDED_DATE_TIME_DEIDENTIFIED'] = pd.to_datetime(raw_data_pulse_i['RECORDED_DATE_TIME_DEIDENTIFIED'])
        t0 = raw_data_pulse_i['RECORDED_DATE_TIME_DEIDENTIFIED'].min()
        raw_data_pulse_i['RECORDED_DATE_TIME_DEIDENTIFIED'] = raw_data_pulse_i['RECORDED_DATE_TIME_DEIDENTIFIED'] - t0
        raw_data_pulse_i['RECORDED_DATE_TIME_DEIDENTIFIED'] = raw_data_pulse_i['RECORDED_DATE_TIME_DEIDENTIFIED'].apply(timeinmin)
        raw_data_pulse_i = raw_data_pulse_i.dropna()
        raw_data_pulse_i=raw_data_pulse_i.drop(raw_data_pulse_i[raw_data_pulse_i.RECORDED_DATE_TIME_DEIDENTIFIED < 0].index)
        raw_data_pulse_i.RECORDED_DATE_TIME_DEIDENTIFIED = pd.to_numeric(raw_data_pulse_i.RECORDED_DATE_TIME_DEIDENTIFIED)
        raw_data_pulse_i = raw_data_pulse_i.sort_values('RECORDED_DATE_TIME_DEIDENTIFIED',ascending=True)
        time_i = raw_data_pulse_i['RECORDED_DATE_TIME_DEIDENTIFIED']
        time_i = np.asarray(time_i)
        diffs_i = np.diff(time_i)
        avg_i = np.mean(diffs_i)
        avg_time.append(avg_i)
        #median_i = np.median(diffs_i)
        #median_time.append(median_i)
        if (avg_i > 480):
            mrn_csn_abnormalmean.append(i)
    output = {'Time Averages': avg_time, 'MRN_CSN Abnormal': mrn_csn_abnormalmean}
    return output


# In[21]:


pulse_hist_avg = average_times('PULSE')
bp_hist_avg = average_times('BLOOD PRESSURE')
po_hist_avg = average_times('PULSE OXIMETRY')


# In[ ]:





# In[ ]:




