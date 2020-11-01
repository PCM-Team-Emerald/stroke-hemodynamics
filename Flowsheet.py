#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df_flowsheet = pd.read_csv("S:/Dehydration_stroke/CCDA_Data/flowsheet.txt",delimiter="\t")


# In[3]:


len(df_flowsheet.groupby(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"]))


# In[5]:


len(df_flowsheet.MRN_DEIDENTIFIED.unique())


# In[4]:


len(df_flowsheet.loc[df_flowsheet.TEMPLATE_NAME == "T JHH NEURO PHYSICIAN STROKE DOCUMENTATION"])# R JHH NEURO NIHSS UPON ARRIVAL (REPORTED) be a single number bet 0-42; and they should be timestamped to know if they were single or had multpiple 


# In[12]:


# 'T AD ED ASSESS NEURO'
# 'R JHH NEURO NIHSS UPON ARRIVAL (REPORTED)'
# 'R JHH NEURO NIHSS DATE'] )
df_flowsheet.loc[df_flowsheet.TEMPLATE_NAME == 'R JHH NEURO NIHSS UPON ARRIVAL (REPORTED)'] 


# In[3]:


df_flowsheet.head()


# In[15]:


df_flowsheet.shape


# In[5]:


#df_flowsheet.set_index(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"])


# In[6]:


df_flowsheet.groupby(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"]).size().sort_values()


# In[7]:


df_flowsheet.dtypes


# In[22]:


def dtypes_with_examples(table):
    return pd.merge(table.head(1).T, table.dtypes.to_frame('Type').Type, left_index = True, right_index = True)


# In[23]:


dtypes_with_examples(df_flowsheet)


# In[24]:


df_flowsheet[['RECORDED_DATE_TIME_DEIDENTIFIED']] = df_flowsheet[['RECORDED_DATE_TIME_DEIDENTIFIED']].applymap(pd.Timestamp)
# df_flowsheet['RECORDED_DATE_TIME_DEIDENTIFIED'] = df_flowsheet['RECORDED_DATE_TIME_DEIDENTIFIED'].apply(pd.Timestamp) this will work as we are passing only one column and it's a series so its applied to every item of the column. While passing multiple columns, need to pass them as a list ini df[] as list itself is a series
# use apply for series; applymap for df;
#df is a collection of series; series is a collection of items; apply on series applies to individual items on series; applying on a df applies to each column ie. series. Also, apply can also be used for rows instead of columns by using axis=1 parameter. 


# In[25]:


dtypes_with_examples(df_flowsheet)


# In[13]:


list=[]
list.append(df_flowsheet.TEMPLATE_NAME.unique())
for x in zip(*list):
    print (x)
len(df_flowsheet.TEMPLATE_NAME.unique())


# In[14]:


df_flowsheet.FLOWSHEET_ROW_NAME.unique()


# In[15]:


len(df_flowsheet.FLOWSHEET_ROW_NAME.unique())


# In[29]:


import matplotlib.pyplot as plt
print('Histogram of vital values')


a = df_flowsheet.loc[df_flowsheet.FLOWSHEET_ROW_NAME =='TEMPERATURE']["MEAS_VALUE"].apply(float)
plt.hist(a,bins=int(len(df_flowsheet.loc[(df_flowsheet.FLOWSHEET_ROW_NAME== 'TEMPERATURE')]["MEAS_VALUE"].unique())))


                                                                                                                                                 
                                                                                                                                                 


# In[27]:


a[5]


# In[ ]:





# In[16]:


df_flowsheet.groupby(['MRN_DEIDENTIFIED', 'CSN_DEIDENTIFIED']).size().sort_values(ascending = False).head()


# In[17]:


df_flowsheet.loc[(df_flowsheet.MRN_DEIDENTIFIED == 277) & (df_flowsheet.CSN_DEIDENTIFIED == 72)].groupby("FLOWSHEET_ROW_NAME").size()


# In[18]:


df_flowsheet.loc[(df_flowsheet.MRN_DEIDENTIFIED == 277) & (df_flowsheet.CSN_DEIDENTIFIED == 72) & (df_flowsheet.FLOWSHEET_ROW_NAME =="BLOOD PRESSURE")]


# In[19]:


# sorting according to time
df_flowsheet = df_flowsheet.sort_values(['MRN_DEIDENTIFIED', 'CSN_DEIDENTIFIED','RECORDED_DATE_TIME_DEIDENTIFIED']).reset_index(drop = True)
df_fs_277 = df_flowsheet.loc[(df_flowsheet.MRN_DEIDENTIFIED == 277) & (df_flowsheet.CSN_DEIDENTIFIED == 72)]


# In[20]:


#df_fs_277.loc[:, ['BLOOD PRESSURE:SYS']] = 
df_fs_277.loc[(df_fs_277.FLOWSHEET_ROW_NAME =="BLOOD PRESSURE"), ['RECORDED_DATE_TIME_DEIDENTIFIED', 'MEAS_VALUE']].set_index('RECORDED_DATE_TIME_DEIDENTIFIED').rename(columns = {'MEAS_VALUE': 'SYS'}).SYS.str.split('/').str.get(0).apply(float).plot() # apply(float) as there are values NaN; int of Nan does not work however flot of NaN does
df_fs_277.loc[(df_fs_277.FLOWSHEET_ROW_NAME =="BLOOD PRESSURE"), ['RECORDED_DATE_TIME_DEIDENTIFIED', 'MEAS_VALUE']].set_index('RECORDED_DATE_TIME_DEIDENTIFIED').rename(columns = {'MEAS_VALUE': 'DIA'}).DIA.str.split('/').str.get(1).apply(float).plot() 


# In[21]:


#JHM AN ADULT ADJUSTED BODY WEIGHT
df_fs_277.loc[(df_fs_277.FLOWSHEET_ROW_NAME =="JHM AN ADULT ADJUSTED BODY WEIGHT")]


# In[22]:


df_fs_277.loc[(df_fs_277.FLOWSHEET_ROW_NAME =="JHM AN PEDS ADJUSTED BODY WEIGHT")]


# ![image.png](attachment:image.png)
# 

# As the patient has both, adult and peds adjusted body weight. As seen above both the values were measured 4 times at the same timestamps. As the patient is 61 years old, I don't think we need PEDS ADJUSTED BODY WEIGHT.
# 

# In[23]:


df_fs_277.groupby('RECORDED_DATE_TIME_DEIDENTIFIED').size().sort_values(ascending = False).head(10)


# In[24]:


df_fs_277.loc[(df_fs_277.RECORDED_DATE_TIME_DEIDENTIFIED == "2017-05-05 08:00:00")]


# In[63]:


df_flowsheet.groupby('FLOWSHEET_ROW_NAME')['MRN_DEIDENTIFIED'].unique().apply(len).sort_values(ascending = False)


# In[26]:


#df_flowsheet.groupby(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED']).size().shape
df_flowsheet.groupby(['MRN_DEIDENTIFIED']).size().shape


# The data is present for only a small subset of the total patients (unique mrn) present in demographics.txt. ![image.png](attachment:image.png)
# 

# In[27]:


df_flowsheet.groupby('FLOWSHEET_ROW_NAME')['MRN_DEIDENTIFIED'].unique().apply(len).sort_values(ascending = False).hist(bins = 50)


# In[11]:


df_flowsheet.groupby(["FLOWSHEET_ROW_NAME"]).MRN_DEIDENTIFIED.unique().apply(len).sort_values(ascending = False).head(60).tail(30).plot.bar(figsize = (20,10))


# In[28]:


df_flowsheet.loc[df_flowsheet.FLOWSHEET_ROW_NAME=='R PERCENT WEIGHT CHANGE SINCE BIRTH','MEAS_VALUE'].unique()


# This field is present in 1086 patients, yet the values are of no significance.
# We need to filter out fields that contain relevant/significant data (values). To do this we first find all the unique measured values per field. We then remove any value "nan" as the record is missing/incomplete; this value is not dropped, this is just for the display.

# In[29]:


#pd.set_option('display.max_rows', 200)


# In[30]:


df_flowsheet.groupby('FLOWSHEET_ROW_NAME')['MEAS_VALUE'].unique().apply(lambda l: [x for x in l if str(x).lower() != 'nan']).apply(len)


# In[31]:


df_flowsheet.loc[df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("OXIMETRY") | df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("PULSE")].groupby("FLOWSHEET_ROW_NAME")['MEAS_VALUE'].unique().apply(len)


# Can any of the above pulse/oximetry fields be combined?
# ![image.png](attachment:image.png)

# In[5]:


df_flowsheet.loc[df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("OXIMETRY")].groupby('FLOWSHEET_ROW_NAME')['TEMPLATE_NAME'].unique()


# In[6]:


df_flowsheet.loc[df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("TEMP")].groupby('FLOWSHEET_ROW_NAME')['TEMPLATE_NAME'].unique()


# In[16]:


# drop all rows with R JHH IP ECMO WATER BATH TEMPERATURE (C)
df_flowsheet = df_flowsheet.loc[df_flowsheet.FLOWSHEET_ROW_NAME != 'R JHH IP ECMO WATER BATH TEMPERATURE (C)']                                                     


# In[26]:


#converting the temperature to a uniform unit of measurement
df_temp = df_flowsheet.loc[df_flowsheet.FLOWSHEET_ROW_NAME.str.contains("TEMP")]


# In[30]:


df_flowsheet.shape


# In[48]:


df_temp.MEAS_VALUE.unique()


# In[49]:


def check_if_number(x):
    try:
        float(x)
        return True
    except:
        return False


# In[51]:


df_temp.loc[~df_temp.MEAS_VALUE.apply(check_if_number)]


# In[53]:


df_temp.loc[(df_temp.FLOWSHEET_ROW_NAME == "R JHM IP NEUROVASCULAR RLE TEMPERATURE") & (df_temp.MEAS_VALUE.apply(check_if_number))]


# OBSERVATION:
# FLOWSHEET_ROW_NAME = R JHM IP NEUROVASCULAR RLE TEMPERATURE has no valid numerical temperatures, only adjectives like "warm","dry","clammy",etc or NaN

# In[54]:


df_temp.FLOWSHEET_ROW_NAME.unique()


# In[30]:


temprature_labels =['R APACHE TEMPERATURE',
                    'TEMPERATURE',
                    'R JHM IP VAT TEMPERATURE > 38.5 (2)',
                    'R JHM IP VAT TEMPERATURE > 38.5',
                    'R JHM IP RT LUNG TEMPERATURE',
                    'R TEMPERATURE 2'
                    ]


# In[39]:


oximetry_labels = ['PULSE OXIMETRY', 
                   'R JHH IP_RT_ PULSE OXIMETRY',
                   'R JHM OP OT NEW PULSE OXIMETRY',
                   '$ R BMC HC PULSE OXIMETRY',
                   'R ED TRAUMA PRE-ARRIVAL PULSE OXIMETRY-REPEAT',
                   'R JHM ED PRE-ARRIVAL PULSE OXIMETRY', 
                   'R PULSE OXIMETRY TYPE',
                   'R ED TRAUMA PRE-ARRIVAL PULSE OXIMETRY',
                   '$ R JHM IP RT PULSE OXIMETRY-SINGLE READING',
                   'IP_RT_ PULSE OXIMETRY']
#(df_flowsheet.loc[df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("OXIMETRY")]["FLOWSHEET_ROW_NAME"].unique())


# In[69]:


df_flowsheet.loc[df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("BLOOD PRESSURE")]["FLOWSHEET_ROW_NAME"].unique()


# In[46]:


df_flowsheet.loc[df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("HEART RATE")]["FLOWSHEET_ROW_NAME"].unique()


# In[45]:


blood_pressure_labels =['BLOOD PRESSURE',
                        'R JHM OP OT NEW BLOOD PRESSURE',
                        'R ARTERIAL LINE BLOOD PRESSURE',
                        'R ARTERIAL LINE BLOOD PRESSURE 2', 
                        'R ARTERIAL BLOOD PRESSURE 2',
                        'R AN MEAN ARTERIAL BLOOD PRESSURE 2',
                        'R BMC IP RT BLOOD PRESSURE', 
                        'R ARTERIAL LINE BLOOD PRESSURE 3',
                        'R JHM OR HIGH BLOOD PRESSURE', 
                        'R JHM REC THERAPY BLOOD PRESSURE']
# should heart rate and pulses be combined


# In[ ]:





# In[51]:


pulse_labels=df_flowsheet.loc[(df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("PULSE")) &(~df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("OXIMETRY")) ]["FLOWSHEET_ROW_NAME"].unique().tolist()


# In[29]:


(df_temp.FLOWSHEET_ROW_NAME == 'R JHM IP RT TEMPERATURE HEATED HIGH FLOW').unique()


# In[52]:


vitals = pulse_labels + temprature_labels + oximetry_labels + blood_pressure_labels


# In[58]:


df_flowsheet.loc[df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("NIHSS")]["FLOWSHEET_ROW_NAME"].unique()


# In[61]:


df_flowsheet.loc[((df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("NEURO")) &(~df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("NIHSS"))) |(df_flowsheet.FLOWSHEET_ROW_NAME.str.upper().str.contains("STROKE")) ]["FLOWSHEET_ROW_NAME"].unique().tolist()


# In[56]:


df_temp.loc[~df_temp.MEAS_VALUE.apply(check_if_number)].groupby(["FLOWSHEET_ROW_NAME"]).MEAS_VALUE.unique()


# In[60]:


df_temp.loc[(df_temp.FLOWSHEET_ROW_NAME.str.contains("VAT")) & (df_temp.MEAS_VALUE == "Yes")]


# In[40]:


def save_state(df, name):
    df.to_pickle('S:/Dehydration_stroke/IntermediateData/{}_{}.pickle'.format(name, 
                                                                              pd.Timestamp.now().strftime('%Y%m%d%H%M%S')))
save_state(df_temp, 'df_temp')
save_state(df_flowsheet, 'df_flowsheet')


# In[67]:


df_flowsheet.loc[df_flowsheet.TEMPLATE_NAME.str.upper().str.contains("STROKE")].groupby(["TEMPLATE_NAME", "FLOWSHEET_ROW_NAME"]).size().sort_values(ascending = False)

OBSERVATIONS: 
For FLOWSHEET_ROW_NAME:
R JHH IP ECMO WATER BATH TEMPERATURE (C) is not related to bidy temp, has value = OFF everywhere. Can be ignored
R JHM IP VAT TEMPERATURE > 38.5 : what is VAT?
what is the difference between R JHM IP VAT TEMPERATURE > 38.5 and R JHM IP VAT TEMPERATURE > 38.5(2). Together they have 120 rows. For value = Yes, there are 5 entries. All (2) values correspond to same MRN and CSN as (1) and MEAS_VALUE are same. I think the field can be ignored
None of the NEUROVASLCULAR temp. have a valid numerical value

# In[68]:


df_flowsheet


# In[ ]:




