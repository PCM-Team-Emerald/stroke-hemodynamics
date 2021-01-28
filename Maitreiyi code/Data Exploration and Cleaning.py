#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df_demographics = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\demographics.txt",delimiter="|")


# In[2]:


len(df_demographics.mrn_deidentified.unique())


# In[5]:


len(df_demographics.groupby(["mrn_deidentified", "CSN_DEIDENTIFIED"]))


# In[9]:


len(df_ref.MRN_DEIDENTIFIED.unique())


# In[10]:


len(df_ref.groupby(["MRN_DEIDENTIFIED", "CSN_DEIDENTIFIED"]))


# In[2]:


pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Procedure.txt",delimiter="\t",skiprows = lambda x:x>50)


# In[3]:


pd.read_csv("S:\Dehydration_stroke\CCDA_Data\IO_flowsheet.txt",delimiter="|",skiprows = lambda x: x>5)


# In[4]:


pd.read_csv("S:\Dehydration_stroke\CCDA_Data\LDA.txt",delimiter="\t", encoding = "ISO-8859-1")


# In[5]:


pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Med.txt",delimiter="\t")


# In[6]:


id_cols = ['MRN_DEIDENTIFIED', 'CSN_DEIDENTIFIED']


# In[6]:


#df_lab=pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Lab.txt",delimiter="\t")
#df_IO_flowsheet = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\IO_flowsheet.txt",delimiter="|")
df_demographics = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\demographics.txt",delimiter="|")
#df_Dx = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Dx.txt",delimiter="|",encoding = "ISO-8859-1")
#df_feeding =pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Feeding.txt",delimiter="\t")
#df_lda = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\LDA.txt",delimiter="\t", encoding = "ISO-8859-1")
#df_mar = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\MAR.txt",delimiter="\t")
#df_med = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Med.txt",delimiter="\t")
df_ref = pd.read_csv("S:/Dehydration_stroke/CCDA_Data/referral.txt",delimiter="\t") 
#df_proc = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Procedure.txt",delimiter="\t")
#df_med_his = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Med_HX.txt",delimiter="\t")   
df_adt = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\ADT_Transfer.txt",delimiter="\t")
#df_flowsheet = pd.read_csv("S:/Dehydration_stroke/CCDA_Data/flowsheet.txt",delimiter="|")
#df_cogs = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Dx.txt",delimiter="\")


# In[8]:


df_flowsheet = pd.read_csv("S:/Dehydration_stroke/CCDA_Data/flowsheet.txt",delimiter="\t")


# In[9]:


files = [df_lab,df_IO_flowsheet,df_demographics,df_Dx,df_feeding,df_mar,df_lda,df_med,df_ref,df_proc,df_med_his,df_adt]
for df in files:
    print(any(["MRN" in x.upper() for x in df.columns]))


# In[ ]:


files = [df_lab,df_IO_flowsheet,df_demographics,df_Dx,df_feeding,df_mar,df_lda,df_med,df_ref,df_proc,df_med_his,df_adt]
for df in files:
    print(any(["CSN" in x.upper() for x in df.columns]))


# In[ ]:


df_lab.head(20)


# In[ ]:


df1 = df_lab.set_index(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED'])


# In[ ]:


print (df1.index.is_unique)


# In[ ]:


df1.groupby(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED']).size().reset_index().rename(columns={0 : 'count'})


# In[ ]:


df2.groupby(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED']).size().to_frame('count')
# df2.groupby(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED'])['MEAS_VALUE'].count()


# In[ ]:


df2= df_IO_flowsheet.set_index(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED'])


# In[ ]:


print (df2.index.is_unique)


# df3 = df_demographics.set_index(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED'])
# print (df3.index.is_unique)
# df2.groupby(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED']).size().reset_index().rename(columns={0 : 'count'})

# ## Demographics

# In[10]:


df_demographics.head(1)


# In[11]:


df3 = df_demographics.set_index(['mrn_deidentified','CSN_DEIDENTIFIED'])
print (df3.index.has_duplicates)


# In[12]:


df_demographics.shape


# In[13]:


df_demographics.columns = [x.upper() for x in df_demographics.columns] # make all column names Uppercase


# In[14]:


len(df_demographics.reset_index().MRN_DEIDENTIFIED.unique())


# In[15]:


df_demographics.groupby(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED']).size().sort_values(ascending = False).head()


# In[19]:


df_demographics.loc[(df_demographics.MRN_DEIDENTIFIED == 2559) & (df_demographics.CSN_DEIDENTIFIED == 2785)].T


# In[20]:


#clean up above:
df_demographics = df_demographics.groupby(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED']).last()


# In[21]:


def dtypes_with_example(table):
    return pd.merge(table.head(1).T, table.dtypes.to_frame('Type').Type, left_index = True, right_index = True)
dtypes_with_example(df_demographics)


# In[22]:


df_demographics.head(1).T


# In[23]:


#converting timestamp columns to pandas timestamp type
df_demographics[['ADMISSION_DATE_DEIDENTIFIED', 'DISCHARGE_DATE_DEIDENTIFIED', 'ED_ARRIVAL_DATE_TIME_DEIDENTIFIED']] = df_demographics[['ADMISSION_DATE_DEIDENTIFIED', 'DISCHARGE_DATE_DEIDENTIFIED', 'ED_ARRIVAL_DATE_TIME_DEIDENTIFIED']].applymap(pd.Timestamp)


# In[24]:


def dtypes_with_example(table):
    return pd.merge(table.head(1).T, table.dtypes.to_frame('Type').Type, left_index = True, right_index = True)
dtypes_with_example(df_demographics)


# In[25]:


df_demographics.loc[(277,72)]


# In[27]:


df_demographics.loc[~df_demographics.CHARLSON_COMORBITY_INDEX.isnull()]


# In[33]:


df_demographics.reset_index()


# In[35]:


df_demographics.AGEDURINGVISIT.describe()


# In[41]:


df_demographics.groupby(['AGEDURINGVISIT']).size().sort_values(ascending = True).head(20)


# In[42]:


dtypes_with_example(df_adt)


# ## ADT

# In[43]:


df_adt[['IN_DTTM_DEIDENTIFIED', 'OUT_DTTM_DEIDENTIFIED']] = df_adt[['IN_DTTM_DEIDENTIFIED', 'OUT_DTTM_DEIDENTIFIED']].applymap(pd.Timestamp)


# In[44]:


dtypes_with_example(df_adt)


# In[45]:


df_adt = df_adt.sort_values(['MRN_DEIDENTIFIED', 'CSN_DEIDENTIFIED', 'IN_DTTM_DEIDENTIFIED', 'OUT_DTTM_DEIDENTIFIED']).reset_index(drop=True)


# In[46]:


## check that all the time of the patients in the care is accounted for (end time of one location matches start time of another)
print (df_adt.loc[
    (df_adt.IN_DTTM_DEIDENTIFIED != 
     df_adt.OUT_DTTM_DEIDENTIFIED.shift(1))
].groupby(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED']).size().max())

print (df_adt.loc[
    (df_adt.IN_DTTM_DEIDENTIFIED.shift(-1) != 
     df_adt.OUT_DTTM_DEIDENTIFIED)
].groupby(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED']).size().max())

# The 1 and 1 show that the times line up perfectly 
# The only mismatch is due to pre-admission IN time in the first case and due to post admission OUT time in the second case.


# In[47]:


# Example with same start and end time (worth removing, as number of these entries per mrn/csn could be a feature)
df_adt.loc[(df_adt.MRN_DEIDENTIFIED == 523) & (df_adt.CSN_DEIDENTIFIED == 670)]


# In[48]:


# Removing redundent line
df_adt = df_adt.loc[df_adt.IN_DTTM_DEIDENTIFIED != df_adt.OUT_DTTM_DEIDENTIFIED]


# In[49]:


df_adt_cleaned = df_adt.groupby(['MRN_DEIDENTIFIED', 'CSN_DEIDENTIFIED', 'UNIT_NAME']).agg({'IN_DTTM_DEIDENTIFIED': min, 'OUT_DTTM_DEIDENTIFIED': max}).reset_index().sort_values(['MRN_DEIDENTIFIED', 'CSN_DEIDENTIFIED', 'IN_DTTM_DEIDENTIFIED', 'OUT_DTTM_DEIDENTIFIED']).set_index(['MRN_DEIDENTIFIED', 'CSN_DEIDENTIFIED', 'UNIT_NAME'])
df_adt_cleaned.head(17)


# In[ ]:


## Interesting feature could be the transition: PRE-ADMISSION -> BMC MAIN PERIOP v/s PRE-ADMISSION -> BMC EMERGENCY SERVICES	
## Even post-first-level transition could be interesting: from emergency care, which CCU/UNIT do they transfer to.


# In[50]:


df_adt.UNIT_NAME.unique()


# In[ ]:


df_adt.groupby(['UNIT_NAME']).size().sort_values(ascending = False).to_clipboard()

'BMC NEUROSCIENCES UNIT', 
       'BMC NEUROSCIENCE CCU', 'JHH ZAYED 3W', 'JHH ZAYED 12W BRU',
       'JHH ZAYED 12W', -> stroke units
# These need to be googled and replaced with some general terms like Emergency room, Burn Unit, Surgery, ICU (different kind), etc. Hopefully reduce this to <20 categories. Ideally, there should only be ~5 types of 1st transition, and similar number of 2nd transitions, so that we could have some strong signals.

# In[51]:


# interesting case. Not sure what this means:
filter_vals = df_adt.loc[df_adt.UNIT_NAME == 'LEAVE OF ABSENCE'][['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED']]
df_adt.loc[df_adt.MRN_DEIDENTIFIED.isin(filter_vals.MRN_DEIDENTIFIED) & 
           df_adt.CSN_DEIDENTIFIED.isin(filter_vals.CSN_DEIDENTIFIED) ]


# In[55]:


df_adt_cleaned.loc[(277,72)].sort_values(['IN_DTTM_DEIDENTIFIED', 'OUT_DTTM_DEIDENTIFIED'])


# ## Referral

# In[54]:


df_ref.head()


# In[56]:


df_ref.groupby(['MRN_DEIDENTIFIED', 'CSN_DEIDENTIFIED']).size().sort_values(ascending = False).head()


# In[ ]:


df_ref.set_index(id_cols).loc[(2594, 2954)]


# Multiple referrals are possible, although quite rare (as can be seen above)

# In[57]:


dtypes_with_example(df_ref)


# In[58]:


# Looks like REFERRING and REFERRAL DEPARTMENT is always NAN. Let's check
df_ref[['REFERRING_DEPARTMENT', 'REFERRAL_DEPARTMENT']].isnull().all().all()


# In[59]:


# Yes, they are always nan, so let's remove them.
df_ref.drop(['REFERRING_DEPARTMENT', 'REFERRAL_DEPARTMENT'], axis = 1, inplace = True)


# In[60]:


# also, applying pd.Timestamp to time columns:
df_ref[['ENCOUNTER_DATE_DEIDENTIFIED', 'REFERRAL_DATE_DEIDENTIFIED']] = df_ref[['ENCOUNTER_DATE_DEIDENTIFIED', 'REFERRAL_DATE_DEIDENTIFIED']].applymap(pd.Timestamp)

# also renaming column to be _ based
df_ref.rename(columns = {"REFERRAL DOCTOR SPECIALTY": "REFERRAL_DOCTOR_SPECIALTY"}, inplace = True)


# In[61]:


dtypes_with_example(df_ref)


# In[62]:


df_ref.REFERRAL_DOCTOR_SPECIALTY.unique()
# might need to group these into level of complexity, seriousness, or something like that


# In[ ]:


## That's all we can do for now. Feature engineering once we can look at all the files. 


# In[ ]:


df_IO_flowsheet.groupby(['MRN_DEIDENTIFIED', 'CSN_DEIDENTIFIED']).size().sort_values(ascending = False).head()


# In[ ]:


dtypes_with_example(df_IO_flowsheet)


# In[ ]:


df_IO_flowsheet.head() # delete all rows with meas_value =1


# In[ ]:


df_flowsheet.head()


# In[ ]:


dtypes_with_example(df_flowsheet)


# In[ ]:


list=[]
list.append(df_flowsheet.TEMPLATE_NAME.unique())
for x in zip(*list):
    print (x)
len(df_flowsheet.TEMPLATE_NAME.unique())


# Presence of a femoral pulse has been estimated to indicate a systolic blood pressure of more than 50 mmHg, as given by the 50% percentile.
# 

# In[ ]:


df_flowsheet.FLOWSHEET_ROW_NAME.unique()
#len(df_flowsheet.FLOWSHEET_ROW_NAME.unique())


# In[ ]:


len(df_flowsheet.FLOWSHEET_ROW_NAME.unique())


# In[ ]:


#filter_vals = 
#df_flowsheet.head()[df_flowsheet.MRN_DEIDENTIFIED == 277 & df_flowsheet.CSN_DEIDENTIFIED == 72]#[['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED']]
df_flowsheet.loc[(df_flowsheet.MRN_DEIDENTIFIED == 277) & (df_flowsheet.CSN_DEIDENTIFIED == 72)].groupby("FLOWSHEET_ROW_NAME").size()
 #          df_flowsheet.CSN_DEIDENTIFIED.isin(filter_vals.CSN_DEIDENTIFIED) ]


# In[ ]:


df_flowsheet.groupby(['MRN_DEIDENTIFIED', 'CSN_DEIDENTIFIED']).size().sort_values(ascending = False).head()

