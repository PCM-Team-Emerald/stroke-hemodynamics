#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[7]:


df_lab = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Labs_1007.txt",delimiter="|")


# In[9]:


df_lab.head(20)


# In[2]:


df_med = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Med.txt",delimiter="\t")


# In[8]:


df_med.shape


# In[3]:


def dtypes_with_examples(table):
    return pd.merge(table.head(1).T, table.dtypes.to_frame("Type").Type, left_index = True, right_index = True)


# In[4]:


dtypes_with_examples(df_med)


# In[5]:


df_med[["ORDERING_DATE_DEIDENTIFIED"]] = df_med[["ORDERING_DATE_DEIDENTIFIED"]].applymap(pd.Timestamp)


# In[6]:


dtypes_with_examples(df_med)


# In[30]:


len(df_med.MEDICATION_NAME.unique()) # 3032 medications prescribed; 1743 administered


# In[7]:


df_med.groupby(['MEDICATION_NAME']).size().sort_values(ascending=False).head()


# In[10]:


df_med.loc[df_med.MEDICATION_NAME.str.startswith("SODIUM CHLORIDE")]


# In[41]:


df_med.loc[df_med.MAX_DOSE.isnull() == False] # 8 unique routes + nan value in route


# In[53]:


def row_func(row):
    try:
        return float(row['DOSAGE/UNIT'].split('/')[0]) != row.MIN_DISCRETE_DOSE
    except:
        return True


# In[54]:


df_med_filtered = df_med.loc[df_med.MIN_DISCRETE_DOSE.isnull() == False]


# In[55]:


df_med_filtered.loc[df_med_filtered.apply(row_func, axis = 1)]


# In[40]:


df_med.ROUTE.unique()


# In[38]:


df_med.loc[(df_med.MIN_DISCRETE_DOSE.isnull()) & (~df_med["DOSAGE/UNIT"].isnull())]

observation: Dosage/unit is null for every null min dosage.
# In[46]:


df_med.loc[df_med.MIN_DISCRETE_DOSE == df_med["DOSAGE/UNIT"]] # to check if the values are same, coz then the 2 columns are redundant

Observation : The numerical values are same + the DOSAGE column contains more information about unit of measurement. Maybe min dosage column can be eliminated
# In[23]:


has_max_dose_no_min = df_med.loc[(df_med.MIN_DISCRETE_DOSE.isnull()) & (~df_med.MAX_DOSE.isnull())]


# In[28]:


has_max_dose_no_min.MEDICATION_NAME.unique() # has 7 such medications

