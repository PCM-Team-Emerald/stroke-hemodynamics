#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_Dx = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Dx.txt",delimiter="|",encoding = "ISO-8859-1")


# In[3]:


df_Dx.head()


# In[4]:


def dtypes_with_examples(table):
    return pd.merge(table.head(1).T , table.dtypes.to_frame('Type').Type , left_index = True, right_index = True)


# In[5]:


dtypes_with_examples(df_Dx)


# In[6]:


df_Dx.groupby(['MRN_DEIDENTIFIED','CSN_DEIDENTIFIED']).size().sort_values(ascending = False).head()


# In[7]:


df_Dx.loc[(df_Dx.MRN_DEIDENTIFIED == 836) & (df_Dx.CSN_DEIDENTIFIED == 1639)]


# In[8]:


df_Dx.PRIMARY_DX_YN.unique()


# In[9]:


df_Dx["ED_Diagnosis?"].unique()


# In[10]:


df_Dx.loc[df_Dx.PRIMARY_DX_YN == "Y"]


# In[11]:


len(df_Dx.DIAGNOSIS_NAME.unique())


# In[12]:


len(df_Dx.ICD_9_CODE.unique())


# In[13]:


len(df_Dx.ICD_10_CODE.unique())


# In[14]:


df_Dx.loc[df_Dx.ICD_9_CODE.isnull()== True]


# In[15]:


df_Dx.loc[df_Dx.ICD_10_CODE.isnull()== True]


# In[16]:


ischemic_strokes = df_Dx.loc[df_Dx.DIAGNOSIS_NAME.str.lower().str.contains("ischemic")]["DIAGNOSIS_NAME"].unique().tolist()


# In[17]:


hemiplegia_strokes = df_Dx.loc[df_Dx.DIAGNOSIS_NAME.str.lower().str.contains("hemiplegia")]["DIAGNOSIS_NAME"].unique().tolist()


# In[18]:


# Intracerebral hemorrhage 
intracerebral_hemorrhage_strokes = df_Dx.loc[df_Dx.DIAGNOSIS_NAME.str.lower().str.contains("intracerebral hemorrhage")]["DIAGNOSIS_NAME"].unique().tolist()


# In[19]:


strokes = ischemic_strokes + intracerebral_hemorrhage_strokes + hemiplegia_strokes # (add hemiplegia with ischemic)


# In[20]:


# check if either of ed_diag or primary_diag are yes
df_Dx_has_yes = df_Dx.loc[(df_Dx["ED_Diagnosis?"] == "Y") | (df_Dx.PRIMARY_DX_YN == "Y")] # 4212 rows  in the result


# In[21]:


# list of all the rows where either ED_diag or Primary_diag is Yes and the diagnosis is stroke
df_Dx_has_yes.loc[~df_Dx_has_yes.DIAGNOSIS_NAME.isin(strokes)].groupby(["DIAGNOSIS_NAME"]).size().sort_values(ascending= False).to_clipboard()
                                                                      


# In[22]:


df_Dx.shape


# In[ ]:




