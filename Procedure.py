#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_proc = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Procedure.txt",delimiter="\t")


# In[3]:


df_proc.shape


# In[4]:


def dtypes_with_examples(table):
    return pd.merge(table.head(1).T, table.dtypes.to_frame("Type").Type, left_index = True, right_index = True)


# In[5]:


dtypes_with_examples(df_proc)


# In[6]:


df_proc[["ORDER_TIME_DEIDENTIFIED","RESULT_TIME_DEIDENTIFIED","PROC_START_TIME_DEIDENTIFIED",
         "PROC_ENDING_TIME_DEIDENTIFIED"]] = df_proc[["ORDER_TIME_DEIDENTIFIED","RESULT_TIME_DEIDENTIFIED","PROC_START_TIME_DEIDENTIFIED",
         "PROC_ENDING_TIME_DEIDENTIFIED"]].applymap(pd.Timestamp)


# In[7]:


dtypes_with_examples(df_proc)


# In[9]:


df_proc.loc[((df_proc.PROC_START_TIME_DEIDENTIFIED) == (df_proc.PROC_ENDING_TIME_DEIDENTIFIED))]


# In[8]:


df_proc.loc[((df_proc.PROC_START_TIME_DEIDENTIFIED) != (df_proc.PROC_ENDING_TIME_DEIDENTIFIED))]


# In[44]:


df_proc.loc[((df_proc.PROC_START_TIME_DEIDENTIFIED) != (df_proc.PROC_ENDING_TIME_DEIDENTIFIED)) & (df_proc.PROC_NAME =="CT HEAD/BRAIN WO CONTRAST")]


# In[45]:


df_proc.groupby(["PROC_NAME"]).size().sort_values(ascending = False).head(50)


# In[46]:


df_proc.PROC_NAME.unique()#48 unique labels


# In[47]:


df_proc.loc[df_proc.PROC_NAME == "PET/CT SKULL BASE TO MID-THIGH INITIAL STAGING AND CT W/IV CON"]


# In[48]:


df_proc.loc[df_proc.PROC_NAME == "PET/CT BRAIN FDG"]


# In[49]:


df_proc.groupby(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"]).size() # there are 2594 unique MRN and CSN pairs


# In[56]:


df_proc.groupby(["MRN_DEIDENTIFIED"]).size().shape


# In[59]:


import matplotlib as plt


# In[75]:


df_proc.groupby(["PROC_NAME"]).MRN_DEIDENTIFIED.unique().apply(len).sort_values(ascending = False).plot.bar(figsize = (20,10))


# In[76]:


import pandasql


# In[77]:


get_ipython().system('pip install pandasql')


# In[ ]:




