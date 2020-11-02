#!/usr/bin/env python
# coding: utf-8

# In[ ]:





df_med_his = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Med_HX.txt",delimiter="\t")   



# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df_mar = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\MAR.txt",delimiter="\t")


# In[4]:


len(df_mar.groupby(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"]))


# In[5]:


len(df_mar.MRN_DEIDENTIFIED.unique())


# In[48]:


df_mar.shape


# In[49]:


def dtypes_with_example(table):
    return pd.merge(table.head(1).T, table.dtypes.to_frame("Type").Type, left_index = True, right_index =True)


# In[50]:


dtypes_with_example(df_mar)


# In[63]:


df_mar[["MEDICATION_ADMINISTRATION_START_TIME_DEIDENTIFIED","MEDICATION_ADMINISTRATION_END_TIME_DEIDENTIFIED"]] = df_mar[["MEDICATION_ADMINISTRATION_START_TIME_DEIDENTIFIED","MEDICATION_ADMINISTRATION_END_TIME_DEIDENTIFIED"]].applymap(pd.Timestamp)


# In[64]:


dtypes_with_example(df_mar)


# In[53]:


df_mar.MEDICATION_NAME.unique()# 1743 unique values


# In[54]:


df_mar.groupby(["MEDICATION_NAME"]).size().sort_values(ascending = False)


# In[55]:


df_mar.groupby(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED","MEDICATION_NAME"]).size().sort_values(ascending = False).tail()


# In[56]:


df_1937_2068 = df_mar.loc[(df_mar.MRN_DEIDENTIFIED == 1937) & (df_mar.CSN_DEIDENTIFIED == 2068)]


# In[57]:


len(df_1937_2068.MEDICATION_NAME.unique())


# In[58]:


df_all_meds = df_mar.groupby(["MEDICATION_NAME"])


# In[59]:


df_all_meds[["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"]].apply(lambda x: len(np.unique(x))).sort_values()


# In[60]:


df_mar.loc[df_mar.MEDICATION_NAME.str.startswith("SODIUM CHLORIDE")]


# In[61]:


df_mar.loc[df_mar.MEDICATION_NAME.str.startswith("FUROSEMIDE")]


# In[82]:


df_med = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Med.txt",delimiter="\t")


# In[97]:


#common_list = [x for x in set(list(zip(df_mar.MRN_DEIDENTIFIED, df_mar.CSN_DEIDENTIFIED))) if x in set(list(zip(df_med.MRN_DEIDENTIFIED, df_med.CSN_DEIDENTIFIED)))]

df1 = df_mar[["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"]].groupby(["MRN_DEIDENTIFIED"])["CSN_DEIDENTIFIED"].unique().to_dict()
df2 = df_med[["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"]].groupby(["MRN_DEIDENTIFIED"])["CSN_DEIDENTIFIED"].unique().to_dict()


# In[100]:


df1


# In[108]:


df1.values()


# In[104]:


#for i in df1:
if df1.keys().isin(df2.keys()):
    print( i.keys())
else:
    print(False)
    
    


# In[125]:


df_mar.groupby(["MRN_DEIDENTIFIED"]).size().shape


# In[126]:


df_med.groupby(["MRN_DEIDENTIFIED"]).size().shape


# In[120]:


shared_items ={k:df1[k] for k in df1 if k in df2 and len(set(list(df1[k])+list(df2[k]))) == len(df1[k])and len(set(list(df1[k])+list(df2[k]))) == len(df1[k])}
print(len(shared_items))


# In[123]:


shared_items


# In[128]:


df_mar.groupby(["MEDICATION_NAME"]).MRN_DEIDENTIFIED.unique().apply(len).sort_values(ascending = False).head(30).plot.bar(figsize = (20,10))


# In[129]:


df_mar.groupby(["MEDICATION_NAME"]).MRN_DEIDENTIFIED.unique().apply(len).sort_values(ascending = False).head(60).tail(30).plot.bar(figsize = (20,10))

