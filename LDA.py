#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_lda = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\LDA.txt",delimiter="\t", encoding = "ISO-8859-1")


# In[3]:


def dtypes_with_examples(table):
    return pd.merge(table.head(1).T , table.dtypes.to_frame("Type").Type, left_index = True, right_index = True)


# In[4]:


dtypes_with_examples(df_lda)


# In[5]:


df_lda.head(10)


# In[4]:


df_lda[['DATE_PLACED_DEIDENTIFIED','DATE_REMOVED_DEIDENTIFIED']]= df_lda[['DATE_PLACED_DEIDENTIFIED','DATE_REMOVED_DEIDENTIFIED']].applymap(pd.Timestamp)


# In[5]:


dtypes_with_examples(df_lda)


# In[8]:


df_lda.LDA_NAME.unique()


# In[16]:


# if LDA column is NaN, replace the values as follows:
# 1. Get all lda columns that are nan; modifications made only to this set - 1115 rows 
replace_lda_tag = df_lda.loc[df_lda.LDA_NAME.isnull() == True]


# # modified LDA file according to follwoing conditions: 
# * GI Tube Gastrostomy is a very important feature; LDA tag = CVC
# * Puncture Site Arterial Right Groin; LDA tag = ART Line
# * Negative Pressure Wound ; LDA tag = Wound
# * Intentionally Retained Foreign ; delete row
# * Biopsy ; delete row
# * Pheresis ; LDA tag = CVC
# * Introducer with side port Right ; LDA tag = CVC
# * Puncture Site Groin Anterior; LDA tag = CVC
# * Tracheostomy ; this should itself become an LDA tag
# * Infiltration/Extravasation; delete row
# * Intraosseous ; LDA tag = Intraosseous
# 
# Group all biopsy together. Doubt - Do these need to be deleted as well?
# 

# In[20]:


import numpy as np
conditions = ["GI Tube Gastrostomy",
              "Puncture Site Arterial Right Groin",
              "Negative Pressure Wound", 
              "Intentionally Retained Foreign", 
              "Pheresis",
              "Introducer with side port Right",
              "Puncture Site Groin Anterior",
              "Tracheostomy",
              "Infiltration/Extravasation",
              "Intraosseous"
             ]

values =['CVC Line','ART Line','Wound','Delete','CVC Line','CVC Line','CVC Line',"Tracheostomy",'Delete','Intraosseous Line' ]

replace_lda_tag["LDA_NAME"] = np.select([replace_lda_tag.LDA_MEASUREMENTS_AND_ASSESSMENTS.str.contains(conditions[i]) for i in range(len(conditions))],values)
# for i in range (len(conditions)):
#     replace_lda_tag.LDA_MEASUREMENTS_AND_ASSESSMENTS.str.contains(conditions[i])
#     replace_lda_tag.LDA_NAME = values[i]
#     break
    
    


# In[10]:


df_lda.loc[(df_lda.LDA_NAME.isnull() == True) & (df_lda.TEMPLATE_NAME.isnull()!= True) ]


# In[21]:


replace_lda_tag


# In[11]:


df_lda.loc[(df_lda.LDA_NAME.isnull() == True) & (df_lda.TEMPLATE_NAME.isnull()!= True)& (df_lda.LDA_MEASUREMENTS_AND_ASSESSMENTS != '[REMOVED] Implanted Port (Peds) Right')]


# Observation : for 1115 entries LDA_NAME is null. Most of the values in LDA_TEMPLATE_NAME are null. Only 3 enties are not null and they correspond to LDA_MEASUREMENTS_AND_ASSESSMENTS = [REMOVED] Implanted Port (Peds) Right.
# 
# I believe they are repeat entries as MRN, CSN and both timestamps are same. Only thing differing is the Template. 
# 
# TODO : understand which entry to keep

# In[21]:


df_lda.LDA_MEASUREMENTS_AND_ASSESSMENTS.unique()# has 3028 unique values ; [REMOVED] Implanted Port (Peds) Right is used only for the 3 rows mentioned above


# In[22]:


df_lda.loc[df_lda.LDA_MEASUREMENTS_AND_ASSESSMENTS == "[REMOVED] Implanted Port (Peds) Right"]


# In[24]:


df_lda.groupby(["LDA_MEASUREMENTS_AND_ASSESSMENTS"]).size().sort_values(ascending = False)


# In[40]:


df_lda.groupby(['MRN_DEIDENTIFIED',"CSN_DEIDENTIFIED"]).size().sort_values(ascending = False).tail(1500).head(50)


# In[41]:


df_lda.loc[~df_lda.LDA_MEASUREMENTS_AND_ASSESSMENTS.str.startswith("[REMOVED]")].groupby(['MRN_DEIDENTIFIED',"CSN_DEIDENTIFIED"]).size()


# In[39]:


df_lda.loc[(df_lda.MRN_DEIDENTIFIED == 8) &(df_lda.CSN_DEIDENTIFIED == 1649)]# 8                 1649


# In[3]:


len(df_lda.MRN_DEIDENTIFIED.unique())


# In[5]:


len(df_lda.groupby(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"]))


# In[ ]:




