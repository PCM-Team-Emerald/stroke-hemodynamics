#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df_feeding =pd.read_csv("S:\Dehydration_stroke\CCDA_Data\Feeding.txt",delimiter="\t")


# In[4]:


df_feeding.head(5)


# In[4]:


def dtypes_with_examples(table):
    return pd.merge(table.head(1).T, table.dtypes.to_frame('Type').Type, left_index = True, right_index = True)


# In[5]:


dtypes_with_examples(df_feeding)


# In[9]:


df_feeding[['ORDER_TIME_DEIDENTIFIED', 'PROC_ENDING_TIME_DEIDENTIFIED','PROC_START_TIME_DEIDENTIFIED']]=df_feeding[['ORDER_TIME_DEIDENTIFIED', 'PROC_ENDING_TIME_DEIDENTIFIED','PROC_START_TIME_DEIDENTIFIED']].applymap(pd.Timestamp)


# In[10]:


dtypes_with_examples(df_feeding)


# In[12]:


df_feeding.PROC_NAME.unique() # maybe we can create a more generic proc name eg: combine adult BMC and JHH in one tag


# In[13]:


df_feeding.RESULT_TIME.unique() # can drop this entire column


# In[14]:


df_feeding.RESULT_TIME_DEIDENTIFIED.unique() # can drop this entire column


# In[17]:


df_feeding = df_feeding.drop(['RESULT_TIME','RESULT_TIME_DEIDENTIFIED'],axis =1)


# In[18]:


dtypes_with_examples(df_feeding)


# In[20]:


len(df_feeding.display_name.unique())


# In[30]:


df_feeding.groupby(['display_name']).size().sort_values(ascending = False).head(20)


# In[24]:


df_feeding.loc[(df_feeding.display_name == "Diet NPO")]


# In[26]:


df_feeding.loc[(df_feeding.display_name == "Diet NPO Diet Type: NPO with Tube Feeding") & (df_feeding.PROC_NAME != "DIET NPO")]


# In[27]:


df_feeding.loc[(df_feeding.display_name == "Diet NPO Diet Type:: NPO") & (df_feeding.PROC_NAME != "DIET NPO")]


# In[29]:


df_feeding.loc[(df_feeding.display_name == "NPO (with tube feeding) Diet Type: NPO with Tube Feeding") & (df_feeding.PROC_NAME != "NPO (WITH TUBE FEEDING)")]


# #TODO : check if proc_name and display name are the same 
# NOTE : All NPO diet variations are with PROC_NAME NPO DIET. ANd NPO with TUBE feeding is with PROC_NAME NPO (WITH TUBE FEEDING)

# In[5]:


len(df_feeding.MRN_DEIDENTIFIED.unique())


# In[6]:


len(df_feeding.groupby(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"]))


# In[ ]:




