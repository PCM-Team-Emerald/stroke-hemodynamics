#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df_IO_flowsheet = pd.read_csv("S:\Dehydration_stroke\CCDA_Data\IO_flowsheet.txt",delimiter="|")


# In[4]:


df_IO_flowsheet.shape


# In[5]:


df_IO_flowsheet.groupby(['MRN_DEIDENTIFIED', 'CSN_DEIDENTIFIED']).size().sort_values(ascending = False).head()


# In[6]:


def dtypes_with_examples(table):
    return pd.merge(table.head(1).T, table.dtypes.to_frame('Type').Type, left_index = True, right_index = True)


# In[7]:


dtypes_with_examples(df_IO_flowsheet)


# In[8]:


df_IO_flowsheet['RECORDED_DATE_TIME_DEIDENTIFIED'] =df_IO_flowsheet['RECORDED_DATE_TIME_DEIDENTIFIED'].apply(pd.Timestamp)


# In[9]:


dtypes_with_examples(df_IO_flowsheet)


# In[10]:


df_IO_flowsheet.head() 


# In[11]:


df_IO_flowsheet.loc[df_IO_flowsheet.MEAS_VALUE == "1"]


# I believe that values 0 and 1 show whether activity mentioned in FLOWSHEET_ROW_NAME happened or not. i.e they are boolean values and not as earlier thought (that they represent that the value was measured only once)
# 
# Below is the code to delete all entries with MEAS_VALUE =1 in case required.

# In[12]:


df_IO_flowsheet = df_IO_flowsheet.loc[df_IO_flowsheet.MEAS_VALUE != "1"] # delete all rows with meas_value =1


# In[13]:


df_IO_flowsheet.loc[(df_IO_flowsheet.MRN_DEIDENTIFIED == 3) & (df_IO_flowsheet.CSN_DEIDENTIFIED == 440)]


# In[14]:


df_IO_flowsheet.head()


# In[15]:


df_IO_flowsheet.MEAS_VALUE.astype(float) #this gives error since not all values are numerical. We also have strings.


# In[16]:


print (df_IO_flowsheet.loc[df_IO_flowsheet.MEAS_VALUE =="Yellow"])


# In[17]:


measured_values = df_IO_flowsheet.MEAS_VALUE.unique()


# In[18]:


measured_values.T


# In[19]:


len(measured_values)


# In[20]:


df_IO_flowsheet.groupby('MEAS_VALUE').size().sort_values(ascending = False)


# In[21]:


df_IO_flowsheet.loc[(df_IO_flowsheet.MRN_DEIDENTIFIED == 3) & (df_IO_flowsheet.CSN_DEIDENTIFIED == 440)]


# In[22]:


removal_time = df_IO_flowsheet.loc[df_IO_flowsheet.FLOWSHEET_ROW_NAME == "REMOVAL TIME"] # TODO find out what pattern in the time encoded in


# In[23]:


removal_time


# In[24]:


removal_time.MEAS_VALUE.apply(float).describe() # max time is 86340. Total sec in a day are 86400.


# In[25]:


df_IO_flowsheet.FLOWSHEET_ROW_NAME.unique()


# In[26]:


len(df_IO_flowsheet.FLOWSHEET_ROW_NAME.unique())


# In[27]:


def sec_to_hours(x):
    return 1000000000*pd.Timedelta(float(x)) # cannot assign unit sec here as there is a nan value, and only the native 
# form of func handles it. Without mentioning a unit the func converts to nanosec by default.


# In[28]:


replaced_time = removal_time.MEAS_VALUE.apply(sec_to_hours)


# In[29]:


replaced_time


# In[30]:


removal_time.loc[removal_time.MEAS_VALUE.isnull()] # total 15 such values


# In[31]:


removal_time.dropna(subset = ["MEAS_VALUE"])


# In[32]:


df_IO_flowsheet.groupby(["FLOWSHEET_ROW_NAME"]).MRN_DEIDENTIFIED.unique().apply(len).sort_values(ascending = False).head(60).plot.bar(figsize = (20,10))


# In[33]:


urine_output_labels = ['R URINARY INCONTINENCE OCCURRENCE',
                       'URINE OUTPUT',
                       'R JHM URINE NET OUTPUT',
                       'R JHM URINE OUTPUT',
                       'R URINE OCCURRENCE', 
                       'R URINE COLOR',
                       'R ENTEROSTOMY DRAINAGE APPEARANCE',
                       'R JHM IP URINARY CATHETER STATUS',
                       'R JHM NEW LDA DRAIN OUTPUT - NO I/O',
                       'R JHM IP URINE COLOR-CATHETER',
                       'JHM R IP LDA DRAIN OUTPUT NET', 
                       'R JHM IP BLADDER SCAN VOLUME', 
                       '*OLD R URINARY INCONTINENCE',
                       'R POST VOID CATH RESIDUAL (ML)',
                       'R URINE APPEARANCE',
                       'R JHM IP URINE APPEARANCE- DEVICE', 
                       'R JHM PED URINE OUTPUT MEASURED',
                       'R JHM UROSTOMY OUTPUT',
                       'R JHM UROSTOMY NET OUTPUT', 
                       'R JHM IP URINE OSTOMY OUTPUT COLOR',
                      ]


# In[34]:


oral_intake_labels =['R SALINE FLUSH',
                     'ORAL INTAKE',
                     'R JHM LDA GI TUBE FEEDING INTAKE (ML)',
                     'R JHM LDA GI TUBE FEEDING MEDICATIONS (ML)',
                     'R JHM LDA GI TUBE FEEDING FLUSHES (ML)',
                     'R NTRN PERCENT MEALS EATEN',
                     'R JHM LDA GI TUBE FEEDING RATE', 
                     'R NTRN ORAL SUPPLEMENTS', 'R JHM IP  FREE WATER',
                     '*OLD R JHM IP TUBE FEEDING RATE',
                     '*OLD R JUM IP TUBE FEEDING INTAKE (ML)','*OLD R TUBE FEEDING FLUSHES (ML)',
                     '*OLD R TUBE FEEDING FREQUENCY',
                     '*OLD R JHM IP TUBE FEEDING FORMULA',
                     'R JHM PEDS ENTERAL FEEDING SELECT',
                     'R JHM IP FORMULA ENTERAL AMOUNT (ML)',
                     'R JHM FORMULA HOURLY RATE', 'R PEDS TUBE FEEDING FLUSHES (ML)',
                     'R JHM NEW TUBE INTAKE - I/O', 
                    ]


# In[35]:


type_of_tybe_feeding = ['R JHM LDA GI TUBE FEEDING FREQUENCY',
                       'R JHM LDA GI TUBE FEEDING FORMULA']


# In[36]:


blood_transfusion_labels = ['R IP TPN VOLUME',
                           'R JHM ED BLOOD MASSIVE TRANSFUSION-RBC',
                           'R JHM ED BLOOD MASSIVE TRANSFUSION-PLASMA',
                           'R JHM ED BLOOD MASSIVE TRANSFUSION CRYOPRECIPITATE',
                           'R JHM ED BLOOD MASSIVE TRANSFUSION-PLATELET']


# In[37]:


conditions =[(df_IO_flowsheet.FLOWSHEET_ROW_NAME.isin(urine_output_labels)),
             (df_IO_flowsheet.FLOWSHEET_ROW_NAME.isin(oral_intake_labels)),
             (df_IO_flowsheet.FLOWSHEET_ROW_NAME.isin(type_of_tybe_feeding)),
             (df_IO_flowsheet.FLOWSHEET_ROW_NAME.isin(blood_transfusion_labels))]


# In[38]:


values =['urine output','oral intake','type of feeding tube','blood transfusion']


# In[39]:


df_IO_flowsheet['flowsheet_row_type'] = np.select(conditions,values)


# In[40]:


dtypes_with_examples(df_IO_flowsheet)


# In[41]:


df_IO_flowsheet.flowsheet_row_type.unique()


# In[42]:


df_IO_flowsheet


# In[43]:


df_IO_flowsheet.groupby(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"]).size().sort_values(ascending = False)


# In[44]:


df_IO_flowsheet.loc[(df_IO_flowsheet.MRN_DEIDENTIFIED == 277 )&(df_IO_flowsheet.CSN_DEIDENTIFIED== 72)]


# In[71]:


for feature in df_IO_flowsheet.FLOWSHEET_ROW_NAME.unique()[:10]:
    data = df_IO_flowsheet.loc[df_IO_flowsheet.FLOWSHEET_ROW_NAME == feature].sort_values(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED","RECORDED_DATE_TIME_DEIDENTIFIED"]).groupby(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"])["RECORDED_DATE_TIME_DEIDENTIFIED"].diff().dt.total_seconds().dropna().values
    filtered = data[~is_outlier(data)]
    plt.hist(filtered)
    plt.title(feature)
    plt.show()


    


# In[66]:


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


# In[ ]:





# In[69]:


from matplotlib import pyplot as plt
feature = "ORAL INTAKE"
data = df_IO_flowsheet.loc[df_IO_flowsheet.FLOWSHEET_ROW_NAME == feature].sort_values(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED","RECORDED_DATE_TIME_DEIDENTIFIED"]).groupby(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"])["RECORDED_DATE_TIME_DEIDENTIFIED"].diff().dt.total_seconds().dropna().values
filtered = data[~is_outlier(data)]
# Plot the results
fig, (ax1, ax2) = plt.subplots(nrows=2)

ax1.hist(data)
ax1.set_title('Original')

ax2.hist(filtered)
ax2.set_title('Without Outliers')

plt.show()


# In[60]:


df_IO_flowsheet.loc[1049941]


# In[64]:


df_IO_flowsheet.loc[(df_IO_flowsheet.MRN_DEIDENTIFIED == 2057)& (df_IO_flowsheet.CSN_DEIDENTIFIED == 343)&(df_IO_flowsheet.FLOWSHEET_ROW_NAME=="ORAL INTAKE")].sort_values("RECORDED_DATE_TIME_DEIDENTIFIED").to_clipboard()


# In[72]:


len(df_IO_flowsheet.groupby(["MRN_DEIDENTIFIED","CSN_DEIDENTIFIED"]))


# In[73]:


len(df_IO_flowsheet.MRN_DEIDENTIFIED.unique())


# In[ ]:




