#!/usr/bin/env python
# coding: utf-8

# # ACL Practicum: Data Cleaning & Preprocessing
# Research Context: This script prepares longitudinal athlete data for analysis. Following global survey insights, we are prioritizing columns involving objective force and power metrics.
# Objective: Clean raw clinical exports and calculate surgical-leg specific metrics to identify early predictors of rehabilitation plateaus.

# In[274]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Load dataset
df = pd.read_csv('/Users/terahskillings/Downloads/ACL Master Longitudinal.csv')



# In[283]:


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Surgery Date'] = pd.to_datetime(df['Surgery Date'], errors='coerce')


# In[284]:


df['Days_Post_Op'] = (df['Date'] - df['Surgery Date']).dt.days

#The data set conatins athletes who visited before surgery also right after surgery. 
#We see an extreme drop off in the athete's first visits after surgury 
#We will filter for visits after this point
df = df[df['Days_Post_Op'] >= 14]


# In[277]:


# Ensure numeric types for analysis while preserving categorical IDs
text_cols = ["Athlete ID", "Procedure", "Surgery Side"]

for col in df.columns:
    if col not in text_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# Athletes that had pre surgury visits along with early post surgery visits had more outliers shown. We see the athetes significangtly drop in their metric scores when they are at their first rehabilitation visit. To limit this outlier, we will be removing the first two weeks after surgery.

# In[278]:


df.info(10)


# A surgical side column is needed. We cannot test only left or right, because this would cause disparity looking at a mix of healthy and injured ACLs. With clinical data exports separate metrics by "Left" and "Right," we must normalize them to "Surgical" vs "Non-Surgical." This allows our model to treat every athlete's surgery side as the main point we can focus on.

# In[279]:


# Create Surgical-side specific columns based on 'Surgery Side' column
# This aligns with the "absolute vs relative" analysis required for our predictive model

# Quadriceps Symmetry (Relative Metric)
df['Quad_LSI_Surgical'] = np.where(df['Surgery Side'] == 'R', df['Quad LSI R'], df['Quad LSI L'])
# Quadriceps Torque (Absolute Metric - Nm/kg equivalent)
df['Quad_Torque_Surgical'] = np.where(df['Surgery Side'] == 'R', df['Quad R Torque KgM'], df['Quad L Torque KgM'])
# Hamstring Symmetry at 90 degrees (Isometric Strength)
df['Ham90_Torque_Surgical'] = np.where(df['Surgery Side'] == 'R', df['HAM 90 R Torque Adjusted-Ham'], df['HAM 90 L Torque Adj-ham'])
# Single Leg Vertical Jump Symmetry (Functional Metric)
df['SL_Vert_LSI_Surgical'] = np.where(df['SL LSI R'] == 'R', df['SL LSI R'], df['SL LSI L'])
# PERFORMANCE NOTE: The 'highly fragmented' warning is expected due to bulk column insertion.


# Each row represents an athletes visit. Every missing value under measured feilds is purposeful. If there is a missing value that means that specific test was not preformed for that given visit. The Missing values such as athlete ID, and days post op, cannot have missing values. With only 18 athletes using longitudinal testing is needed. This requires days post op to to see how the athlete is progressing.

# In[280]:


# We will defragment the frame by making a deep copy.
df_clean = df.dropna(subset=["Days_Post_Op", "Athlete ID"]).copy()


# Saving the standardized dataset for Exploratory Data Analysis (EDA) and Machine Learning. The resulting CleanACL.csv contains normalized surgical metrics ready for plateau prediction.

# In[281]:


# Export cleaned data for the next stage of the pipeline
df_clean.to_csv('CleanACL.csv')

