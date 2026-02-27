#!/usr/bin/env python
# coding: utf-8

# # Machine Learning: Predicting ACL Rehabilitation Plateaus
# Objective: Develop a predictive model to identify athletes at risk of a "rehabilitation plateau" using early-stage data (up to 60 days post-op). Following recent survey data highlighting the importance of objective testing, we focus on both relative symmetry (LSI) and adjusted torque (Nm/kg).

# In[66]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind
# Load cleaned longitudinal data
df = pd.read_csv('/Users/terahskillings/Downloads/CleanACL2.csv')



# In[67]:


# TARGET VARIABLE DEFINITION
# We define a 'Plateau' based on two criteria: 
# 1. Peak LSI < 90% (Clinically accepted number)
# 2. Final 3-visit improvement < 5% (Stalled Progress)
# This provides a binary target (1 = Plateau, 0 = Progressing) for our model.
df = df.sort_values(['Athlete ID', 'Days_Post_Op'])
peak_lsi = df.groupby('Athlete ID')['Quad_LSI_Surgical'].max()

# Calculate improvement in final 3 timepoints
plateau_flags = {}

for athlete in df['Athlete ID'].unique():
    athlete_df = df[df['Athlete ID'] == athlete]
    
    if len(athlete_df) >= 3:
        last_three = athlete_df.tail(3)
        improvement = last_three['Quad_LSI_Surgical'].max() - last_three['Quad_LSI_Surgical'].min()
    else:
        improvement = np.nan
    
    plateau_flags[athlete] = improvement

plateau_df = pd.DataFrame({
    'Peak_LSI': peak_lsi,
    'Final3_Improvement': pd.Series(plateau_flags)
})

# Plateau rule
plateau_df['Plateau'] = (
    (plateau_df['Peak_LSI'] < 90) |
    (plateau_df['Final3_Improvement'] < 5)
)


plateau_df


# In[68]:


# EARLY WINDOW SELECTION 
# We isolate the 0-60 day window (this is excluding the inital visit previously removed). 
#This is the 'Intervention Zone' where PTs can actually pivot the treatment plan 
#if the model identifies a high risk of failure.
early_df = df[(df['Days_Post_Op'] >= 0) & (df['Days_Post_Op'] <= 60)]

early_summary = early_df.groupby('Athlete ID')[[
    'Quad_LSI_Surgical',
    'Quad_Torque_Surgical',
    'Ham90_Torque_Surgical',
    'SL_Vert_LSI_Surgical'
]].mean()

analysis_df = plateau_df.merge(early_summary, left_index=True, right_index=True)

analysis_df


# In[69]:


# --- STEP 4: GROUPED MEANS (from Output #5) ---
# This shows the clear drop in average Torque/BW for the Plateau group
print(analysis_df.groupby('Plateau')[['Quad_LSI_Surgical', 'Quad_Torque_Surgical', 
                                      'Ham90_Torque_Surgical', 'SL_Vert_LSI_Surgical']].mean())



# We calculate the mean values for our key metrics to observe initial differences between the 'Plateau' and 'Progressing' groups. Notably, jump metrics (SL_Vert) are largely absent in the 0–60 day window, confirming they are unsuitable for early prediction.

# In[70]:


# T-test for LSI (p=0.60) vs T-test for Torque (p=0.47)
# Neither is significant alone due to small N, but Torque shows a larger difference
plateau_lsi = analysis_df[analysis_df['Plateau'] == True]['Quad_LSI_Surgical']
non_plateau_lsi = analysis_df[analysis_df['Plateau'] == False]['Quad_LSI_Surgical']
print("LSI T-Test:", ttest_ind(plateau_lsi, non_plateau_lsi, equal_var=False))

plateau_torque = analysis_df[analysis_df['Plateau'] == True]['Quad_Torque_Surgical']
non_plateau_torque = analysis_df[analysis_df['Plateau'] == False]['Quad_Torque_Surgical']
print("Torque T-Test:", ttest_ind(plateau_torque, non_plateau_torque, equal_var=False))


# We first test LSI (relative symmetry) and Torque (absolute strength) individually for the quad. Traditional LSI models often fail to identify "weak but symmetric" athletes. Our goal is to see which metric provides a cleaner "signal" for prediction.

# We perform an independent t-test to determine if early-stage LSI significantly differs between groups. A high p-value (0.60) suggests that LSI alone is a noisy and unreliable early predictor of long-term success.

# # Predictive Modeling: From Baseline to Multivariate
# We compare traditional symmetry models against absolute force models. While LSI is the standard, our results show that absolute Torque provides a better predictive signal.

# In[87]:


#LSI BASELINE
ml_df_lsi = analysis_df.dropna(subset=['Quad_LSI_Surgical'])
X_lsi = ml_df_lsi[['Quad_LSI_Surgical']]
y_lsi = ml_df_lsi['Plateau'].astype(int)
model_lsi = LogisticRegression().fit(X_lsi, y_lsi)
lsi_auc = roc_auc_score(y_lsi, model_lsi.predict_proba(X_lsi)[:,1])

# Calculate the LSI Threshold (The 'danger zone' for PTs)

thresholdLSI = -model.intercept_[0] / model.coef_[0][0]
thresholdLSI
print(f"LSI Baseline AUC: {lsi_auc:.4f}")
print(f"LSI Prediction Threshold: {thresholdLSI:.2f}")


# In[89]:


# STATISTICAL SIGNIFICANCE TEST
# Compare torque between groups
torque_plateau = ml_df[ml_df['Plateau'] == True]['Quad_Torque_Surgical']
torque_non_plateau = ml_df[ml_df['Plateau'] == False]['Quad_Torque_Surgical']
t_stat, p_val = ttest_ind(torque_plateau, torque_non_plateau, equal_var=False)

# Calculate the Torque Threshold (The 'danger zone' for PTs)

torque_threshold = -model_torque.intercept_[0] / model_torque.coef_[0][0]
print(f"Torque Baseline AUC: {torque_auc:.4f}")
print(f"Torque Prediction Threshold (Nm/kg): {torque_threshold:.2f}")


# The integration of absolute force metrics significantly outperformed the traditional symmetry-based approach, providing a more robust early-warning system for rehabilitation plateaus.

# In[91]:


# FINAL MULTIVARIATE MODEL
# Combining both yields the highest AUC (0.71)
X_multi = ml_df[['Quad_LSI_Surgical', 'Quad_Torque_Surgical']]
model_multi = LogisticRegression().fit(X_multi, y)
multi_auc = roc_auc_score(y, model_multi.predict_proba(X_multi)[:,1])

print(f"Final Multivariate AUC: {multi_auc:.4f}")
#Compare Coefficients
print("LSI Coefficient:", model_multi.coef_[0][0])
print("Torque Coefficient:", model_multi.coef_[0][1])


# In the final multivariate model, the Torque Coefficient is the primary driver of the prediction.
# 
# Torque Coefficient (-0.212): A strong negative relationship, meaning as absolute strength increases, the probability of a plateau significantly decreases.
# 
# LSI Coefficient (0.018): A near-zero value, suggesting that once absolute strength is accounted for, symmetry adds very little extra predictive power early in rehab.

# The "LSI Trap" Confirmed: Our model proves that an agjusted Torque is a much more reliable early indicator of success than LSI. Athletes may achieve high symmetry by detraining their "healthy" leg, but they cannot "fake" absolute torque, making it the superior metric for clinical decision-making.
