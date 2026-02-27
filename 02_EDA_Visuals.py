#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis: Identifying Rehab Patterns
# Objective: Examine athlete’s longitudinal testing data and identify data sparsity. This builds a needed foundation for our data that we can create a recovery timeline for our predictive model.

# In[173]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the surgical-normalized dataset
df = pd.read_csv('CleanACL.csv')



# In[174]:


# Longitudinal Audit: observing visit frequency and duration per athlete
# This provides context on the "density" of our longitudinal data.
df.groupby("Athlete ID").agg(
    visits=("Date", "count"),
    min_days=("Days_Post_Op", "min"),
    max_days=("Days_Post_Op", "max")
)


# In[175]:


key_metrics = [
    'Quad_LSI_Surgical',
    'Quad_Torque_Surgical',
    'Ham90_Torque_Surgical',
    'SL_Vert_LSI_Surgical'
]

df[key_metrics].describe()


# In[176]:


print(df['Days_Post_Op'].describe())


# In[180]:


#Our bins align with our >= 14 day filter
# We start at 0 but know data begins at 14. 
# 'Early Rehab' now covers 0-90 days, matching the model's 0-60 day focus.
df["Rehab_Phase"] = pd.cut(
    df["Days_Post_Op"],
    bins=[0, 90, 180, 365, 1000], 
    labels=["Early Rehab", "Mid Rehab", "Late Rehab", "Long-term Follow-up"]
)
phase_counts.plot(kind="bar", color='skyblue', edgecolor='black')
plt.title("Number of Testing Sessions by Rehab Phase (>= 14 Days)")
plt.xlabel("Rehab Phase")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=45)
plt.show()


# In[193]:


# Sparsity Check: Visualizing missingness in key predictive features
# High sparsity in jump metrics (SL_Vert) confirms they are better as 'Outcome' metrics
# rather than 'Early Predictors' due to clinical clearance timelines.
metrics_of_interest = ['Quad_LSI_Surgical', 'Quad_Torque_Surgical', 'Ham90_Torque_Surgical', 'SL_Vert_LSI_Surgical', 'Days_Post_Op']
plt.figure(figsize=(10, 4))
sns.heatmap(df[metrics_of_interest].isna(), cbar=False, cmap='viridis')
plt.title("Sparsity Check: Final Selected Features")
plt.show()


# In[194]:


plt.figure()
sns.heatmap(df[key_metrics].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Key Metrics")
plt.show()


# This heatmap allows us to inspect for multicollinearity between our key metrics. While Quad LSI and Quad Torque are strongly correlated (0.73), they are not redundant, confirming that each provides distinct predictive value for identifying rehab plateaus.

# We categorize the timeline into Early, Mid, and Late phases to analyze how strength profiles evolve. There will be no "pre op" visits or visits within two weeks of surgery due to previous cleaning. This categorization is necissary for defining our 0–60 day "Early Rehab" window for the predictive model.

# In[195]:


# Binning the timeline based on clinical recovery stages
df["Rehab_Phase"] = pd.cut(
    df["Days_Post_Op"],
    bins=[-1000, 0, 90, 180, 1000],
    labels=["Pre-Op", "Early Rehab", "Mid Rehab", "Late Rehab"])

# Boxplot Analysis: Visualizing strength progression
#We observe the variance in early-stage Quad LSI, which underscores the need for absolute torque metrics to 
#clarify recovery status.

sns.boxplot(data=df, x="Rehab_Phase", y="Quad_LSI_Surgical", 
            order=["Early Rehab", "Mid Rehab", "Late Rehab"])
plt.title("Quad Symmetry (LSI) Progression by Phase")
plt.show()


# This section identifies the "LSI Trap." By plotting relative symmetry against absolute torque, we identify outliers who meet LSI goals but lack the absolute power required for safe return-to-sport. This discovery justifies the use of a Multivariate Logistic Regression.

# In[184]:


# Trajectory Visualization: Mapping Quad LSI over time per athlete
# Red line indicates the 90% LSI clinical gold standard.
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="Days_Post_Op", y="Quad_LSI_Surgical", hue="Athlete ID", marker="o", alpha=0.5, legend=False)
plt.axhline(90, color='red', linestyle='--', label='90% LSI Goal')
plt.title("ACL Recovery Trajectories: Quad LSI Over Time")
plt.xlabel("Days Post-Surgery")
plt.show()


# In[185]:


# This scatter plot that proves why LSI alone is insufficient, Symmetry vs. Absolute Strength
# Athletes in the bottom-right (High LSI / Low Torque) are our 'hidden risk' cases.
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df, 
    x='Quad_LSI_Surgical', 
    y='Quad_Torque_Surgical', 
    s=100,
    palette='viridis'
)

#"Success Zones" added
plt.axvline(x=90, color='red', linestyle='--', alpha=0.5, label='90% LSI Goal')
plt.axhline(y=df['Quad_Torque_Surgical'].mean(), color='blue', linestyle='--', alpha=0.5, label='Avg Torque')

plt.title("The LSI Trap: Symmetry vs. Absolute Strength (0-60 Days)", fontsize=14)
plt.xlabel("Quad LSI (Relative Symmetry %)", fontsize=12)
plt.ylabel("Quad Adjusted Torque (Absolute Strength Nm/kg)", fontsize=12)
plt.legend(title="Future Plateau", loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()


# In[187]:


df['Recovered_90'] = (df['Quad_LSI_Surgical'] >= 90).astype(int)


# In[188]:


df = df.sort_values(['Athlete ID', 'Days_Post_Op'])


# In[196]:


# Custom logic to identify athletes who stalled or failed to reach clinical goals.
# This loop creates our 'Target' variable for the Machine Learning model.
plateau_results = []
for athlete, group in df.groupby('Athlete ID'):
    peak = group['Quad_LSI_Surgical'].max()
    
     # Check for improvement in the final 3 visits to identify late-stage stalls.
    if len(group) >= 3:
        last_three = group.tail(3)
        improvement = last_three['Quad_LSI_Surgical'].max() - last_three['Quad_LSI_Surgical'].min()
    else:
        improvement = np.nan
    # 1 = Plateaued, 0 = Progressing
    is_plateau = 1 if (peak < 90 or (pd.notna(improvement) and improvement < 5)) else 0
    
    plateau_results.append({
        'Athlete ID': athlete, 
        'Plateau': is_plateau, 
        'Peak_LSI': peak, 
        'Final_Improvement': improvement
    })


# In[197]:


plateau_df = pd.DataFrame(plateau_results)


# Now we can compare early-rehab Quad LSI scores against the final Plateau classification. This boxplot visualizes whether athletes who eventually plateaued looked different in their first 60 days of rehab than those who didn't.

# In[198]:


# Comparing early-stage performance between 'Plateaued' and 'Progressing' groups.
# Significant overlap here confirms that LSI alone is a weak early predictor.
plt.figure(figsize=(8, 5))
sns.boxplot(data=analysis_df, x='Plateau', y='Quad_LSI_Surgical', palette="Set2")
plt.title("Early Quad LSI: Stalled (1) vs. Progressing (0) Athletes")
plt.xticks([0, 1], ['Progressing', 'Plateaued'])
plt.show()


# In[199]:


# Save the finalized dataset
df.to_csv('CleanACL2.csv')

