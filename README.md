# ACL-Rehab-Plateau-Prediction

### **Project Overview**

This practicum project aims to improve the early identification of athletes at risk of a rehabilitation "plateau" following ACL reconstruction. While traditional clinical benchmarks rely heavily on **Limb Symmetry Index (LSI)**, this research demonstrates that **Absolute Quadriceps Torque** is a superior early predictor of long-term success.

### **The "LSI Trap"**

Through Exploratory Data Analysis (EDA), we identified a critical clinical gap: many athletes achieve 90% symmetry (LSI) by the end of rehab not because the surgical leg is strong, but because the healthy leg has detrained. This project utilizes a **Multivariate Logistic Regression** to catch these "symmetric but weak" athletes in the early-rehab window (0–60 days).

### **Key Findings**

* **Model Improvement:** Moving from an LSI-only baseline to a Multivariate model (LSI + Torque) increased predictive accuracy from **0.59 AUC to 0.71 AUC**.
* **Clinical Threshold:** Athletes falling below **6.79 Nm/kg** of torque/BW in the first 60 days are significantly more likely to plateau.
* **Feature Importance:** Torque was found to be the dominant predictor, with a coefficient of **-0.212** compared to LSI's near-zero impact in the combined model.

### **Repository Structure**

1. **`01_Data_Cleaning.py`**: Handles date-time conversions (fixing nanosecond deltas) and filters for a $\ge 14$-day post-op start date.
2. **`02_EDA_Visuals.py`**: Generates the "LSI vs. Torque" scatter plot and correlation heatmaps.
3. **`03_Machine_Learning.py`**: The core predictive pipeline comparing baseline and multivariate models.
4. **`Model_Template_SHELL.py`**: A functional script using **mock data** to demonstrate code logic while protecting athlete privacy.

### **How to Use**

To verify the machine learning logic without access to the private master dataset, please run:

```bash
python Model_Template_SHELL.py

```

This will execute the Logistic Regression pipeline and output a logic-check AUC score.

---

### **Final Tip**

You don't need to provide your full proposal unless the teacher specifically asks for a PDF upload. This README covers the "What, Why, and How" perfectly.

**Would you like me to help you draft the final "Conclusion" slide for your presentation using these specific AUC and Torque numbers?**
