# Arizona Heart Disease Treatment Cost Analysis
### CLC Capstone Project Thesis — Analytics Program

---

## Overview

This project analyzes factors that affect the cost of heart disease treatment in the state of Arizona. The goal is to help healthcare stakeholders strategically minimize treatment costs, save lives, and better allocate resources across the state.

Heart disease is the leading cause of death in Arizona and the United States. Despite its prevalence, the financial burden of treatment remains a major barrier to care for many patients. This study applies advanced analytics to identify the key cost drivers and provide actionable recommendations.

---

## Business Problem

> *How can the state of Arizona identify and address the factors driving heart disease treatment costs to minimize expenses, save lives, and more effectively allocate healthcare resources?*

---

## Analytics Approach

Two primary models were applied:

| Model | Purpose |
|---|---|
| **Linear Regression** | Identify which cost variables (professional fee, diagnosis, lab, medication) most significantly predict total treatment charges |
| **K-Means Clustering** | Segment patients into cost groups to support resource allocation and eligibility criteria for subsidized care |

Additional tools and techniques used:
- Correlation matrix analysis
- Descriptive statistics and data diagnostics
- 10-fold cross-validation for model evaluation
- Variable importance ranking

---

## Key Findings

- **Professional fee** (salaries of medical practitioners) emerged as the strongest predictor of treatment cost, with a coefficient of ~159 in the regression model.
- Three distinct patient cost clusters were identified via K-means, enabling targeted resource allocation.
- Patients with **Major severity** consistently fall in the highest-cost cluster, highlighting the value of early diagnosis programs.
- Government subsidization of professional fees, diagnosis, and medication costs — especially for lower-income patients — could meaningfully reduce the financial burden of heart disease treatment.

---

## Recommendations

**For Practice:**
- Build community clinics specifically for heart disease patients who cannot afford hospital charges.
- Recruit volunteer or reduced-rate medical practitioners for these clinics.
- Establish donation programs directed toward heart disease treatment access.
- Have the government subsidize diagnosis, lab, and medication costs based on patient income levels.

**For Future Research:**
- Use a **larger dataset** to improve model accuracy and enable proper train/test splitting.
- Select datasets with **historical data** to support external model validation.
- Explore additional modeling techniques (e.g., Random Forest, XGBoost) for deeper insight.
- Research funding and eligibility frameworks for government-subsidized heart disease clinics.

---

## Repository Structure

```
arizona-heart-disease-cost-analysis/
│
├── arizona_heart_disease_analysis.R   # Full R analysis script
├── README.md                          # Project documentation (this file)
└── data/
    └── arizona_heart_disease_data.csv # Source dataset (add your own)
```

> **Note:** The R script includes a simulated dataset for reproducibility. Replace it with the actual dataset by updating the `read.csv()` path at the top of the script.

---

## Requirements

Install the following R packages before running the script:

```r
install.packages(c("tidyverse", "cluster", "factoextra", "corrplot",
                   "ggplot2", "caret", "scales"))
```

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/arizona-heart-disease-cost-analysis.git
   ```
2. Open `arizona_heart_disease_analysis.R` in RStudio or any R environment.
3. (Optional) Replace the simulated data with your actual CSV dataset.
4. Run the script from top to bottom. All visualizations and model outputs will be generated automatically.

---

## Authors

CLC Capstone Project Team — Analytics Program

---

## References

- World Health Organization. (2011). *Prevention and treatment of HIV and other sexually transmitted infections among men who have sex with men and transgender people: Recommendations for a public health approach.*
- Arizona Department of Health Services — Heart Disease Data
- Centers for Disease Control and Prevention (CDC) — Heart Disease Facts

---

## License

This project is for academic purposes only.
