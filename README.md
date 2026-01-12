# 🔥 CST Product Category Prediction

![Industry](https://img.shields.io/badge/Industry-Retail%20%26%20E--commerce-1f6feb)
![Project%20Type](https://img.shields.io/badge/Project%20Type-Real%20World-2ea043)
![Task](https://img.shields.io/badge/Task-Multi--Class%20Classification-8250df)
![Dataset](https://img.shields.io/badge/Dataset-Customer%20Shopping%20Trends%20(CST)-0ea5e9)
![Models](https://img.shields.io/badge/Models-DT%20%7C%20RF%20%7C%20LR%20%7C%20XGBoost%20%7C%20LightGBM-f97316)
![Audit](https://img.shields.io/badge/Evaluation-Robustness%20Audit%20(WITH%20vs%20WITHOUT)-ef4444)
![Python](https://img.shields.io/badge/Python-3.10%2B-374151)
![License](https://img.shields.io/badge/License-MIT-7c3aed)

Predict **product category preferences** using customer demographic + behavioural attributes from the **Customer Shopping Trends (CST)** dataset.  
This repository is built as a hiring-ready ML project: reproducible pipelines, multi-model benchmarking, uncertainty quantification, significance testing, and a targeted **proxy/leakage audit**.

---

## What makes this project different
Many retail classification projects report a single score and stop. This repo stress-tests *trustworthiness*.

**Key audit:** `Item Purchased` behaves like a near-answer proxy for Category.  
I evaluate models **WITH** and **WITHOUT** this feature to quantify reliance and real-world validity.

<p align="center">
  <img src="reports/figures/s12_delta_f1_macro.png" width="900" />
</p>

---

## Research aim, questions, hypotheses (as implemented)
**Aim:** Develop and evaluate classification models that predict customer product category preferences using CST demographic and behavioural features.

**Research Questions**
1) What associations exist between customer attributes and product category purchased?  
2) How accurately can supervised ML models predict product category using available features?  
3) How do algorithms compare under cross-validation and standard classification metrics?  
4) Which features are most influential, and how does importance vary across models?

**Hypotheses**
- H₀₁: No significant association exists between customer features and category purchased.  
- H₁₁: At least one feature is significantly associated with category purchased.  
- H₀₂: No classifier exceeds the majority-class baseline accuracy.  
- H₁₂: At least one classifier significantly outperforms the baseline.  
- H₀₃: No significant performance difference exists between models.  
- H₁₃: At least two models differ significantly in performance.  
- H₀₄: Features do not significantly contribute to prediction.  
- H₁₄: At least one feature significantly improves prediction accuracy.

---

## Results snapshot (test set)
### WITH `Item Purchased` (upper-bound / proxy-risk scenario)
- **Best Macro-F1:** **0.617** (Decision Tree)  
- **Best Accuracy:** **0.704** (Decision Tree)

| Model               | Accuracy  | Precision (Macro) | Recall (Macro) | F1 (Macro) | F1 (Weighted) | ROC-AUC (Macro OVR) | PR-AUC (Macro) |
| ------------------- | --------- | ----------------- | -------------- | ---------- | ------------- | ------------------- | -------------- |
| Decision Tree       | **0.704** | **0.900**         | 0.557          | **0.617**  | **0.681**     | 0.852               | 0.626          |
| LightGBM            | 0.679     | 0.687             | 0.562          | 0.595      | 0.666         | 0.852               | 0.693          |
| Logistic Regression | 0.682     | 0.759             | 0.556          | 0.602      | 0.668         | 0.854               | 0.696          |
| Random Forest       | 0.691     | 0.762             | **0.566**      | 0.610      | 0.678         | 0.851               | **0.699**      |
| XGBoost             | 0.689     | 0.795             | 0.554          | 0.605      | 0.670         | **0.855**           | **0.699**      |

### WITHOUT `Item Purchased` (behaviour-only scenario)
- **Best Macro-F1:** **0.240** (LightGBM)  
- **Best Accuracy:** **0.448** (Random Forest)


| Model               | Accuracy  | Precision (Macro) | Recall (Macro) | F1 (Macro) | F1 (Weighted) |
| ------------------- | --------- | ----------------- | -------------- | ---------- | ------------- |
| Decision Tree       | 0.395     | 0.183             | 0.235          | 0.196      | 0.320         |
| Random Forest       | **0.448** | 0.179             | 0.247          | 0.187      | 0.314         |
| Logistic Regression | 0.432     | 0.284             | **0.260**      | 0.219      | 0.350         |
| XGBoost             | 0.415     | 0.211             | 0.247          | 0.203      | 0.329         |
| LightGBM            | 0.393     | **0.265**         | 0.255          | **0.240**  | **0.355**     |

<p align="center">
  <img src="reports/figures/WITH_Item_Purchased_test_f1_macro.png" width="900" />
</p>

<p align="center">
  <img src="reports/figures/WITHOUT_Item_Purchased_test_f1_macro.png" width="900" />
</p>

### 🧪 Statistical validation (beyond accuracy)
### Baseline vs Models (Wilcoxon test)

### All models significantly outperform the majority-class baseline:

| Metric   | Model               | Mean Model | Mean Baseline | p (Holm)   | Effect size (r) |
| -------- | ------------------- | ---------- | ------------- | ---------- | --------------- |
| Macro-F1 | Decision Tree       | 0.666      | 0.154         | **0.0016** | **0.88**        |
| Macro-F1 | LightGBM            | 0.649      | 0.154         | **0.0016** | **0.88**        |
| Macro-F1 | Logistic Regression | 0.656      | 0.154         | **0.0016** | **0.88**        |
| Macro-F1 | Random Forest       | 0.671      | 0.154         | **0.0016** | **0.88**        |
| Macro-F1 | XGBoost             | 0.660      | 0.154         | **0.0016** | **0.88**        |



### Cross-model significance (Friedman + Wilcoxon)

### The Friedman test confirms overall model differences:

| Metric   | χ²    | p-value      | Significant |
| -------- | ----- | ------------ | ----------- |
| Macro-F1 | 29.71 | **0.000006** | ✅           |


### Key pairwise results (Holm-corrected):
| Model A             | Model B       | p (Holm)   | Effect size |
| ------------------- | ------------- | ---------- | ----------- |
| LightGBM            | Random Forest | **0.0066** | **0.88**    |
| Logistic Regression | Random Forest | **0.0264** | **0.75**    |
| Random Forest       | XGBoost       | **0.0271** | **0.73**    |

This supports H₁₃ (models differ meaningfully).


### 🔍 Feature importance (model-based)

### Across all models, Item Purchased is ranked #1, confirming it acts as a proxy for Category.

Top features (example):
| Rank | Feature                | Model                         |
| ---- | ---------------------- | ----------------------------- |
| 1    | Item Purchased         | Decision Tree / RF / LightGBM |
| 2    | Purchase Amount (USD)  | Tree-based                    |
| 3    | Payment Method         | Tree-based                    |
| 4    | Review Rating          | Tree-based                    |
| 5    | Frequency of Purchases | LightGBM                      |





---

## Credibility checks (beyond a single score)
### Bootstrap uncertainty (95% CI on Macro-F1)
## 🧪 Uncertainty & Statistical Reliability

Model performance is not reported as single point estimates. Uncertainty and statistical significance are explicitly quantified.

### Bootstrap confidence intervals (Macro-F1)
Each model’s Macro-F1 is bootstrapped 1,000 times on the test set to estimate 95% confidence intervals.

<p align="center">
  <img src="reports/figures/s12_bootstrap_ci_f1_with.png" width="450" />
  <img src="reports/figures/s12_bootstrap_ci_f1_without.png" width="450" />
</p>

These intervals show:
- Narrow uncertainty in the **WITH Item Purchased** scenario
- Substantially wider uncertainty when the proxy feature is removed, reflecting real-world instability

<p align="center">
  <img src="reports/figures/s12_bootstrap_ci_f1_with.png" width="450" />
  <img src="reports/figures/s12_bootstrap_ci_f1_without.png" width="450" />
</p>

### Paired significance (McNemar test on top models)
<p align="center">
  <img src="reports/figures/s12_mcnemar_bubble_with.png" width="650" />
</p>

---

## Slice parity sanity checks
Macro-F1 is evaluated across slices to detect instability or uneven performance.

## 📊 Paired Significance Testing (McNemar)

To determine whether performance differences are real or due to chance, McNemar’s test is applied to the same test instances.

### WITH `Item Purchased`
Decision Tree vs XGBoost  
p = 0.143 → no statistically significant difference  

### WITHOUT `Item Purchased`
LightGBM vs Logistic Regression  
p = 0.077 → borderline but not significant  

### Cross-scenario stability
Each model was compared WITH vs WITHOUT `Item Purchased`.  
All models showed **p < 0.001**, confirming that removing the proxy feature causes a **real and statistically significant collapse in performance**.

<p align="center">
  <img src="reports/figures/s12_mcnemar_bubble_with.png" width="600" />
</p>

<p align="center">
  <img src="reports/figures/s12_parity_Gender_with.png" width="450" />
  <img src="reports/figures/s12_parity_Season_with.png" width="450" />
</p>

> Note: Location-level slices can be noisy when support is small. Interpret gaps alongside sample counts.



## ⚖️ Slice Parity & Stability Checks

Macro-F1 is evaluated across demographic and behavioural slices to detect instability or bias.

<p align="center">
  <img src="reports/figures/s12_parity_Gender_with.png" width="420" />
  <img src="reports/figures/s12_parity_Season_with.png" width="420" />
</p>

Key findings:
- **Gender gap (WITH):** 0.021 → stable and fair
- **Season gap (WITH):** 0.125 → moderate seasonal variation
- **Location gap (WITH):** large, but driven by small-sample states (e.g., Illinois n=7)

After removing `Item Purchased`, slice gaps remain small for Gender and Season, confirming that performance collapse is not driven by demographic bias but by loss of proxy information.

---

## Repository layout
- `notebooks/` — analysis notebooks (EDA → modelling → robustness audit)
- `src/` — reusable pipeline code (prep, training, evaluation, statistical tests)
- `reports/figures/` — curated final figures used in this README
- `reports/tables/` — key result tables (metrics, tests, feature ranks)
- `docs/model_card.md` — intended use, limitations, and responsible interpretation

---

## Quickstart
```bash
pip install -r requirements.txt
