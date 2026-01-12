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

### WITHOUT `Item Purchased` (behaviour-only scenario)
- **Best Macro-F1:** **0.240** (LightGBM)  
- **Best Accuracy:** **0.448** (Random Forest)

<p align="center">
  <img src="reports/figures/WITH_Item_Purchased_test_f1_macro.png" width="900" />
</p>

<p align="center">
  <img src="reports/figures/WITHOUT_Item_Purchased_LightGBM_test_f1_macro.png" width="900" />
</p>

---

## Credibility checks (beyond a single score)
### Bootstrap uncertainty (95% CI on Macro-F1)
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

<p align="center">
  <img src="reports/figures/s12_parity_Gender_with.png" width="450" />
  <img src="reports/figures/s12_parity_Season_with.png" width="450" />
</p>

> Note: Location-level slices can be noisy when support is small. Interpret gaps alongside sample counts.

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
