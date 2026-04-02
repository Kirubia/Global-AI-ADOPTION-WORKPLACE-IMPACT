# 🌐 Global AI Adoption & Workplace Impact

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=for-the-badge)](https://global-ai-adoption-workplace-impact-m5tytcqf4y28237mev4rmk.streamlit.app/)

> A four-module machine learning system that tracks how 10,000 companies across 9 industries are navigating AI adoption — and what separates the leaders from the laggards.

---

## 📊 Dataset at a Glance

| Attribute | Detail |
|---|---|
| **Rows** | 150,000 quarterly survey responses |
| **Companies** | 10,000 unique (10–16 appearances each) |
| **Time span** | 2023–2026 (4 years) |
| **Features** | 43 raw → 52 after engineering |
| **Coverage** | 9 industries · 6 regions · 3 company sizes |

---

## 🧩 Project Architecture

```
Global AI Adoption
│
├── Module 1 — Adoption Stage Classifier     (XGBoost)
├── Module 2 — AI Maturity Segmentation      (K-Means + PCA)
├── Module 3 — ROI & Business Impact         (LightGBM + SHAP)
└── Module 4 — Ethics & Risk Factor Analysis (OLS + Logistic Regression)
```

---

## Module 1 — Adoption Stage Classifier

**Goal:** Predict which stage of AI adoption a company is at: `none / pilot / partial / full`

**Model:** XGBoost multi-class classifier with inverse-frequency class weighting

### Results

| Metric | Score |
|---|---|
| Macro F1 | **0.828** |
| Weighted F1 | **0.887** |
| ROC-AUC (OvR) | **0.979** |

**Per-class F1:** `none=1.00` · `pilot=0.88` · `partial=0.89` · `full=0.54`

> The `full` class (only 1.1% of data) remains the hardest to predict even with class weighting — SMOTE is an avenue worth exploring.

### Top Features (by gain)

| Feature | Importance |
|---|---|
| `years_using_ai` | 0.473 |
| `ai_maturity_score` | 0.185 |
| `ai_intensity` (budget × training) | 0.081 |
| `num_ai_tools_used` | 0.078 |
| `task_automation_rate` | 0.024 |

💡 **Key insight:** Experience (`years_using_ai`) dominates over budget, tool choice, or company size. *Time in the game is the strongest predictor of reaching advanced adoption stages.*

---

## Module 2 — AI Maturity Segmentation

**Goal:** Cluster companies into interpretable AI maturity segments

**Model:** K-Means (k=4) on 13 behavioural/investment features, PCA for visualisation

### Clusters Discovered

| Cluster | Name | Size | Maturity Score | Failure Rate | Revenue Growth |
|---|---|---|---|---|---|
| C1 | **AI Leaders** | 18.7% | 0.527 | 17.7% | **7.7%** |
| C2 | Steady Adopters | 28.8% | 0.382 | 22.8% | 5.5% |
| C0 | Struggling Experimenters | 23.8% | 0.334 | 27.4% | 4.1% |
| C3 | Early-stage / Laggards | 28.7% | 0.197 | 32.6% | 2.1% |

**Evaluation:** Silhouette = 0.133 · Davies-Bouldin = 2.03

💡 **Key insight:** Bottom to top cluster represents a **3.6× difference in revenue growth** (2.1% → 7.7%). The "Struggling Experimenters" are analytically notable — decent maturity but high failure rates, pointing to under-investment in training *relative to* tooling.

> **Note on silhouette score:** 0.13 is low by textbook standards but expected for dense, overlapping real-world behavioural data. The clusters form a clear business gradient, not geometric blobs.

---

## Module 3 — ROI & Business Impact Regression

**Goal:** Predict revenue growth and cost reduction from AI investment signals

**Model:** LightGBM regressors with early stopping and SHAP explainability

### Results

| Target | RMSE | MAE | R² |
|---|---|---|---|
| `revenue_growth_percent` | 4.660 | 3.763 | 0.236 |
| `cost_reduction_percent` | 2.921 | 2.379 | 0.245 |

### Top Revenue Drivers (LightGBM gain)

| Feature | Importance |
|---|---|
| `ai_adoption_rate` | 611 |
| `productivity_change_percent` | 595 |
| `customer_satisfaction` | 235 |
| `task_automation_rate` | 210 |
| `ai_budget_percentage` | 165 |

💡 **Key insight:** An R² of ~0.24 is intentionally honest — revenue is shaped by many factors beyond AI metrics. The SHAP beeswarm confirms `ai_maturity_score` and `task_automation_rate` as the strongest directional drivers: high maturity pushes revenue up; high failure rate drags it down.

---

## Module 4 — Ethics & Risk Factor Analysis

**Goal:** Estimate the causal effect of AI ethics committees on outcomes and identify failure risk drivers

**Method:** OLS regression adjustment with a 14-variable confounder set

> Propensity score matching (IPW/AIPW) was not viable — **99.92% of companies** have an ethics committee (9,992/10,000), violating the positivity assumption.

### OLS Results (Regression-Adjusted)

| Outcome | Coefficient | p-value | Significance |
|---|---|---|---|
| `revenue_growth_percent` | −0.037 | 0.216 | ns |
| `ai_failure_rate` | +0.093 | 0.002 | ** |
| `ai_maturity_score` | −0.003 | <0.001 | *** |

### Failure Risk — Logistic Odds Ratios

| Feature | Odds Ratio | Direction |
|---|---|---|
| `adoption_stage_ord` | **0.292** | ⬇ Strongly reduces risk |
| `ai_training_hours` | 0.889 | ⬇ Reduces risk |
| `ai_budget_percentage` | 0.953 | ⬇ Reduces risk |
| `has_ethics_committee` | 0.966 | ⬇ Slight reduction |
| `num_ai_tools_used` | 1.019 | ⬆ Slight increase |
| `data_privacy_ord` | 1.052 | ⬆ Slight increase |

💡 **Key insight:** After adjusting for confounders, ethics committees show *no* revenue benefit and correlate with slightly higher failure rates. This is almost certainly **reverse causality** — companies earlier in their AI journey proactively establish ethics governance. The committee is a response to risk, not a cause of it.

The single strongest failure risk reducer is `adoption_stage_ord` (OR=0.29) — advancing along the adoption ladder cuts failure risk by **71%**, far outweighing any governance structure.

---

## 🔑 Cross-Module Findings

**1. Experience over budget.**
`years_using_ai` is the top predictor of adoption stage (Module 1) and a key failure risk factor (Module 4). Maturity is built through time, not just spending.

**2. Maturity is the master variable.**
`ai_maturity_score` appears in the top features of all four modules — it predicts adoption stage, defines cluster identity, drives revenue growth, and reduces failure risk. The single best lever for a company is closing the gap between their current maturity and AI leaders (0.53).

**3. Training hours are undervalued.**
`ai_training_hours` consistently reduces failure risk (OR=0.89) and is a top revenue driver across modules. Yet the "Struggling Experimenters" cluster reveals companies that invest heavily in tools without proportional training investment.

**4. Ethics governance timing matters.**
Companies establish governance reactively, not proactively. Governance metrics should *not* be used as proxies for AI capability or outcome prediction.

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| **Modelling** | XGBoost, LightGBM, Scikit-learn |
| **Clustering** | K-Means, PCA |
| **Explainability** | SHAP |
| **Causal Analysis** | Statsmodels OLS, Logistic Regression |
| **Tuning** | Optuna |
| **App** | Streamlit |
| **Environment** | Python 3.10+, Jupyter Notebook |

---

## 📁 Repository Structure

```
├── Data/                   # Dataset files
├── Notebook/               # Jupyter notebooks (EDA + modelling)
├── app.py                  # Streamlit dashboard
├── optuna tuning.py        # Hyperparameter search
├── requirements.txt        # Python dependencies
└── Packages.txt            # System-level packages
```

---

## 🚀 Running Locally

```bash
# Clone the repo
git clone https://github.com/Kirubia/Global-AI-ADOPTION-WORKPLACE-IMPACT.git
cd Global-AI-ADOPTION-WORKPLACE-IMPACT

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

---

## ⚙️ Methodology Notes

- **Train/test splits:** `GroupShuffleSplit(groups=company_id, test_size=0.2)` — prevents data leakage across company time-series
- **Class imbalance (Module 1):** Handled via `sample_weight` proportional to inverse class frequency
- **Skewed features** (`annual_revenue`, `num_employees`, `ai_investment_per_employee`): log-transformed for linear models, raw for tree models
- **Module 4 limitation:** OLS controls for observed confounders only; unmeasured confounding cannot be ruled out

---

## 🌍 Live Demo

👉 **[Launch the Streamlit App](https://global-ai-adoption-workplace-impact-m5tytcqf4y28237mev4rmk.streamlit.app/)**

---

*Built with curiosity about how organisations actually learn to work with AI — and what the data says about who's doing it right.*
