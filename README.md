## Dataset

- 150,000 rows — quarterly survey responses across 4 years (2023–2026)
- 10,000 unique companies — each appears 10–16 times
- 43 raw features → 52 after feature engineering
- Covers 9 industries, 6 regions, 3 company sizes


### Module 1 — Adoption Stage Classifier
Goal: Predict which stage of AI adoption a company is at (none / pilot / partial / full).
Model: XGBoost multi-class classifier with class weighting.
Results:
MetricScoreMacro F10.828Weighted F10.887ROC-AUC (OvR)0.979
Per-class F1: none=1.00 · pilot=0.88 · partial=0.89 · full=0.54
The full class (1.1% of data) is the hardest to predict even with class weighting — SMOTE is an alternative worth exploring.
Top features by importance (gain):
FeatureImportanceyears_using_ai0.473ai_maturity_score0.185ai_intensity (budget × training)0.081num_ai_tools_used0.078task_automation_rate0.024
Key insight: Experience (years_using_ai) dominates over budget, tool choice, or company size. Time in the game is the strongest predictor of reaching advanced adoption stages.

### Module 2 — AI Maturity Segmentation
Goal: Cluster companies into interpretable AI maturity segments.
Model: K-Means (k=4) on 13 behavioural/investment features, PCA for visualisation.
Clusters discovered:
ClusterNameSizeMaturityFailure rateRevenue growthC1AI leaders18.7%0.52717.7%7.7%C2Steady adopters28.8%0.38222.8%5.5%C0Struggling experimenters23.8%0.33427.4%4.1%C3Early-stage / laggards28.7%0.19732.6%2.1%
Evaluation: Silhouette=0.133 · Davies-Bouldin=2.03
Key insight: The gap from bottom to top cluster represents a 3.6× difference in revenue growth (2.1% → 7.7%). The "Struggling experimenters" cluster is analytically interesting — decent maturity but high failure rates, likely due to under-investment in training relative to tooling.

Note on silhouette score: 0.13 is low by textbook standards but expected for dense, overlapping real-world behavioural data. The clusters form a clear business gradient, not geometric blobs.

### Module 3 — ROI & Business Impact Regression
Goal: Predict revenue growth and cost reduction from AI investment signals.
Model: LightGBM regressors with early stopping and SHAP explainability.
Results:
TargetRMSEMAER²revenue_growth_percent4.6603.7630.236cost_reduction_percent2.9212.3790.245
Top revenue drivers (LightGBM gain):
FeatureImportanceai_adoption_rate611productivity_change_percent595customer_satisfaction235task_automation_rate210ai_budget_percentage165
Key insight: R² of ~0.24 is intentionally honest — revenue is shaped by many factors beyond AI metrics. The SHAP beeswarm shows ai_maturity_score and task_automation_rate as the strongest directional pushers: high maturity drives revenue up; high failure rate drives it down.

### Module 4 — Ethics & Risk Factor Analysis
Goal: Estimate the causal effect of AI ethics committees on outcomes and identify failure risk drivers.
Method: OLS regression adjustment with a 14-variable confounder set. Propensity score matching (IPW/AIPW) was not viable — 99.92% of companies have an ethics committee (9,992/10,000), violating the positivity assumption.
OLS results (regression-adjusted):
OutcomeCoefficientp-valueSignificancerevenue_growth_percent−0.0370.216nsai_failure_rate+0.0930.002**ai_maturity_score−0.003<0.001***
Failure risk — logistic odds ratios:
FeatureOdds RatioDirectionadoption_stage_ord0.292Strongly reduces riskai_training_hours0.889Reduces riskai_budget_percentage0.953Reduces riskhas_ethics_committee0.966Slight reductionnum_ai_tools_used1.019Slight increasedata_privacy_ord1.052Slight increase
Key insight: After adjusting for confounders, ethics committees show no revenue benefit and correlate with slightly higher failure rates. This is almost certainly reverse causality — companies earlier in their AI journey (lower maturity, higher failure) proactively establish ethics governance. The committee is a response to risk, not a cause of it.
The single strongest failure risk reducer is adoption_stage_ord (OR=0.29) — advancing along the adoption ladder cuts failure risk by 71%, far outweighing any governance structure.

## Cross-module findings

Experience over budget. years_using_ai is the top predictor of adoption stage (Module 1) and a moderate failure risk factor (Module 4). Maturity is built through time, not just spending.
Maturity is the master variable. ai_maturity_score appears in the top features of all four modules — it predicts adoption stage, defines cluster identity, drives revenue growth, and reduces failure risk. The single best lever for a company is closing the gap between their current maturity and AI leaders (0.53).
Training hours undervalued. Across modules, ai_training_hours consistently reduces failure risk (OR=0.89) and is a top revenue driver. Yet the "Struggling experimenters" cluster shows companies that invest in tools without proportional training investment.
Ethics governance timing matters. Companies establish governance reactively, not proactively. This means governance metrics should not be used as proxies for AI capability or outcome prediction.


## Methodology notes

All train/test splits: GroupShuffleSplit(groups=company_id, test_size=0.2)
Class imbalance in Module 1: handled via sample_weight proportional to inverse class frequency
Skewed features (annual_revenue, num_employees, ai_investment_per_employee): log-transformed for linear models, raw for tree models
Module 4 limitation: OLS controls for observed confounders only; unmeasured confounding cannot be ruled out
