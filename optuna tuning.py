import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings, os

import optuna
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder, label_binarize
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score, r2_score, mean_squared_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs("outputs", exist_ok=True)
os.makedirs("data",    exist_ok=True)

# ── 1. LOAD & ENCODE ─────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_parquet("Data/cleaned.parquet")

CATEGORICAL = ["region","industry","company_size","ai_primary_tool","ai_use_case","data_privacy_level"]
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[CATEGORICAL] = enc.fit_transform(df[CATEGORICAL])
for col in df.select_dtypes(include=["object","string"]).columns:
    df[col] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(df[[col]])
df = df.apply(lambda c: c.cat.codes if hasattr(c,"cat") else c)

STAGE_ORDER = ["none","pilot","partial","full"]
TARGETS_REG = ["revenue_growth_percent","cost_reduction_percent"]

EXCLUDE_M1 = ["response_id","company_id","country","ai_adoption_stage","adoption_stage_ord",
              "ai_adoption_rate","survey_year","quarter","time_index","productivity_per_dollar",
              "annual_revenue_usd_millions_w","num_employees_w","ai_investment_per_employee_w"]
FEATURES_M1 = [c for c in df.columns if c not in EXCLUDE_M1]

EXCLUDE_M3 = EXCLUDE_M1 + TARGETS_REG + ["roi_proxy"]
FEATURES_M3 = [c for c in df.columns if c not in EXCLUDE_M3]

# Normalize target column
df["ai_adoption_stage"] = df["ai_adoption_stage"].astype(str).str.lower().str.strip()

# Map target
df["y"] = df["ai_adoption_stage"].map({s:i for i,s in enumerate(STAGE_ORDER)})

# Debug (optional)
print("Unique stages:", df["ai_adoption_stage"].unique())

# Remove invalid rows
df = df.dropna(subset=["y", "company_id"])
df = df.reset_index(drop=True)
X_all   = df[FEATURES_M1].astype(float)
y_all   = df["y"].copy()
groups  = df["company_id"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups))
X_train_m1, X_test_m1 = X_all.iloc[train_idx], X_all.iloc[test_idx]
y_train_m1, y_test_m1 = y_all.iloc[train_idx], y_all.iloc[test_idx]

# Remove NaNs AFTER split (critical)
mask = y_train_m1.notna()
y_train_m1 = y_train_m1[mask]
X_train_m1 = X_train_m1.loc[y_train_m1.index]

class_counts   = y_train_m1.value_counts().sort_index()
sample_weights = y_train_m1.map(lambda c: len(y_train_m1) / (4 * class_counts.get(c, 1)))

# Subsample 30k for fast trial evaluation
rng       = np.random.RandomState(42)
sub_idx   = rng.choice(len(X_train_m1), size=min(30000, len(X_train_m1)), replace=False)
X_sub_m1  = X_train_m1.iloc[sub_idx]
y_sub_m1  = y_train_m1.iloc[sub_idx]
w_sub_m1  = sample_weights.iloc[sub_idx]

# ── 2. OPTUNA — MODULE 1 (XGBoost) ───────────────────────────────────────────

print("\n── Tuning Module 1 (XGBoost) — 20 trials ──")
BASELINE_M1_F1  = 0.8277
BASELINE_M1_ROC = 0.9787

def objective_m1(trial):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators",    200, 500, step=100),
        max_depth        = trial.suggest_int("max_depth",        4,  8),
        learning_rate    = trial.suggest_float("learning_rate", 0.05, 0.15, log=True),
        subsample        = trial.suggest_float("subsample",     0.7,  1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight",  3, 12),
        reg_alpha        = trial.suggest_float("reg_alpha",    0.01, 2.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda",   0.01, 2.0, log=True),
        objective="multi:softprob", num_class=4,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    m = XGBClassifier(**params)
    m.fit(X_sub_m1, y_sub_m1, sample_weight=w_sub_m1)
    return f1_score(y_test_m1, m.predict(X_test_m1), average="macro")

study_m1 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study_m1.optimize(objective_m1, n_trials=20, timeout=150)
print(f"  Best trial F1: {study_m1.best_value:.4f}")

# Final retrain on full training set
best_m1 = XGBClassifier(
    **{**study_m1.best_params,
       "objective":"multi:softprob","num_class":4,"random_state":42,"n_jobs":-1,"verbosity":0}
)
best_m1.fit(X_train_m1, y_train_m1, sample_weight=sample_weights)
y_pred_m1 = best_m1.predict(X_test_m1)
y_prob_m1 = best_m1.predict_proba(X_test_m1)
macro_f1  = f1_score(y_test_m1, y_pred_m1, average="macro")
roc_auc   = roc_auc_score(
    label_binarize(y_test_m1, classes=[0,1,2,3]),
    y_prob_m1, multi_class="ovr", average="macro"
)
print(f"  Tuned — Macro F1: {macro_f1:.4f}  ROC-AUC: {roc_auc:.4f}  "
      f"(baseline F1: {BASELINE_M1_F1}  Δ={macro_f1-BASELINE_M1_F1:+.4f})")

pickle.dump({
    "model":        best_m1,
    "features":     list(FEATURES_M1),
    "stage_order":  STAGE_ORDER,
    "best_params":  study_m1.best_params,
    "best_f1":      macro_f1,
    "weighted_f1":  f1_score(y_test_m1, y_pred_m1, average="weighted"),
    "roc_auc":      roc_auc,
    "baseline_f1":  BASELINE_M1_F1,
    "baseline_roc": BASELINE_M1_ROC,
    "trials":       [(t.number, round(t.value, 4)) for t in study_m1.trials],
}, open("Notebook/data/m1_model_tuned.pkl","wb"))
print("  Saved: Notebook/data/m1_model_tuned.pkl")

# ── 3. OPTUNA — MODULE 3 (LightGBM) ──────────────────────────────────────────

print("\n── Tuning Module 3 (LightGBM) — 15 trials per target ──")
BASELINE_R2 = {"revenue_growth_percent": 0.2357, "cost_reduction_percent": 0.2445}

X_m3    = df[FEATURES_M3].astype(float)
m3_results = {}
m3_models  = {}

for target in TARGETS_REG:
    y_t = df[target].copy()
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss2.split(X_m3, y_t, groups=groups))
    Xtr, Xte = X_m3.iloc[tr_idx], X_m3.iloc[te_idx]
    ytr, yte = y_t.iloc[tr_idx],  y_t.iloc[te_idx]

    sub_i  = rng.choice(len(Xtr), size=min(30000, len(Xtr)), replace=False)
    Xsub, ysub = Xtr.iloc[sub_i], ytr.iloc[sub_i]

    def objective_m3(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators",    200, 500, step=100),
            learning_rate    = trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            max_depth        = trial.suggest_int("max_depth",        4,   8),
            num_leaves       = trial.suggest_int("num_leaves",      31, 127),
            subsample        = trial.suggest_float("subsample",     0.6,  1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples= trial.suggest_int("min_child_samples", 10,  50),
            reg_alpha        = trial.suggest_float("reg_alpha",    0.01, 2.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda",   0.01, 2.0, log=True),
            random_state=42, n_jobs=-1, verbose=-1,
        )
        m = lgb.LGBMRegressor(**params)
        m.fit(Xsub, ysub, eval_set=[(Xte, yte)],
              callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
        return r2_score(yte, m.predict(Xte))

    study_m3 = optuna.create_study(direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=42))
    study_m3.optimize(objective_m3, n_trials=15, timeout=100)
    bp = study_m3.best_params

    final_m = lgb.LGBMRegressor(**{**bp, "random_state":42, "n_jobs":-1, "verbose":-1})
    final_m.fit(Xtr, ytr, eval_set=[(Xte, yte)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    y_pred  = final_m.predict(Xte)
    r2      = r2_score(yte, y_pred)
    rmse    = np.sqrt(mean_squared_error(yte, y_pred))
    base_r2 = BASELINE_R2[target]

    print(f"  {target}: R²={r2:.4f}  RMSE={rmse:.4f}  (baseline R²={base_r2}  Δ={r2-base_r2:+.4f})")
    m3_results[target] = {
        "r2": r2, "rmse": rmse, "best_params": bp,
        "baseline_r2": base_r2,
        "trials": [(t.number, round(t.value, 4)) for t in study_m3.trials],
    }
    m3_models[target] = final_m

pickle.dump({
    "models":   m3_models,
    "features": list(FEATURES_M3),
    "results":  m3_results,
}, open("Notebook/data/m3_models_tuned.pkl","wb"))
print("  Saved: Notebook/data/m3_models_tuned.pkl")

# ── 4. CONVERGENCE PLOTS ─────────────────────────────────────────────────────

print("\nGenerating convergence plots...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

def plot_convergence(ax, trials, baseline, best, title, ylabel):
    nums = [t[0] for t in trials]
    vals = [t[1] for t in trials]
    best_curve = np.maximum.accumulate(vals)
    ax.plot(nums, vals,        "o",  color="#B5D4F4", markersize=5, alpha=0.7, label="Trial")
    ax.plot(nums, best_curve,  "-",  color="#378ADD", linewidth=2,  label="Best so far")
    ax.axhline(baseline, color="#D85A30", linestyle="--", linewidth=1.2, label=f"Baseline {baseline:.3f}")
    ax.axhline(best,     color="#1D9E75", linestyle="--", linewidth=1.2, label=f"Tuned   {best:.3f}")
    ax.set_title(title); ax.set_xlabel("Trial"); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

m1_data = pickle.load(open("Notebook/data/m1_model_tuned.pkl","rb"))
plot_convergence(axes[0], m1_data["trials"], m1_data["baseline_f1"], m1_data["best_f1"],
                 "Module 1 — XGBoost (Macro F1)", "Macro F1")

m3_data = pickle.load(open("Notebook/data/m3_models_tuned.pkl","rb"))
for i, target in enumerate(TARGETS_REG):
    r = m3_data["results"][target]
    plot_convergence(axes[i+1], r["trials"], r["baseline_r2"], r["r2"],
                     f"Module 3 — LightGBM\n{target.replace('_percent','').replace('_',' ')} (R²)", "R²")

plt.suptitle("Optuna convergence — TPE sampler", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("outputs/optuna_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: outputs/optuna_convergence.png")

print("""
══════════════════════════════════════════════════════
 OPTUNA TUNING COMPLETE
══════════════════════════════════════════════════════
 Module 1 (XGBoost)
   Baseline Macro F1: 0.8277
   Tuned    Macro F1: {m1f1:.4f}  ({m1d:+.4f})

 Module 3 (LightGBM)
   revenue_growth  R²: {r2r:.4f}  ({r2rd:+.4f} vs baseline)
   cost_reduction  R²: {r2c:.4f}  ({r2cd:+.4f} vs baseline)

 Tuned models saved — Streamlit app will use these automatically.
══════════════════════════════════════════════════════
""".format(
    m1f1=m1_data["best_f1"],
    m1d=m1_data["best_f1"]-m1_data["baseline_f1"],
    r2r=m3_data["results"]["revenue_growth_percent"]["r2"],
    r2rd=m3_data["results"]["revenue_growth_percent"]["r2"]-0.2357,
    r2c=m3_data["results"]["cost_reduction_percent"]["r2"],
    r2cd=m3_data["results"]["cost_reduction_percent"]["r2"]-0.2445,
))
