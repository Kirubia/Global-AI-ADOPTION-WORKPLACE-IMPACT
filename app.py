"""
AI Company Adoption — Streamlit Dashboard
==========================================
Run locally:  streamlit run app.py
Deploy:       push to GitHub → connect to share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OrdinalEncoder, label_binarize
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Adoption ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 0.8rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-delta { font-size: 0.85rem; margin-top: 4px; }
    .delta-pos { color: #1D9E75; }
    .delta-neg { color: #D85A30; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #1a1a2e;
        border-left: 4px solid #378ADD; padding-left: 10px; margin: 1.5rem 0 1rem;
    }
    .insight-box {
        background: #EAF3DE; border-left: 4px solid #1D9E75;
        border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
        margin: 0.75rem 0; font-size: 0.9rem;
    }
    .warning-box {
        background: #FAEEDA; border-left: 4px solid #EF9F27;
        border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
        margin: 0.75rem 0; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Data & model loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet("Data/cleaned.parquet")
    return df

@st.cache_resource
def load_models():
    models = {}
    for name, path in [
        ("m1", "data/m1_model_tuned.pkl"),
        ("m2", "data/m2_model.pkl"),
        ("m3", "data/m3_models_tuned.pkl"),
        ("m4", "data/m4_results.pkl"),
    ]:
        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            models[name] = None
    return models

df      = load_data()
models  = load_models()

STAGE_ORDER = ["none", "pilot", "partial", "full"]
COLORS      = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]
CLUSTER_NAMES = {0: "Struggling experimenters", 1: "AI leaders",
                 2: "Steady adopters",          3: "Early-stage / laggards"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 AI Adoption ML")
    st.caption("150,000 survey responses · 10,000 companies · 2023–2026")
    st.divider()

    page = st.radio("Navigate", [
        "📊 Overview",
        "🎯 Module 1 — Adoption Classifier",
        "🔵 Module 2 — Maturity Clusters",
        "💰 Module 3 — ROI Regression",
        "⚖️ Module 4 — Ethics & Risk",
        "🔧 Optuna Tuning",
        "🔮 Predict a Company",
    ])

    st.divider()
    st.markdown("**Filters**")
    sel_industry = st.multiselect(
        "Industry", options=sorted(df["industry"].unique()), default=[]
    )
    sel_region = st.multiselect(
        "Region", options=sorted(df["region"].unique()), default=[]
    )
    sel_size = st.multiselect(
        "Company size", options=sorted(df["company_size"].unique()), default=[]
    )

# Apply sidebar filters
df_f = df.copy()
if sel_industry: df_f = df_f[df_f["industry"].isin(sel_industry)]
if sel_region:   df_f = df_f[df_f["region"].isin(sel_region)]
if sel_size:     df_f = df_f[df_f["company_size"].isin(sel_size)]

# ── PAGE: Overview ────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("AI Company Adoption — Research Dashboard")
    st.caption(f"Showing **{len(df_f):,}** rows after filters  ·  {df_f['company_id'].nunique():,} unique companies")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Avg adoption rate",      f"{df_f['ai_adoption_rate'].mean():.1f}%",       None),
        ("Full adoption",          f"{(df_f['ai_adoption_stage']=='full').mean()*100:.1f}%", None),
        ("Avg productivity gain",  f"{df_f['productivity_change_percent'].mean():.1f}%", None),
        ("Avg revenue growth",     f"{df_f['revenue_growth_percent'].mean():.1f}%",  None),
        ("Avg AI maturity score",  f"{df_f['ai_maturity_score'].mean():.3f}",        None),
    ]
    for col, (label, val, delta) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.metric(label, val)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Adoption stage breakdown</div>', unsafe_allow_html=True)
        vc = df_f["ai_adoption_stage"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(vc.index, vc.values, color=["#378ADD","#1D9E75","#888780","#EF9F27"])
        ax.set_ylabel("Count"); ax.set_xlabel("")
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Adoption rate by industry</div>', unsafe_allow_html=True)
        ind = df_f.groupby("industry")["ai_adoption_rate"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(ind.index, ind.values, color="#7F77DD")
        ax.set_xlabel("Mean adoption rate (%)")
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">Adoption rate trend (2023–2026)</div>', unsafe_allow_html=True)
        trend = df_f.groupby("survey_year")["ai_adoption_rate"].mean()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(trend.index, trend.values, "o-", color="#378ADD", linewidth=2)
        ax.set_ylabel("Mean adoption rate (%)"); ax.set_xlabel("Year")
        ax.set_ylim(30, 42)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        st.markdown('<div class="section-header">Maturity vs productivity</div>', unsafe_allow_html=True)
        samp = df_f.sample(min(3000, len(df_f)), random_state=42)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(samp["ai_maturity_score"], samp["productivity_change_percent"],
                   alpha=0.25, s=6, color="#1D9E75")
        ax.set_xlabel("AI maturity score"); ax.set_ylabel("Productivity change (%)")
        corr = df_f[["ai_maturity_score","productivity_change_percent"]].corr().iloc[0,1]
        ax.set_title(f"r = {corr:.3f}", fontsize=10)
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("""
    <div class="insight-box">
    <strong>Key finding:</strong> AI maturity score (r=0.74 with productivity) is the single most predictive
    variable across all four ML modules. The average adoption rate grew from 33.5% in 2023 to 39.2% in 2026.
    </div>
    """, unsafe_allow_html=True)

# ── PAGE: Module 1 ────────────────────────────────────────────────────────────
elif page == "🎯 Module 1 — Adoption Classifier":
    st.title("Module 1 — AI Adoption Stage Classifier")
    st.caption("XGBoost multi-class classifier · Target: none / pilot / partial / full")

    m1 = models.get("m1")
    if m1 is None:
        st.error("Run `02_module1_adoption_classifier.py` first.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro F1",    f"{m1['best_f1']:.4f}",  f"+{m1['best_f1']-m1['baseline_f1']:.4f} vs baseline")
    c2.metric("Weighted F1", f"{m1.get('weighted_f1',0.889):.4f}", "")
    c3.metric("ROC-AUC",     f"{m1.get('roc_auc',0.979):.4f}", "")
    c4.metric("Classes", "4", "none · pilot · partial · full")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Feature importance (top 15)</div>', unsafe_allow_html=True)
        model   = m1["model"]
        features = m1["features"]
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = ["#378ADD"]*5 + ["#7F77DD"]*5 + ["#888780"]*5
        ax.barh(range(len(fi)), fi.values[::-1], color=colors[::-1])
        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(fi.index[::-1], fontsize=9)
        ax.set_xlabel("Importance (gain)")
        sns.despine()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Confusion matrix (normalised)</div>', unsafe_allow_html=True)
        try:
            img = plt.imread("outputs/m1_confusion_matrix.png")
            st.image(img)
        except Exception:
            st.info("Run the module script to generate the confusion matrix plot.")

    st.markdown("""
    <div class="insight-box">
    <strong>Key insight:</strong> <code>years_using_ai</code> dominates with 47% of total importance — experience
    predicts adoption stage far better than budget, tool choice, or company size. The model nails the <em>none</em>
    class (F1=1.00) but struggles with <em>full</em> (F1=0.54) due to class imbalance (1.1% of data).
    </div>
    """, unsafe_allow_html=True)

# ── PAGE: Module 2 ────────────────────────────────────────────────────────────
elif page == "🔵 Module 2 — Maturity Clusters":
    st.title("Module 2 — AI Maturity Segmentation")
    st.caption("K-Means (k=4) · 13 behavioural features · PCA visualisation")

    m2 = models.get("m2")
    if m2 is None:
        st.error("Run `03_module2_maturity_clustering.py` first.")
        st.stop()

    profile = m2["profile"]
    cnames  = m2["cluster_names"]

    c1, c2, c3, c4 = st.columns(4)
    cluster_colors = ["#D85A30", "#1D9E75", "#378ADD", "#888780"]
    for i, (col, (c, name)) in enumerate(zip([c1,c2,c3,c4], cnames.items())):
        rev = profile.loc[c, "revenue_growth_percent"]
        mat = profile.loc[c, "ai_maturity_score"]
        col.metric(name, f"{rev:.1f}% rev growth", f"maturity={mat:.3f}")

    st.divider()
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-header">PCA cluster map</div>', unsafe_allow_html=True)
        try:
            st.image("outputs/m2_clusters.png")
        except Exception:
            st.info("Run module 2 script to generate cluster plots.")

    with col2:
        st.markdown('<div class="section-header">Cluster profile comparison</div>', unsafe_allow_html=True)
        metrics_show = ["ai_maturity_score","ai_failure_rate","revenue_growth_percent",
                        "task_automation_rate","ai_training_hours"]
        profile_show = profile[metrics_show].copy()
        profile_show.index = [cnames[c] for c in profile_show.index]
        st.dataframe(profile_show.T.round(3), use_container_width=True)

        st.divider()
        st.markdown('<div class="section-header">Heatmap</div>', unsafe_allow_html=True)
        try:
            st.image("outputs/m2_heatmap.png")
        except Exception:
            pass

    st.markdown("""
    <div class="insight-box">
    <strong>Key insight:</strong> AI leaders vs Early-stage laggards — a 3.6× revenue growth gap (7.7% vs 2.1%).
    Struggling experimenters have decent maturity but high failure rates (27.4%), suggesting under-investment
    in training relative to tooling.
    </div>
    <div class="warning-box">
    <strong>Methodology note:</strong> Silhouette=0.13 is expected for dense, overlapping real-world data.
    Cluster utility is in the business gradient, not geometric separation.
    </div>
    """, unsafe_allow_html=True)

# ── PAGE: Module 3 ────────────────────────────────────────────────────────────
elif page == "💰 Module 3 — ROI Regression":
    st.title("Module 3 — ROI & Business Impact Regression")
    st.caption("LightGBM regressors · Targets: revenue_growth_percent, cost_reduction_percent")

    m3 = models.get("m3")
    if m3 is None:
        st.error("Run `04_module3_roi_regression.py` first.")
        st.stop()

    results = m3["results"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue R²",   f"{results['revenue_growth_percent']['r2']:.4f}",
              f"+{results['revenue_growth_percent']['r2']-results['revenue_growth_percent'].get('baseline_r2',0.235):.4f}")
    c2.metric("Revenue RMSE", f"{results['revenue_growth_percent']['rmse']:.3f}")
    c3.metric("Cost R²",      f"{results['cost_reduction_percent']['r2']:.4f}",
              f"+{results['cost_reduction_percent']['r2']-results['cost_reduction_percent'].get('baseline_r2',0.235):.4f}")
    c4.metric("Cost RMSE",    f"{results['cost_reduction_percent']['rmse']:.3f}")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Feature importance", "SHAP beeswarm", "Actual vs predicted"])

    with tab1:
        col1, col2 = st.columns(2)
        for col, target in zip([col1, col2], ["revenue_growth_percent","cost_reduction_percent"]):
            with col:
                model    = m3["models"][target]
                features = m3["features"]
                fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(5,5))
                colors = ["#1D9E75"]*5+["#378ADD"]*5+["#888780"]*5
                ax.barh(range(len(fi)), fi.values[::-1], color=colors[::-1])
                ax.set_yticks(range(len(fi)))
                ax.set_yticklabels(fi.index[::-1], fontsize=9)
                ax.set_title(target.replace("_percent","").replace("_"," "), fontsize=10)
                ax.set_xlabel("Gain")
                sns.despine()
                st.pyplot(fig, use_container_width=True)
                plt.close()

    with tab2:
        try:
            st.image("outputs/m3_shap_revenue.png", caption="SHAP beeswarm — revenue_growth_percent")
        except Exception:
            st.info("SHAP plot not found — run module 3 script.")

    with tab3:
        try:
            st.image("outputs/m3_actual_vs_predicted.png")
            st.image("outputs/m3_residuals.png")
        except Exception:
            st.info("Plots not found — run module 3 script.")

    st.markdown("""
    <div class="insight-box">
    <strong>On R²≈0.24:</strong> Revenue growth is shaped by macroeconomics, competition, and leadership —
    far beyond AI metrics. Explaining 24% of variance with internal AI signals alone is capturing genuine signal.
    The SHAP plot shows <em>direction</em>: high maturity → revenue up; high failure rate → revenue down.
    </div>
    """, unsafe_allow_html=True)

# ── PAGE: Module 4 ────────────────────────────────────────────────────────────
elif page == "⚖️ Module 4 — Ethics & Risk":
    st.title("Module 4 — Ethics & Risk Factor Analysis")
    st.caption("OLS regression adjustment · Causal framing · Failure risk odds ratios")

    m4 = models.get("m4")
    if m4 is None:
        st.error("Run `05_module4_ethics_risk.py` first.")
        st.stop()

    ols = m4["ols_results"]
    col1, col2, col3 = st.columns(3)
    for col, outcome in zip([col1,col2,col3], ["revenue_growth_percent","ai_failure_rate","ai_maturity_score"]):
        r = ols[outcome]
        sig = "***" if r["pval"]<0.001 else "**" if r["pval"]<0.01 else "*" if r["pval"]<0.05 else "ns"
        col.metric(
            outcome.replace("_percent","").replace("_"," ").title(),
            f"{r['coef']:+.4f}",
            f"p={r['pval']:.3f} {sig}"
        )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.image("outputs/m4_ethics_risk.png")

    with col2:
        st.markdown('<div class="section-header">Failure risk — odds ratios</div>', unsafe_allow_html=True)
        risk = m4["risk_fi"].reset_index()
        risk.columns = ["Feature","Odds ratio"]
        risk["Direction"] = risk["Odds ratio"].apply(lambda x: "Reduces risk" if x < 1 else "Increases risk")
        st.dataframe(
            risk.set_index("Feature").style
                .background_gradient(subset=["Odds ratio"], cmap="RdYlGn_r")
                .format({"Odds ratio": "{:.3f}"}),
            use_container_width=True, height=320
        )

    st.markdown("""
    <div class="insight-box">
    <strong>Reverse causality finding:</strong> After adjusting for confounders, ethics committees show
    no revenue benefit and correlate with slightly higher failure rates. Companies establish governance
    <em>reactively</em> — the committee is a response to risk, not a cause of outcomes.
    </div>
    <div class="insight-box">
    <strong>Strongest failure reducer:</strong> <code>adoption_stage_ord</code> OR=0.29 — advancing
    along the adoption ladder cuts failure risk by 71%, far outweighing any governance structure.
    </div>
    <div class="warning-box">
    <strong>Limitation:</strong> 99.9% of companies have an ethics committee — propensity score matching
    was not viable (positivity assumption violated). OLS cannot rule out unmeasured confounding.
    </div>
    """, unsafe_allow_html=True)

# ── PAGE: Optuna ──────────────────────────────────────────────────────────────
elif page == "🔧 Optuna Tuning":
    st.title("Optuna Hyperparameter Tuning")
    st.caption("TPE sampler · 15 trials per model · Subsampled training for speed")

    m1t = models.get("m1")
    m3t = models.get("m3")

    c1, c2, c3 = st.columns(3)
    if m1t:
        delta_f1 = m1t["best_f1"] - m1t["baseline_f1"]
        c1.metric("M1 Macro F1 (tuned)",    f"{m1t['best_f1']:.4f}", f"{delta_f1:+.4f} vs baseline")
    if m3t:
        r2_rev = m3t["results"]["revenue_growth_percent"]["r2"]
        r2_cst = m3t["results"]["cost_reduction_percent"]["r2"]
        c2.metric("M3 Revenue R² (tuned)",  f"{r2_rev:.4f}", f"{r2_rev-0.2357:+.4f}")
        c3.metric("M3 Cost R² (tuned)",     f"{r2_cst:.4f}", f"{r2_cst-0.2445:+.4f}")

    st.divider()
    try:
        st.image("outputs/optuna_convergence.png", caption="Convergence — best score per trial")
    except Exception:
        st.info("Run the tuning scripts to generate convergence plots.")

    st.divider()
    if m1t:
        st.markdown('<div class="section-header">Best hyperparameters — Module 1 (XGBoost)</div>', unsafe_allow_html=True)
        bp1 = m1t["best_params"]
        st.dataframe(pd.DataFrame(bp1.items(), columns=["Parameter","Value"]).set_index("Parameter"),
                     use_container_width=False)
    if m3t:
        st.markdown('<div class="section-header">Best hyperparameters — Module 3 (LightGBM)</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        for col, target in zip([col1, col2], ["revenue_growth_percent","cost_reduction_percent"]):
            with col:
                st.caption(target)
                bp = m3t["results"][target]["best_params"]
                st.dataframe(pd.DataFrame(bp.items(), columns=["Parameter","Value"]).set_index("Parameter"),
                             use_container_width=True)

# ── PAGE: Predict ─────────────────────────────────────────────────────────────
elif page == "🔮 Predict a Company":
    st.title("Predict AI Adoption Stage")
    st.caption("Enter company attributes to get a live prediction from the tuned XGBoost model")

    m1 = models.get("m1")
    if m1 is None:
        st.error("Run module 1 script first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        years_ai      = st.slider("Years using AI",       0, 15, 3)
        ai_budget_pct = st.slider("AI budget %",          0.0, 25.0, 8.0, 0.5)
        ai_training   = st.slider("AI training hours",    0.0, 80.0, 25.0, 1.0)
        num_tools     = st.slider("Num AI tools used",    1, 6, 2)
    with col2:
        ai_projects   = st.slider("Active AI projects",   0, 10, 3)
        task_auto     = st.slider("Task automation rate", 0.0, 50.0, 20.0, 0.5)
        failure_rate  = st.slider("AI failure rate (%)",  0.0, 40.0, 25.0, 0.5)
        maturity      = st.slider("AI maturity score",    0.0, 0.9, 0.35, 0.01)
    with col3:
        company_size  = st.selectbox("Company size",  ["Startup","SME","Enterprise"])
        industry      = st.selectbox("Industry",      ["Technology","Finance","Healthcare","Manufacturing","Retail","Agriculture","Education","Logistics","Consulting"])
        region        = st.selectbox("Region",        ["Asia","Europe","North America","South America","Africa","Oceania"])
        annual_rev    = st.number_input("Annual revenue (USD M)", 1.0, 10000.0, 100.0, 10.0)

    if st.button("Predict adoption stage", type="primary"):
        model    = m1["model"]
        features = m1["features"]

        # Build a minimal input row matching the feature list
        base = df.iloc[[0]].copy()
        num_emp = {"Startup": 50, "SME": 300, "Enterprise": 2000}[company_size]

        base["years_using_ai"]                 = years_ai
        base["ai_budget_percentage"]           = ai_budget_pct
        base["ai_training_hours"]              = ai_training
        base["num_ai_tools_used"]              = num_tools
        base["ai_projects_active"]             = ai_projects
        base["task_automation_rate"]           = task_auto
        base["ai_failure_rate"]                = failure_rate
        base["ai_maturity_score"]              = maturity
        base["ai_intensity"]                   = ai_budget_pct * ai_training
        base["log_annual_revenue_usd_millions"]= np.log1p(annual_rev)
        base["log_num_employees"]              = np.log1p(num_emp)

        CATEGORICAL = ["region","industry","company_size","ai_primary_tool","ai_use_case","data_privacy_level"]
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(df[CATEGORICAL])
        base["industry"]    = industry
        base["region"]      = region
        base["company_size"]= company_size
        base[CATEGORICAL]   = enc.transform(base[CATEGORICAL])

        for col in base[features].select_dtypes(include=["object","string"]).columns:
            base[col] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(df[[col]]).mean()

        X_input = base[features].apply(lambda c: c.cat.codes if hasattr(c,"cat") else c).astype(float)
        probs   = model.predict_proba(X_input)[0]
        pred    = STAGE_ORDER[np.argmax(probs)]

        st.divider()
        c1, c2 = st.columns([1,2])
        with c1:
            st.markdown(f"### Predicted stage: **{pred.upper()}**")
            st.caption("Probability breakdown")
            for stage, prob in zip(STAGE_ORDER, probs):
                st.progress(float(prob), text=f"{stage}: {prob*100:.1f}%")
        with c2:
            fig, ax = plt.subplots(figsize=(5,3))
            colors  = ["#378ADD","#1D9E75","#888780","#EF9F27"]
            ax.bar(STAGE_ORDER, probs, color=colors)
            ax.set_ylabel("Probability"); ax.set_ylim(0,1)
            ax.set_title("Predicted class probabilities")
            sns.despine()
            st.pyplot(fig, use_container_width=True)
            plt.close()
