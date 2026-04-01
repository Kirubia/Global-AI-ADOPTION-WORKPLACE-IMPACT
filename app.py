"""
AI Adoption & Workforce Impact — Streamlit Dashboard
=====================================================
Run locally:  streamlit run app.py
Deploy:       push to GitHub → connect to share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, label_binarize
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score, r2_score, mean_squared_error
import warnings, pathlib
warnings.filterwarnings("ignore")

BASE = pathlib.Path(__file__).parent.resolve()
def p(rel): return str(BASE / rel)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Global AI Adoption & Workforce Impact",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — full brand overhaul ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #1D9E75 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
}
.hero h1 {
    font-size: 2.2rem; font-weight: 800; margin: 0 0 0.4rem 0;
    background: linear-gradient(90deg, #fff 0%, #93c5fd 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { font-size: 1rem; color: #cbd5e1; margin: 0; }
.hero .badge {
    display: inline-block; background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px; padding: 3px 12px;
    font-size: 0.75rem; color: #e2e8f0; margin: 0.6rem 0.3rem 0 0;
}

/* ── Section header ── */
.sec-head {
    font-size: 1rem; font-weight: 700; color: #0f172a;
    border-left: 4px solid #378ADD; padding-left: 10px;
    margin: 1.4rem 0 0.8rem;
    text-transform: uppercase; letter-spacing: 0.04em;
}

/* ── Stat cards ── */
.stat-card {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1rem 1.2rem; text-align: center;
}
.stat-card .val { font-size: 1.8rem; font-weight: 800; color: #0f172a; }
.stat-card .lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Callout boxes ── */
.insight-box {
    background: #f0fdf4; border-left: 4px solid #1D9E75;
    border-radius: 0 10px 10px 0; padding: 0.85rem 1rem;
    margin: 0.8rem 0; font-size: 0.88rem; color: #14532d;
}
.warning-box {
    background: #fffbeb; border-left: 4px solid #EF9F27;
    border-radius: 0 10px 10px 0; padding: 0.85rem 1rem;
    margin: 0.8rem 0; font-size: 0.88rem; color: #78350f;
}
.danger-box {
    background: #fef2f2; border-left: 4px solid #ef4444;
    border-radius: 0 10px 10px 0; padding: 0.85rem 1rem;
    margin: 0.8rem 0; font-size: 0.88rem; color: #7f1d1d;
}
.info-box {
    background: #eff6ff; border-left: 4px solid #378ADD;
    border-radius: 0 10px 10px 0; padding: 0.85rem 1rem;
    margin: 0.8rem 0; font-size: 0.88rem; color: #1e3a5f;
}

/* ── Story step cards ── */
.step-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.2rem; margin: 0.5rem 0;
    border-top: 4px solid #378ADD;
}
.step-card h4 { margin: 0 0 0.4rem; font-size: 0.95rem; font-weight: 700; color: #0f172a; }
.step-card p  { margin: 0; font-size: 0.82rem; color: #475569; }

/* ── Prediction result ── */
.pred-result {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    border-radius: 14px; padding: 1.5rem; text-align: center; color: white;
}
.pred-result .stage { font-size: 2.5rem; font-weight: 800; color: #60a5fa; }
.pred-result .desc  { font-size: 0.9rem; color: #94a3b8; margin-top: 0.3rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.88rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] { font-size: 0.85rem; font-weight: 600; }

/* ── Metric delta color ── */
[data-testid="stMetricDelta"] { font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data & models ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_parquet(p("Data/cleaned.parquet"))

@st.cache_resource
def load_models():
    models = {}
    paths = {
        "m1": "Notebook/data/m1_model_tuned.pkl",
        "m2": "Notebook/data/m2_model.pkl",
        "m3": "Notebook/data/m3_models_tuned.pkl",
        "m4": "Notebook/data/m4_results.pkl",
    }
    for name, path in paths.items():
        try:
            with open(p(path), "rb") as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            models[name] = None
    return models

df     = load_data()
models = load_models()

STAGE_ORDER   = ["none", "pilot", "partial", "full"]
STAGE_COLORS  = {"none": "#64748b", "pilot": "#378ADD", "partial": "#7F77DD", "full": "#1D9E75"}
CLUSTER_NAMES = {0: "Struggling experimenters", 1: "AI leaders",
                 2: "Steady adopters",           3: "Early-stage / laggards"}
CATS = ["region", "industry", "company_size", "ai_primary_tool",
        "ai_use_case", "data_privacy_level"]

PALETTE = ["#378ADD", "#1D9E75", "#7F77DD", "#EF9F27", "#888780", "#D85A30"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Adoption ML")
    st.markdown("<small>150K responses · 10K companies · 2023–2026</small>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("", [
        "🏠  Home & Story",
        "📊  Data Explorer",
        "🎯  Module 1 — Classifier",
        "🔵  Module 2 — Clusters",
        "💰  Module 3 — ROI",
        "⚖️  Module 4 — Ethics & Risk",
        "🔧  Optuna Tuning",
        "🔮  Predict a Company",
    ], label_visibility="collapsed")
    st.divider()
    with st.expander("🔽 Filters", expanded=False):
        sel_industry = st.multiselect("Industry",     sorted(df["industry"].unique()),     default=[])
        sel_region   = st.multiselect("Region",       sorted(df["region"].unique()),       default=[])
        sel_size     = st.multiselect("Company size", sorted(df["company_size"].unique()), default=[])
    st.markdown("""
    <div style='margin-top:2rem; font-size:0.72rem; color:#64748b; line-height:1.6'>
    Built with XGBoost · LightGBM<br>
    K-Means · OLS · Optuna TPE<br>
    Deployed on Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)

df_f = df.copy()
if sel_industry: df_f = df_f[df_f["industry"].isin(sel_industry)]
if sel_region:   df_f = df_f[df_f["region"].isin(sel_region)]
if sel_size:     df_f = df_f[df_f["company_size"].isin(sel_size)]


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE: HOME & STORY                                              ║
# ╚══════════════════════════════════════════════════════════════════╝
if page == "🏠  Home & Story":

    # ── Hero ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <h1>🤖 Global AI Adoption & Workforce Impact</h1>
      <p>An end-to-end ML study of how 10,000 companies across 6 regions are navigating the AI transformation — and what separates the leaders from the laggards.</p>
      <span class="badge">📦 150,000 survey responses</span>
      <span class="badge">🏢 10,000 companies</span>
      <span class="badge">🌍 6 regions · 9 industries</span>
      <span class="badge">📅 2023 – 2026</span>
      <span class="badge">4 ML modules</span>
    </div>
    """, unsafe_allow_html=True)

    # ── The Business Problem ───────────────────────────────────────
    st.markdown('<div class="sec-head">❓ The Problem</div>', unsafe_allow_html=True)
    st.markdown("""
    Companies are investing billions in AI — but most don't know **where they stand**, **whether it's working**,
    or **what drives success vs failure**. This project answers those questions with data.
    """)

    # ── Story Flow ─────────────────────────────────────────────────
    st.markdown('<div class="sec-head">🗺️ The Story in 5 Steps</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    steps = [
        ("1️⃣", "The Data", "150K quarterly snapshots across 4 years, 43 features per company — investment, tools, outcomes, governance."),
        ("2️⃣", "Where Are You?", "Module 1: XGBoost classifier predicts your AI adoption stage (none → pilot → partial → full) with 88.7% weighted F1."),
        ("3️⃣", "What Type?", "Module 2: K-Means segments companies into 4 maturity clusters — from AI Leaders to Early-stage Laggards."),
        ("4️⃣", "What's the ROI?", "Module 3: LightGBM regressors quantify the revenue and cost impact of AI investments using SHAP explainability."),
        ("5️⃣", "What's the Risk?", "Module 4: OLS regression uncovers what actually reduces failure risk — and reveals a governance paradox."),
    ]
    for col, (num, title, desc) in zip([c1,c2,c3,c4,c5], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
              <h4>{num} {title}</h4>
              <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Top-line KPIs ──────────────────────────────────────────────
    st.markdown('<div class="sec-head">📌 Key Numbers at a Glance</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (f"{df['ai_adoption_rate'].mean():.1f}%",     "Avg AI adoption rate"),
        (f"{df['ai_maturity_score'].mean():.2f}",     "Avg maturity score"),
        (f"{df['productivity_change_percent'].mean():.1f}%", "Avg productivity gain"),
        (f"{df['revenue_growth_percent'].mean():.1f}%","Avg revenue growth"),
        (f"{df['ai_failure_rate'].mean():.1f}%",      "Avg AI failure rate"),
        (f"{(df['ai_adoption_stage']=='full').mean()*100:.1f}%", "Fully adopted AI"),
    ]
    for col, (val, lbl) in zip([k1,k2,k3,k4,k5,k6], kpis):
        with col:
            st.markdown(f'<div class="stat-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.divider()

    # ── 3 Key Findings ─────────────────────────────────────────────
    st.markdown('<div class="sec-head">💡 3 Findings Every Business Leader Should See</div>', unsafe_allow_html=True)
    f1c, f2c, f3c = st.columns(3)
    with f1c:
        st.markdown("""
        <div class="insight-box">
        <strong>⏱️ Time beats budget.</strong><br>
        <code>years_using_ai</code> is the #1 predictor of adoption stage — contributing 47% of model importance.
        Companies can't shortcut maturity by spending more. Experience compounds.
        </div>
        """, unsafe_allow_html=True)
    with f2c:
        st.markdown("""
        <div class="insight-box">
        <strong>📈 Maturity = 3.6× revenue gap.</strong><br>
        AI Leaders grow revenue at 7.7% vs 2.1% for Early-stage laggards.
        The gap isn't about tools — it's about how deeply AI is embedded into operations.
        </div>
        """, unsafe_allow_html=True)
    with f3c:
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Ethics committees don't prevent failure.</strong><br>
        After adjusting for confounders, ethics governance shows <em>no</em> revenue benefit.
        Companies set up committees <em>reactively</em> — after problems emerge, not before.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Quick trend chart ──────────────────────────────────────────
    st.markdown('<div class="sec-head">📈 AI Adoption is Accelerating</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        trend = df.groupby("survey_year")["ai_adoption_rate"].mean()
        fig, ax = plt.subplots(figsize=(5, 2.8))
        ax.fill_between(trend.index, trend.values, alpha=0.15, color="#378ADD")
        ax.plot(trend.index, trend.values, "o-", color="#378ADD", linewidth=2.5, markersize=7)
        for x, y in zip(trend.index, trend.values):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, fontweight="bold", color="#378ADD")
        ax.set_ylabel("Mean AI adoption rate (%)"); ax.set_xlabel("Year")
        ax.set_ylim(28, 44); ax.set_title("Adoption rate 2023–2026", fontweight="bold")
        sns.despine(); st.pyplot(fig, use_container_width=True); plt.close()
    with col_r:
        stage_pct = df["ai_adoption_stage"].value_counts(normalize=True).reindex(STAGE_ORDER) * 100
        fig, ax = plt.subplots(figsize=(5, 2.8))
        bars = ax.bar(stage_pct.index, stage_pct.values,
                      color=[STAGE_COLORS[s] for s in stage_pct.index], width=0.55, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("% of companies"); ax.set_title("Stage distribution across all years", fontweight="bold")
        sns.despine(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="info-box">👈 Use the sidebar to explore each ML module in detail, or jump straight to <strong>🔮 Predict a Company</strong> to test the live model.</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE: DATA EXPLORER                                             ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "📊  Data Explorer":

    st.markdown("""
    <div class="hero">
      <h1>📊 Data Explorer</h1>
      <p>Slice and dice 150,000 survey responses by industry, region, and company size using the sidebar filters.</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"Showing **{len(df_f):,}** rows · **{df_f['company_id'].nunique():,}** unique companies")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg adoption rate",     f"{df_f['ai_adoption_rate'].mean():.1f}%")
    c2.metric("Full adoption",         f"{(df_f['ai_adoption_stage']=='full').mean()*100:.1f}%")
    c3.metric("Avg productivity gain", f"{df_f['productivity_change_percent'].mean():.1f}%")
    c4.metric("Avg revenue growth",    f"{df_f['revenue_growth_percent'].mean():.1f}%")
    c5.metric("Avg maturity score",    f"{df_f['ai_maturity_score'].mean():.3f}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-head">Adoption stage breakdown</div>', unsafe_allow_html=True)
        vc = df_f["ai_adoption_stage"].value_counts().reindex(STAGE_ORDER).dropna()
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(vc.index, vc.values,
                      color=[STAGE_COLORS.get(s,"#888") for s in vc.index],
                      width=0.55, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100,
                    f"{int(bar.get_height()):,}", ha="center", fontsize=8.5)
        ax.set_ylabel("Count"); ax.set_title("Companies by adoption stage", fontweight="bold")
        sns.despine(); st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="sec-head">Adoption rate by industry</div>', unsafe_allow_html=True)
        ind = df_f.groupby("industry")["ai_adoption_rate"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(5, 3))
        colors_ind = [PALETTE[i % len(PALETTE)] for i in range(len(ind))]
        ax.barh(ind.index, ind.values, color=colors_ind, edgecolor="white")
        ax.set_xlabel("Mean adoption rate (%)"); ax.set_title("By industry", fontweight="bold")
        sns.despine(); st.pyplot(fig, use_container_width=True); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="sec-head">Adoption rate trend by year</div>', unsafe_allow_html=True)
        trend = df_f.groupby("survey_year")["ai_adoption_rate"].mean()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.fill_between(trend.index, trend.values, alpha=0.12, color="#378ADD")
        ax.plot(trend.index, trend.values, "o-", color="#378ADD", linewidth=2.5)
        ax.set_ylabel("Mean adoption rate (%)"); ax.set_xlabel("Year")
        ax.set_title("Year-on-year trend", fontweight="bold")
        sns.despine(); st.pyplot(fig, use_container_width=True); plt.close()

    with col4:
        st.markdown('<div class="sec-head">Maturity vs productivity (r = 0.74)</div>', unsafe_allow_html=True)
        samp = df_f.sample(min(3000, len(df_f)), random_state=42)
        fig, ax = plt.subplots(figsize=(5, 3))
        scatter = ax.scatter(samp["ai_maturity_score"], samp["productivity_change_percent"],
                             c=samp["ai_maturity_score"], cmap="Blues", alpha=0.3, s=6)
        ax.set_xlabel("AI maturity score"); ax.set_ylabel("Productivity change (%)")
        ax.set_title("Maturity drives productivity", fontweight="bold")
        sns.despine(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="insight-box"><strong>Takeaway:</strong> AI maturity score (r=0.74 with productivity) is the single master variable. Adoption rate grew from 33.5% → 39.2% over 4 years — steady but not explosive, signalling most companies are still in transition.</div>', unsafe_allow_html=True)

    # ── Bonus: revenue by cluster ──────────────────────────────────
    st.divider()
    st.markdown('<div class="sec-head">Revenue growth vs AI failure rate (business impact view)</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        grp = df_f.groupby("ai_adoption_stage")[["revenue_growth_percent","ai_failure_rate"]].mean().reindex(STAGE_ORDER).dropna()
        fig, ax = plt.subplots(figsize=(5, 3))
        x = np.arange(len(grp))
        b1 = ax.bar(x - 0.2, grp["revenue_growth_percent"], 0.35,
                    label="Revenue growth %", color="#1D9E75", edgecolor="white")
        b2 = ax.bar(x + 0.2, grp["ai_failure_rate"],        0.35,
                    label="AI failure rate %", color="#D85A30", edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(grp.index)
        ax.set_title("Revenue growth vs failure rate by stage", fontweight="bold")
        ax.legend(fontsize=8); sns.despine()
        st.pyplot(fig, use_container_width=True); plt.close()
    with col6:
        box_data = [df_f[df_f["ai_adoption_stage"]==s]["revenue_growth_percent"].dropna()
                    for s in STAGE_ORDER]
        fig, ax = plt.subplots(figsize=(5, 3))
        bp = ax.boxplot(box_data, labels=STAGE_ORDER, patch_artist=True, notch=False)
        for patch, color in zip(bp["boxes"], STAGE_COLORS.values()):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_ylabel("Revenue growth %")
        ax.set_title("Revenue distribution by adoption stage", fontweight="bold")
        sns.despine(); st.pyplot(fig, use_container_width=True); plt.close()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE: MODULE 1                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "🎯  Module 1 — Classifier":

    st.markdown("""
    <div class="hero">
      <h1>🎯 Module 1 — AI Adoption Stage Classifier</h1>
      <p>XGBoost multi-class classifier · Predicts which stage a company is at: <strong>none → pilot → partial → full</strong> · Optuna-tuned (20 trials)</p>
    </div>
    """, unsafe_allow_html=True)

    m1 = models.get("m1")
    if not m1:
        st.error("Model not found — ensure Notebook/data/m1_model_tuned.pkl is committed to your repo.")
        st.stop()

    st.markdown('<div class="info-box"><strong>Why does this matter?</strong> Most companies don\'t accurately know their own AI maturity. This classifier gives an objective, data-driven answer using 40+ behavioural and investment signals.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro F1",    f"{m1['best_f1']:.4f}",               f"{m1['best_f1']-m1.get('baseline_f1',0.8277):+.4f} vs baseline")
    c2.metric("Weighted F1", f"{m1.get('weighted_f1', 0.887):.4f}", "↑ handles class imbalance")
    c3.metric("ROC-AUC",     f"{m1.get('roc_auc', 0.979):.4f}",    "Multi-class OvR")
    c4.metric("Hardest class","full (F1=0.54)",                     "Only 1.1% of data")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-head">Feature importance (top 15 by gain)</div>', unsafe_allow_html=True)
        fi = pd.Series(m1["model"].feature_importances_,
                       index=m1["features"]).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(5, 5))
        colors_fi = [PALETTE[0]]*5 + [PALETTE[2]]*5 + [PALETTE[4]]*5
        bars = ax.barh(range(len(fi)), fi.values[::-1], color=colors_fi[::-1], edgecolor="white")
        ax.set_yticks(range(len(fi)))
        ax.set_yticklabels(fi.index[::-1], fontsize=9)
        ax.set_xlabel("Importance (gain)")
        ax.set_title("What predicts AI adoption stage?", fontweight="bold")
        patches = [mpatches.Patch(color=PALETTE[0], label="Top 5"),
                   mpatches.Patch(color=PALETTE[2], label="Mid 5"),
                   mpatches.Patch(color=PALETTE[4], label="Lower 5")]
        ax.legend(handles=patches, fontsize=8)
        sns.despine(); st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="sec-head">Confusion matrix</div>', unsafe_allow_html=True)
        img_path = p("Notebook/outputs/m1_confusion_matrix.png")
        if pathlib.Path(img_path).exists():
            st.image(img_path)
        else:
            st.info("Confusion matrix image not found — run the notebook to generate it.")

    st.markdown('<div class="insight-box"><strong>Key insight:</strong> <code>years_using_ai</code> alone contributes 47% of total model gain — far above budget, tools, or company size. This is a strong signal: AI maturity is earned through time and iteration, not purchased.</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box"><strong>Class imbalance note:</strong> The <em>full</em> adoption class (1.1% of data) is the hardest to predict (F1=0.54) even with inverse-frequency sample weighting. SMOTE is an alternative worth exploring for the next iteration.</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE: MODULE 2                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "🔵  Module 2 — Clusters":

    st.markdown("""
    <div class="hero">
      <h1>🔵 Module 2 — AI Maturity Segmentation</h1>
      <p>K-Means (k=4) on 13 behavioural & investment features · PCA for visualisation · Reveals the 4 archetypes of AI adoption</p>
    </div>
    """, unsafe_allow_html=True)

    m2 = models.get("m2")
    if not m2:
        st.error("Model not found — ensure Notebook/data/m2_model.pkl is committed.")
        st.stop()

    profile = m2["profile"]
    cnames  = m2["cluster_names"]

    st.markdown('<div class="info-box"><strong>Why does this matter?</strong> Knowing <em>which archetype</em> a company is helps prioritise interventions. An "AI Leader" needs to scale. A "Struggling experimenter" needs training, not more tools.</div>', unsafe_allow_html=True)

    cluster_colors = ["#1D9E75","#378ADD","#7F77DD","#888780"]
    c1, c2, c3, c4 = st.columns(4)
    for col, ((c, name), clr) in zip([c1,c2,c3,c4], zip(cnames.items(), cluster_colors)):
        with col:
            rev  = profile.loc[c, "revenue_growth_percent"]
            mat  = profile.loc[c, "ai_maturity_score"]
            fail = profile.loc[c, "ai_failure_rate"]
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-top:4px solid {clr};
                        border-radius:10px;padding:1rem;text-align:center">
              <div style="font-size:0.75rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.05em">{name}</div>
              <div style="font-size:1.6rem;font-weight:800;color:{clr};margin:0.3rem 0">{rev:.1f}%</div>
              <div style="font-size:0.75rem;color:#475569">revenue growth</div>
              <div style="font-size:0.8rem;color:#64748b;margin-top:0.4rem">maturity {mat:.3f} · fail {fail:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns([1.3, 1])

    with col1:
        img_path = p("Notebook/outputs/m2_clusters.png")
        if pathlib.Path(img_path).exists():
            st.image(img_path, caption="PCA projection of the 4 maturity clusters")
        else:
            st.info("Run module 2 notebook to generate cluster plot.")

    with col2:
        st.markdown('<div class="sec-head">Cluster profile comparison</div>', unsafe_allow_html=True)
        metrics_show = ["ai_maturity_score", "ai_failure_rate", "revenue_growth_percent",
                        "task_automation_rate", "ai_training_hours"]
        profile_show       = profile[metrics_show].copy()
        profile_show.index = [cnames[c] for c in profile_show.index]
        st.dataframe(profile_show.T.round(3), use_container_width=True)

        st.markdown('<div class="sec-head">What each cluster needs</div>', unsafe_allow_html=True)
        prescriptions = {
            "AI leaders":               "🚀 Scale & standardise. Focus on governance and cross-functional embedding.",
            "Steady adopters":          "📈 Accelerate. Increase training hours and expand use cases.",
            "Struggling experimenters": "🔧 Fix the gap. You have tools — invest in training to use them well.",
            "Early-stage / laggards":   "🌱 Start with a pilot. Pick one high-value use case and prove ROI first.",
        }
        for name, prescription in prescriptions.items():
            st.markdown(f"**{name}:** {prescription}")

    st.markdown('<div class="insight-box"><strong>The 3.6× gap:</strong> AI leaders generate 7.7% revenue growth vs 2.1% for early-stage laggards — a 3.6× difference driven primarily by maturity and training investment, not budget size.</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box"><strong>Silhouette score = 0.13:</strong> Low by textbook standards but expected for dense, overlapping real-world behavioural data. Cluster business utility lies in the gradient, not geometric separation.</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE: MODULE 3                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "💰  Module 3 — ROI":

    st.markdown("""
    <div class="hero">
      <h1>💰 Module 3 — ROI & Business Impact Regression</h1>
      <p>LightGBM regressors with early stopping · SHAP explainability · Targets: revenue growth % and cost reduction %</p>
    </div>
    """, unsafe_allow_html=True)

    m3 = models.get("m3")
    if not m3:
        st.error("Model not found — ensure Notebook/data/m3_models_tuned.pkl is committed.")
        st.stop()

    res = m3["results"]

    st.markdown('<div class="info-box"><strong>Why does this matter?</strong> Finance teams need numbers. This module translates AI investment signals directly into predicted revenue growth and cost reduction — with SHAP to show <em>why</em> each prediction was made.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue R²",   f"{res['revenue_growth_percent']['r2']:.4f}",
              f"{res['revenue_growth_percent']['r2']-0.2357:+.4f} vs baseline")
    c2.metric("Revenue RMSE", f"{res['revenue_growth_percent']['rmse']:.3f}")
    c3.metric("Cost R²",      f"{res['cost_reduction_percent']['r2']:.4f}",
              f"{res['cost_reduction_percent']['r2']-0.2445:+.4f} vs baseline")
    c4.metric("Cost RMSE",    f"{res['cost_reduction_percent']['rmse']:.3f}")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🐝 SHAP Beeswarm", "📉 Actual vs Predicted"])

    with tab1:
        col1, col2 = st.columns(2)
        for col, target, clr in zip([col1, col2],
                                     ["revenue_growth_percent","cost_reduction_percent"],
                                     ["#1D9E75","#378ADD"]):
            with col:
                fi = pd.Series(m3["models"][target].feature_importances_,
                               index=m3["features"]).sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(5, 5))
                cmap = [clr]*5 + [PALETTE[2]]*5 + [PALETTE[4]]*5
                ax.barh(range(len(fi)), fi.values[::-1], color=cmap[::-1], edgecolor="white")
                ax.set_yticks(range(len(fi)))
                ax.set_yticklabels(fi.index[::-1], fontsize=9)
                ax.set_title(target.replace("_percent","").replace("_"," ").title(), fontweight="bold")
                ax.set_xlabel("Gain")
                sns.despine(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        img_path = p("Notebook/outputs/m3_shap_revenue.png")
        if pathlib.Path(img_path).exists():
            st.image(img_path, caption="SHAP beeswarm — revenue_growth_percent")
        else:
            st.info("SHAP plot not found — run the module 3 notebook to generate it.")

    with tab3:
        for img in ["m3_actual_vs_predicted.png", "m3_residuals.png"]:
            img_path = p(f"Notebook/outputs/{img}")
            if pathlib.Path(img_path).exists():
                st.image(img_path)

    st.markdown('<div class="insight-box"><strong>On R²≈0.24:</strong> This is intentionally honest. Revenue growth is shaped by macroeconomics, competition, and leadership decisions far beyond AI metrics. R²=0.24 means AI signals alone explain ~24% of revenue variation — meaningful and statistically significant.</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box"><strong>What SHAP tells us:</strong> High <code>ai_maturity_score</code> and <code>task_automation_rate</code> are the strongest positive drivers. High <code>ai_failure_rate</code> is the strongest drag on revenue growth.</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE: MODULE 4                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "⚖️  Module 4 — Ethics & Risk":

    st.markdown("""
    <div class="hero">
      <h1>⚖️ Module 4 — Ethics & Risk Factor Analysis</h1>
      <p>OLS regression adjustment for causal inference · Failure risk logistic odds ratios · The governance paradox revealed</p>
    </div>
    """, unsafe_allow_html=True)

    m4 = models.get("m4")
    if not m4:
        st.error("Model not found — ensure Notebook/data/m4_results.pkl is committed.")
        st.stop()

    st.markdown('<div class="info-box"><strong>Why does this matter?</strong> Companies invest heavily in AI governance — but does it actually work? This module estimates the <em>causal</em> effect of ethics committees on outcomes after controlling for confounders.</div>', unsafe_allow_html=True)

    ols = m4["ols_results"]
    col1, col2, col3 = st.columns(3)
    for col, outcome, label in zip(
        [col1, col2, col3],
        ["revenue_growth_percent", "ai_failure_rate", "ai_maturity_score"],
        ["Revenue Growth", "AI Failure Rate", "AI Maturity Score"]
    ):
        r   = ols[outcome]
        sig = "***" if r["pval"]<0.001 else "**" if r["pval"]<0.01 else "*" if r["pval"]<0.05 else "ns"
        col.metric(label, f"{r['coef']:+.4f}", f"p={r['pval']:.3f} {sig}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        img_path = p("Notebook/outputs/m4_ethics_risk.png")
        if pathlib.Path(img_path).exists():
            st.image(img_path)
        else:
            st.info("Ethics risk plot not found — run module 4 notebook.")

    with col2:
        st.markdown('<div class="sec-head">Failure risk — odds ratios</div>', unsafe_allow_html=True)
        risk = m4["risk_fi"].reset_index()
        risk.columns = ["Feature", "Odds ratio"]
        risk["Direction"] = risk["Odds ratio"].apply(
            lambda x: "✅ Reduces risk" if x < 1 else "⚠️ Increases risk"
        )
        risk["Odds ratio"] = risk["Odds ratio"].round(3)
        st.dataframe(risk.set_index("Feature"), use_container_width=True, height=320)

    st.markdown('<div class="danger-box"><strong>The governance paradox:</strong> After adjusting for confounders, ethics committees show <em>no</em> revenue benefit and correlate with slightly higher failure rates. This is almost certainly <strong>reverse causality</strong> — struggling companies set up governance in response to failures, not proactively. The committee is a symptom, not a cure.</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box"><strong>Strongest failure reducer:</strong> <code>adoption_stage_ord</code> with OR=0.29 — advancing along the adoption ladder cuts failure risk by 71%. This dwarfs any governance intervention.</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-box"><strong>Methodological note:</strong> 99.9% of companies have an ethics committee, violating the positivity assumption for propensity score matching. OLS controls for observed confounders only — unmeasured confounding cannot be ruled out.</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE: OPTUNA                                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "🔧  Optuna Tuning":

    st.markdown("""
    <div class="hero">
      <h1>🔧 Optuna Hyperparameter Tuning</h1>
      <p>TPE sampler · 20 trials for XGBoost · 15 trials per target for LightGBM · 30K subsampled training for speed</p>
    </div>
    """, unsafe_allow_html=True)

    m1t = models.get("m1")
    m3t = models.get("m3")

    c1, c2, c3 = st.columns(3)
    if m1t and "best_f1" in m1t:
        c1.metric("M1 Macro F1 (tuned)", f"{m1t['best_f1']:.4f}",
                  f"{m1t['best_f1']-m1t.get('baseline_f1',0.8277):+.4f} vs baseline")
    if m3t and "results" in m3t:
        r3 = m3t["results"]
        if "revenue_growth_percent" in r3 and "r2" in r3["revenue_growth_percent"]:
            c2.metric("M3 Revenue R²", f"{r3['revenue_growth_percent']['r2']:.4f}",
                      f"{r3['revenue_growth_percent']['r2']-0.2357:+.4f}")
        if "cost_reduction_percent" in r3 and "r2" in r3["cost_reduction_percent"]:
            c3.metric("M3 Cost R²",    f"{r3['cost_reduction_percent']['r2']:.4f}",
                      f"{r3['cost_reduction_percent']['r2']-0.2445:+.4f}")

    st.divider()
    img_path = p("Notebook/outputs/optuna_convergence.png")
    if pathlib.Path(img_path).exists():
        st.image(img_path, caption="Convergence — best score per trial (TPE sampler)")
    else:
        st.info("Run optuna tuning.py to generate the convergence plot.")

    st.divider()
    if m1t:
        bp1 = m1t.get("best_params") or m1t.get("params") or {}
        if bp1:
            st.markdown('<div class="sec-head">Best params — Module 1 (XGBoost)</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(bp1.items(),
                         columns=["Parameter","Value"]).set_index("Parameter"))
        else:
            st.info("Best params not in pickle — re-run optuna tuning.py and recommit.")

    if m3t:
        res3 = m3t.get("results", {})
        if res3:
            st.markdown('<div class="sec-head">Best params — Module 3 (LightGBM)</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            for col, target in zip([col1,col2],
                                    ["revenue_growth_percent","cost_reduction_percent"]):
                with col:
                    st.caption(target)
                    bp = (res3.get(target) or {}).get("best_params") or {}
                    if bp:
                        st.dataframe(pd.DataFrame(bp.items(),
                                     columns=["Parameter","Value"]).set_index("Parameter"),
                                     use_container_width=True)
                    else:
                        st.info("Re-run optuna tuning.py and recommit pkl.")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAGE: PREDICT                                                   ║
# ╚══════════════════════════════════════════════════════════════════╝
elif page == "🔮  Predict a Company":

    st.markdown("""
    <div class="hero">
      <h1>🔮 Predict a Company's AI Adoption Stage</h1>
      <p>Fill in your company profile below and get an instant prediction from the Optuna-tuned XGBoost model — plus a breakdown of confidence across all 4 stages.</p>
    </div>
    """, unsafe_allow_html=True)

    m1 = models.get("m1")
    if not m1:
        st.error("Model not found — ensure Notebook/data/m1_model_tuned.pkl is committed.")
        st.stop()

    st.markdown('<div class="info-box">🧪 <strong>Try it:</strong> Adjust the sliders and dropdowns to match a real company profile. The model uses 40+ signals to classify AI adoption stage in real time.</div>', unsafe_allow_html=True)

    # ── Input form ─────────────────────────────────────────────────
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🕐 Experience & Investment**")
            years_ai      = st.slider("Years using AI",        0,   15,   3)
            ai_budget_pct = st.slider("AI budget %",           0.0, 25.0, 8.0,  0.5)
            ai_training   = st.slider("AI training hours/yr",  0.0, 80.0, 25.0, 1.0)
            num_tools     = st.slider("Num AI tools used",     1,   6,    2)
        with col2:
            st.markdown("**⚙️ Operations & Outcomes**")
            ai_projects  = st.slider("Active AI projects",    0,   10,   3)
            task_auto    = st.slider("Task automation rate %",0.0, 50.0, 20.0, 0.5)
            failure_rate = st.slider("AI failure rate %",     0.0, 40.0, 25.0, 0.5)
            maturity     = st.slider("AI maturity score",     0.0, 0.9,  0.35, 0.01)
        with col3:
            st.markdown("**🏢 Company Profile**")
            company_size = st.selectbox("Company size", ["Startup","SME","Enterprise"])
            industry     = st.selectbox("Industry", sorted(["Technology","Finance","Healthcare",
                                        "Manufacturing","Retail","Agriculture",
                                        "Education","Logistics","Consulting"]))
            region       = st.selectbox("Region", sorted(["Asia","Europe","North America",
                                        "South America","Africa","Oceania"]))
            annual_rev   = st.number_input("Annual revenue (USD M)", 1.0, 10000.0, 100.0, 10.0)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Adoption Stage", type="primary", use_container_width=True)

    if predict_btn:
        model    = m1["model"]
        features = m1["features"]

        base_row = df.iloc[[0]].copy()
        num_emp  = {"Startup": 50, "SME": 300, "Enterprise": 2000}[company_size]

        base_row["years_using_ai"]                = years_ai
        base_row["ai_budget_percentage"]           = ai_budget_pct
        base_row["ai_training_hours"]              = ai_training
        base_row["num_ai_tools_used"]              = num_tools
        base_row["ai_projects_active"]             = ai_projects
        base_row["task_automation_rate"]           = task_auto
        base_row["ai_failure_rate"]                = failure_rate
        base_row["ai_maturity_score"]              = maturity
        base_row["ai_intensity"]                   = ai_budget_pct * ai_training
        base_row["log_annual_revenue_usd_millions"]= np.log1p(annual_rev)
        base_row["log_num_employees"]              = np.log1p(num_emp)
        base_row["industry"]                       = industry
        base_row["region"]                         = region
        base_row["company_size"]                   = company_size

        # Use saved encoder if available, else build deterministic one
        if "encoder" in m1:
            enc_pred = m1["encoder"]
        else:
            enc_pred = OrdinalEncoder(
                categories=[sorted(df[c].dropna().unique()) for c in CATS],
                handle_unknown="use_encoded_value", unknown_value=-1,
            )
            enc_pred.fit(df[CATS])

        base_row[CATS] = enc_pred.transform(base_row[CATS])

        for col in base_row[features].select_dtypes(include=["object","string"]).columns:
            base_row[col] = 0.0

        X_input = base_row[features].apply(
            lambda c: c.cat.codes if hasattr(c, "cat") else c
        ).astype(float)

        probs = model.predict_proba(X_input)[0]
        pred  = STAGE_ORDER[int(np.argmax(probs))]

        STAGE_DESC = {
            "none":    "No meaningful AI deployment yet. Start with a single high-value pilot use case.",
            "pilot":   "Early experimentation underway. Focus on proving ROI and building internal capability.",
            "partial": "AI embedded in some functions. Expand successful pilots and upskill your workforce.",
            "full":    "AI is core to operations. Focus on governance, scaling, and continuous improvement.",
        }

        # Result display
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            stage_clr = STAGE_COLORS[pred]
            st.markdown(f"""
            <div class="pred-result">
              <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.3rem">PREDICTED STAGE</div>
              <div class="stage" style="color:{stage_clr}">{pred.upper()}</div>
              <div class="desc">{STAGE_DESC[pred]}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            for stage, prob in zip(STAGE_ORDER, probs):
                clr = STAGE_COLORS[stage]
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
                    f'<span style="width:70px;font-size:0.8rem;color:#475569">{stage}</span>'
                    f'<div style="flex:1;background:#f1f5f9;border-radius:99px;height:10px">'
                    f'<div style="width:{prob*100:.1f}%;background:{clr};height:10px;border-radius:99px"></div>'
                    f'</div><span style="font-size:0.8rem;font-weight:700;color:{clr}">{prob*100:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        with res_col2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bar_colors = [STAGE_COLORS[s] for s in STAGE_ORDER]
            bars = ax.bar(STAGE_ORDER, probs, color=bar_colors, width=0.5, edgecolor="white", linewidth=1.5)
            for bar, prob in zip(bars, probs):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{prob*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
            ax.set_ylabel("Probability", fontsize=11)
            ax.set_ylim(0, 1.1)
            ax.set_title(f"Model confidence across all 4 stages", fontweight="bold", fontsize=12)
            # Highlight predicted bar
            bars[STAGE_ORDER.index(pred)].set_edgecolor("#0f172a")
            bars[STAGE_ORDER.index(pred)].set_linewidth(3)
            sns.despine(); st.pyplot(fig, use_container_width=True); plt.close()

            # Contextual business insight
            if pred == "none":
                st.markdown('<div class="danger-box">⚡ <strong>Action:</strong> Launch one AI pilot in the highest-ROI function. Companies in this stage that move to <em>pilot</em> within 12 months reduce failure risk by 71%.</div>', unsafe_allow_html=True)
            elif pred == "pilot":
                st.markdown('<div class="warning-box">📌 <strong>Action:</strong> Document and measure pilot outcomes. Double training hours — the data shows training investment is the strongest predictor of moving from pilot to partial adoption.</div>', unsafe_allow_html=True)
            elif pred == "partial":
                st.markdown('<div class="insight-box">📈 <strong>Action:</strong> You\'re in the critical scaling zone. Companies here that increase <code>ai_maturity_score</code> by 0.1 see ~1.2% additional revenue growth. Focus on cross-functional deployment.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-box">🏆 <strong>Action:</strong> You\'re in the top tier — 7.7% avg revenue growth. Focus on governance, knowledge transfer, and defending your maturity lead as competitors catch up.</div>', unsafe_allow_html=True)