"""
NovaPay Fraud Scorer - Streamlit demo app
==========================================

What it does
------------
1. Enter (or pick a preset) transaction -> fraud probability.
2. Probability -> tiered decision: APPROVE / REVIEW / BLOCK.
3. SHAP explanation of which features drove the score.
4. Model-performance tab with your real notebook metrics.

Run
---
    pip install streamlit pandas numpy scikit-learn xgboost lightgbm shap joblib matplotlib
    streamlit run streamlit_app.py

Wire in the real model
----------------------
Save your Random Forest pipeline once, in model.ipynb:

    from sklearn.pipeline import Pipeline
    import joblib, os
    pipe = Pipeline([("pre", preprocessor), ("clf", rf_model)])
    pipe.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/rf_fraud_pipeline.pkl")

The app then loads it automatically. Until then it runs on a rule-based
fallback so the demo always works.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- config
st.set_page_config(page_title="NovaPay Fraud Scorer", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rf_fraud_pipeline.pkl")
REVIEW_THRESHOLD = 0.30
BLOCK_THRESHOLD = 0.70

# ---------------------------------------------------------------- styling
# A restrained, minimalist theme: neutral greys, one ink accent, lots of
# whitespace, system fonts, and Streamlit's default chrome hidden.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1A1D23;
}
.stApp { background: #FFFFFF; }

/* hide Streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none; }

/* roomy, centred column */
.block-container { max-width: 880px; padding-top: 3rem; padding-bottom: 4rem; }

/* type */
h1 { font-weight: 700; font-size: 1.9rem; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
.subtle { color: #8A9099; font-size: 0.9rem; }

/* primary button: flat ink */
.stButton > button {
    background: #1A1D23; color: #FFFFFF; border: none; border-radius: 10px;
    padding: 0.55rem 1.5rem; font-weight: 600; font-size: 0.95rem; transition: background .15s ease;
}
.stButton > button:hover { background: #3A3F47; color: #FFFFFF; }
.stButton > button:focus { box-shadow: none; color: #FFFFFF; }

/* inputs: soft, quiet */
[data-baseweb="input"], [data-baseweb="select"] > div { border-radius: 8px; }
[data-testid="stWidgetLabel"] p { color: #6B7280; font-size: 0.82rem; font-weight: 500; }

/* slider + progress in ink, not default red */
[data-testid="stProgress"] > div > div > div { background-color: #1A1D23; }
[data-baseweb="slider"] [role="slider"] { background: #1A1D23; }

/* tabs: quiet, underline on active */
[data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #ECEEF1; }
[data-baseweb="tab"] { padding: 0.4rem 0; font-weight: 500; color: #8A9099; }
[data-baseweb="tab"][aria-selected="true"] { color: #1A1D23; }
[data-baseweb="tab-highlight"] { background-color: #1A1D23; }

/* thin divider */
hr { border: none; border-top: 1px solid #ECEEF1; margin: 1.6rem 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- model loading
@st.cache_resource
def load_model():
    try:
        import joblib
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
    except Exception:
        return None
    return None


def mock_score(row: dict) -> float:
    """Fallback scorer built from the EDA thresholds, so the demo always runs."""
    score = 0.05
    if row.get("txn_velocity_1h", 0) >= 3:
        score += 0.45
    if row.get("device_trust_score", 1.0) < 0.3:
        score += 0.35
    if row.get("ip_risk_score", 0) > 0.8:
        score += 0.30
    elif row.get("ip_risk_score", 0) > 0.7:
        score += 0.10
    if row.get("account_age_days", 9999) < 90:
        score += 0.20
    if 1000 <= row.get("amount_usd", 0) <= 2000:
        score += 0.20
    return float(min(score, 0.99))


# ---------------------------------------------------------------- inputs
def build_inputs():
    """The form is always visible; presets simply pre-fill it via session_state keys."""
    CHANNELS, KYC = ["web", "mobile", "atm"], ["low", "standard", "enhanced"]
    c1, c2, c3 = st.columns(3)
    with c1:
        amount_usd = st.number_input("Amount (USD)", min_value=0.0, step=50.0, key="amount_usd")
        txn_velocity_1h = st.number_input("Transactions, last 1h", min_value=0, key="txn_velocity_1h")
        txn_velocity_24h = st.number_input("Transactions, last 24h", min_value=0, key="txn_velocity_24h")
    with c2:
        ip_risk_score = st.slider("IP risk score", 0.0, 1.0, step=0.01, key="ip_risk_score")
        device_trust_score = st.slider("Device trust score", 0.0, 1.0, step=0.01, key="device_trust_score")
        account_age_days = st.number_input("Account age (days)", min_value=0, key="account_age_days")
    with c3:
        hour = st.slider("Hour of day", 0, 23, key="hour_of_day")
        channel = st.selectbox("Channel", CHANNELS, key="channel")
        kyc_tier = st.selectbox("KYC tier", KYC, key="kyc_tier")
    return {
        "amount_usd": amount_usd, "txn_velocity_1h": txn_velocity_1h,
        "txn_velocity_24h": txn_velocity_24h, "ip_risk_score": ip_risk_score,
        "device_trust_score": device_trust_score, "account_age_days": account_age_days,
        "hour_of_day": hour, "is_night": 1 if hour in range(0, 6) else 0,
        "channel": channel, "kyc_tier": kyc_tier,
    }


DEFAULTS = {
    "amount_usd": 1650.0, "txn_velocity_1h": 7, "txn_velocity_24h": 19,
    "ip_risk_score": 0.95, "device_trust_score": 0.12, "account_age_days": 5,
    "hour_of_day": 2, "channel": "web", "kyc_tier": "low",
}

PRESETS = {
    "Likely fraud": dict(DEFAULTS),
    "Likely legitimate": {
        "amount_usd": 120.0, "txn_velocity_1h": 1, "txn_velocity_24h": 3,
        "ip_risk_score": 0.08, "device_trust_score": 0.94, "account_age_days": 880,
        "hour_of_day": 14, "channel": "mobile", "kyc_tier": "enhanced",
    },
}


# ---------------------------------------------------------------- prediction
def predict(model, row: dict) -> float:
    if model is None:
        return mock_score(row)
    try:
        return float(model.predict_proba(pd.DataFrame([row]))[0][1])
    except Exception:
        return mock_score(row)


# muted, non-garish decision colours: (text, light tint)
DECISIONS = {
    "BLOCK":   ("#B42318", "#FEF3F2"),
    "REVIEW":  ("#B54708", "#FFFAEB"),
    "APPROVE": ("#067647", "#ECFDF3"),
}


def decision_for(prob: float) -> str:
    if prob >= BLOCK_THRESHOLD:
        return "BLOCK"
    if prob >= REVIEW_THRESHOLD:
        return "REVIEW"
    return "APPROVE"


def show_shap(model, row: dict):
    try:
        import shap, matplotlib.pyplot as plt
        X = pd.DataFrame([row])
        if hasattr(model, "named_steps"):
            pre, est = model[:-1], model[-1]
            X_trans = pre.transform(X)
        else:
            est, X_trans = model, X.values
        sv = shap.TreeExplainer(est)(X_trans)
        fig = plt.figure()
        shap.plots.waterfall(sv[0], max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)
    except Exception:
        st.markdown("<p class='subtle'>Top contributing signals</p>", unsafe_allow_html=True)
        signals = sorted(
            [("Velocity burst (1h)", 1.0 if row.get("txn_velocity_1h", 0) >= 3 else 0.0),
             ("High IP risk", row.get("ip_risk_score", 0)),
             ("Low device trust", 1 - row.get("device_trust_score", 1)),
             ("New account", 1.0 if row.get("account_age_days", 9999) < 90 else 0.0)],
            key=lambda t: t[1], reverse=True)
        for name, w in signals[:4]:
            st.markdown(f"<div style='padding:2px 0;color:#3A3F47'>{name}"
                        f"<span style='float:right;color:#8A9099'>{w:.2f}</span></div>",
                        unsafe_allow_html=True)


# ---------------------------------------------------------------- UI
st.markdown("<h1>NovaPay Fraud Scorer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtle'>Real-time transaction risk scoring with explainability</p>", unsafe_allow_html=True)

model = load_model()
mode = "Live model" if model is not None else "Demo mode (rule-based fallback)"
st.markdown(f"<p class='subtle' style='margin-top:-4px'>&#9679; {mode}</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

tab_score, tab_perf, tab_about = st.tabs(["Score", "Performance", "About"])

with tab_score:
    for _k, _v in DEFAULTS.items():
        st.session_state.setdefault(_k, _v)

    def apply_preset():
        name = st.session_state.get("preset_choice")
        if name in PRESETS:
            for k, v in PRESETS[name].items():
                st.session_state[k] = v

    pc, _ = st.columns([2, 3])
    with pc:
        st.selectbox("Preset", ["Custom"] + list(PRESETS.keys()),
                     key="preset_choice", on_change=apply_preset)

    row = build_inputs()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    go = st.button("Score transaction")

    if go:
        prob = predict(model, row)
        label = decision_for(prob)
        col, tint = DECISIONS[label]
        st.markdown("<hr>", unsafe_allow_html=True)
        left, right = st.columns([1, 1.3])
        with left:
            st.markdown("<p class='subtle'>Fraud probability</p>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:54px;font-weight:700;line-height:1;"
                        f"letter-spacing:-0.02em'>{prob:.0%}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='margin-top:14px;display:inline-block;padding:7px 16px;"
                f"border-radius:999px;background:{tint};color:{col};font-weight:700;"
                f"font-size:13px;letter-spacing:0.06em'>{label}</div>", unsafe_allow_html=True)
            st.markdown(f"<p class='subtle' style='margin-top:14px'>Approve &lt; {REVIEW_THRESHOLD:.0%}"
                        f" &nbsp;·&nbsp; Block &#8805; {BLOCK_THRESHOLD:.0%}</p>", unsafe_allow_html=True)
        with right:
            st.progress(min(prob, 1.0))
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            show_shap(model, row)

with tab_perf:
    st.markdown("<p class='subtle'>Fraud-class metrics &#183; held-out test set "
                "(2,240 transactions, 305 fraud)</p>", unsafe_allow_html=True)
    perf = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "LightGBM", "XGBoost"],
        "Precision": [0.76, 0.96, 0.93, 0.88],
        "Recall": [0.95, 0.91, 0.91, 0.91],
        "F1": [0.84, 0.94, 0.92, 0.90],
        "ROC-AUC": [0.981, 0.977, 0.972, 0.968],
    })
    st.dataframe(perf, hide_index=True, use_container_width=True)
    st.markdown("<p class='subtle'>Recommended: Random Forest — best precision (0.96) "
                "and F1 (0.94) with strong recall (0.91).</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p class='subtle'>Random Forest confusion matrix</p>", unsafe_allow_html=True)
    cm = pd.DataFrame([[1924, 11], [26, 279]],
                      index=["Actual: Legit", "Actual: Fraud"],
                      columns=["Pred: Legit", "Pred: Fraud"])
    st.table(cm)
    st.markdown("<p class='subtle'>Caught 279 of 305 frauds (91% recall) with only 11 false "
                "alarms (96% precision). Time-based validation showed PR-AUC ranging "
                "0.03 → 0.95 across periods, so monitor drift and retrain on a schedule.</p>",
                unsafe_allow_html=True)

with tab_about:
    st.markdown("""
NovaPay is a cross-border money transfer company. This model scores each
transaction's fraud risk in real time so analysts focus on the cases that matter.

**Top signals** &nbsp; transaction velocity (1h / 24h), IP risk, internal risk,
device trust, account age.

**Explainability** &nbsp; every flag carries a SHAP breakdown for analysts and regulators.

**Imbalance** &nbsp; baseline, class-weights, undersampling and SMOTE all tested — within
noise of each other, so the simpler baseline was kept.

**Production path** &nbsp; the same model served via FastAPI in Docker, with Evidently
monitoring drift.

<p class='subtle'>github.com/Opeyemi-io</p>
""", unsafe_allow_html=True)