"""
Bitcoin Price Forecasting — Multi-Page Streamlit Application
=============================================================
Pages   : 1. Home / Data Upload
          2. Exploratory Data Analysis (EDA)
          3. Statistical Analysis
          4. Forecasting Engine
          5. Model Insights

Color Palette : NixtNode-inspired deep purple/violet on near-black
                Background — #0A0A10  (near-black)
                Surface    — #0F0F18  (dark card)
                Purple Hi  — #9B6FFF  (vivid violet accent)
                Purple Lo  — #6B3FBF  (deep purple)
                Glow       — rgba(155,111,255,0.18) (ambient halo)
                Text main  — #F0F0F8
                Text muted — #7878A0
                Green      — #5BFFA0
                Red        — #FF5B7A

Font : Public Sans (Google Fonts)

Author : AI Engineering Curriculum
Models : Model 1 (Facebook Prophet) | Model 3 (Prophet + XGBoost Hybrid)
"""

import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ─── Page config (must be first) ─────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Forecasting Lab",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Color constants — NixtNode deep-purple palette ──────────────────────────
C_PRIMARY             = "#9B6FFF"   # Primary violet accent
C_DEEP_PURPLE         = "#6B3ED9"
C_DEEP_LAVENDER       = "#6F7BFF"
C_SOFT_LAVENDER       = "#C4A7FF"   # Soft lavender
C_SOFT_LILAC          = "#BD9EFF"
C_GREEN               = "#5BFFA0"   # Bullish green
C_RED                 = "#FF5B7A"   # Bearish red
C_LIGHT_PASTEL_PURPLE = "#D8C7FF"
C_MUTED               = "#7878A0"   # Muted text

# ─── Global CSS — NixtNode deep-purple / Public Sans ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,300;1,400&display=swap');

:root {
    --accent      : #9B6FFF;
    --accent-dim  : #6B3FBF;
    --accent-glow : rgba(155,111,255,0.15);
    --bg-base     : #0A0A10;
    --bg-card     : #0F0F1A;
    --bg-panel    : #151520;
    --bg-border   : #1E1E30;
    --text-main   : #F0F0F8;
    --text-muted  : #7878A0;
    --green       : #5BFFA0;
    --red         : #FF5B7A;
    --blue        : #5BC8FF;
}

/* ── Base ────────────────────────────────────────────────────────────────── */
html, body, .stApp, .main,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: var(--bg-base) !important;
    font-family: 'Public Sans', sans-serif !important;
    color: var(--text-main);
}

/* Ambient purple radial glow behind content — mimics the NixtNode hero bg */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -10%;
    left: 50%;
    transform: translateX(-50%);
    width: 900px;
    height: 600px;
    background: radial-gradient(ellipse at center,
        rgba(120,60,220,0.30) 0%,
        rgba(80,30,160,0.12) 45%,
        transparent 70%);
    pointer-events: none;
    z-index: 0;
    border-radius: 50%;
}

/* Ensure content sits above glow layer */
[data-testid="stMain"] { position: relative; z-index: 1; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--bg-border) !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Public Sans', sans-serif !important;
}

/* ── Page header ─────────────────────────────────────────────────────────── */
.page-header {
    background: linear-gradient(135deg,
        rgba(155,111,255,0.08) 0%,
        rgba(107,63,191,0.04) 60%,
        transparent 100%);
    border: 1px solid var(--bg-border);
    border-left: 3px solid var(--accent);
    border-radius: 0 12px 12px 0;
    padding: 1.4rem 2rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.page-header::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(155,111,255,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.page-header h1 {
    font-family: 'Public Sans', sans-serif;
    font-weight: 800;
    font-size: 1.7rem;
    color: var(--text-main);
    margin: 0 0 0.25rem;
    letter-spacing: -0.5px;
}
.page-header h1 span { color: var(--accent); }
.page-header p {
    color: var(--text-muted);
    margin: 0;
    font-size: 0.88rem;
    font-weight: 400;
}

/* ── KPI card ────────────────────────────────────────────────────────────── */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--bg-border);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 0.4rem;
    transition: border-color 0.2s ease;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(155,111,255,0.20) 0%, transparent 70%);
    pointer-events: none;
}
.kpi-card:hover { border-color: rgba(155,111,255,0.4); }
.kpi-label {
    color: var(--text-muted);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-weight: 600;
}
.kpi-value {
    color: var(--text-main);
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0.3rem 0 0;
    font-family: 'Public Sans', sans-serif;
    letter-spacing: -0.5px;
}
.kpi-value span { color: var(--accent); }
.kpi-sub { color: var(--text-muted); font-size: 0.68rem; margin-top: 0.15rem; font-weight: 400; }

/* ── Section title ───────────────────────────────────────────────────────── */
.section-title {
    font-family: 'Public Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 2.8px;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--bg-border), transparent);
}

/* ── Insight / info box ──────────────────────────────────────────────────── */
.insight-box {
    background: rgba(155,111,255,0.05);
    border: 1px solid rgba(155,111,255,0.2);
    border-left: 3px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.1rem;
    margin: 0.8rem 0;
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.65;
    font-weight: 400;
}
.insight-box strong { color: var(--accent); font-weight: 600; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    width: 100%;
    background: var(--bg-panel) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--bg-border) !important;
    border-radius: 10px !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    font-family: 'Public Sans', sans-serif !important;
    letter-spacing: 0.2px;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    background: rgba(155,111,255,0.15) !important;
    border-color: var(--accent) !important;
    color: var(--text-main) !important;
}

/* Active nav pill (buttons in sidebar that are "selected") */
.stButton > button:focus,
.stButton > button:active {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(155,111,255,0.25) !important;
}

/* ── Generate Forecast — full accent button ──────────────────────────────── */
/* Target the last button in sidebar (Generate Forecast) */
[data-testid="stSidebar"] .stButton:last-of-type > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stButton:last-of-type > button:hover {
    background: var(--accent-dim) !important;
    border: none !important;
}

/* ── Widget labels ───────────────────────────────────────────────────────── */
label, .stSelectbox label, .stSlider label {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    font-family: 'Public Sans', sans-serif !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.stSelectbox > div > div {
    background: var(--bg-panel) !important;
    border-color: var(--bg-border) !important;
    color: var(--text-main) !important;
    border-radius: 10px !important;
    font-family: 'Public Sans', sans-serif !important;
}
[data-testid="stFileUploader"] {
    background: var(--bg-panel);
    border: 1px dashed var(--bg-border);
    border-radius: 12px;
    padding: 0.5rem;
}

/* ── Upload placeholder ──────────────────────────────────────────────────── */
.upload-cta {
    text-align: center;
    padding: 3.5rem 2rem;
    border: 1px solid var(--bg-border);
    border-radius: 20px;
    background: var(--bg-card);
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.upload-cta::before {
    content: '';
    position: absolute;
    top: -80px; left: 50%; transform: translateX(-50%);
    width: 400px; height: 300px;
    background: radial-gradient(ellipse, rgba(155,111,255,0.15) 0%, transparent 65%);
    pointer-events: none;
}
.upload-cta .big-icon { font-size: 3rem; margin-bottom: 0.9rem; }
.upload-cta h2 {
    color: var(--text-main);
    font-family: 'Public Sans', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    letter-spacing: -0.3px;
}
.upload-cta h2 span { color: var(--accent); }
.upload-cta p { color: var(--text-muted); font-size: 0.88rem; line-height: 1.7; }

/* ── Sidebar brand ───────────────────────────────────────────────────────── */
.btc-logo {
    text-align: center;
    padding: 1.2rem 0 0.3rem;
    font-size: 2rem;
    letter-spacing: 2px;
    color: var(--accent);
    font-weight: 800;
    font-family: 'Public Sans', sans-serif;
}
.btc-brand {
    text-align: center;
    font-family: 'Public Sans', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text-main);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.1rem;
}
.btc-tagline {
    text-align: center;
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: var(--bg-border) !important; }

/* ── Markdown headings ───────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Public Sans', sans-serif !important;
    color: var(--text-main) !important;
    font-weight: 700 !important;
}

/* ── DataFrame table ─────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--bg-border) !important;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ─── Plotly default layout ────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="#0F0F1A",
    plot_bgcolor="#0A0A12",
    font=dict(family="Public Sans, sans-serif", color="#F0F0F8", size=12),
    xaxis=dict(gridcolor="#1A1A28", linecolor="#1E1E30"),
    yaxis=dict(gridcolor="#1A1A28", linecolor="#1E1E30"),
    margin=dict(t=55, b=30, l=10, r=10),
    colorway=[C_PRIMARY, C_GREEN, C_LIGHT_PASTEL_PURPLE, C_RED, C_SOFT_LAVENDER, C_DEEP_PURPLE,
              C_SOFT_LILAC, C_DEEP_LAVENDER, C_MUTED],
)

def apply_layout(fig, title="", height=420):
    """Apply global BTC theme to a Plotly figure."""
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", font=dict(size=16, color="#F0F0F8")),
        height=height,
    )
    return fig

# ─── Shared UI helpers ────────────────────────────────────────────────────────

def page_header(title, subtitle):
    st.markdown(f'<div class="page-header"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)

def kpi_row(metrics):
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            sub = f'<div class="kpi-sub">{m["sub"]}</div>' if m.get("sub") else ""
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{m["label"]}</div>'
                f'<div class="kpi-value">{m["value"]}</div>{sub}</div>',
                unsafe_allow_html=True,
            )

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

def no_data_gate():
    st.markdown(
        '<div class="upload-cta">'
        '<div class="big-icon">₿</div>'
        '<h2>No Data Loaded</h2>'
        '<p>Go to the <strong>Home</strong> page and upload a BTC/USD CSV file<br>'
        'to unlock analysis, statistics and forecasting across all pages.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_btc_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Auto-detect and load Kaggle-style BTC CSVs.
    Supports Binance 1-min bars (aggregated to daily) and standard daily OHLCV.
    Returns a clean daily DataFrame indexed by datetime.
    """
    raw = pd.read_csv(io.BytesIO(file_bytes))

    # Locate timestamp column
    ts_candidates = [c for c in raw.columns
                     if any(k in c.lower() for k in ["time", "date", "timestamp"])]
    if not ts_candidates:
        raise ValueError("No timestamp column found. Expected 'Date', 'Timestamp', or 'Open time'.")

    ts_col = ts_candidates[0]
    raw[ts_col] = pd.to_datetime(raw[ts_col], errors="coerce")
    raw = raw.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    # Detect frequency and aggregate if sub-hourly (Binance 1-min format)
    median_diff_s = raw[ts_col].diff().dropna().dt.total_seconds().median()
    if median_diff_s < 3600:
        raw["Date"] = raw[ts_col].dt.date
        daily = raw.groupby("Date").agg(
            Open=("Open","first"), High=("High","max"),
            Low=("Low","min"),    Close=("Close","last"),
            Volume=("Volume","sum"),
        ).reset_index()
        daily["Date"] = pd.to_datetime(daily["Date"])
    else:
        daily = raw.rename(columns={ts_col: "Date"})
        daily.columns = [c.strip().title() if c != "Date" else c for c in daily.columns]
        daily["Date"] = pd.to_datetime(daily["Date"])

    daily = daily.set_index("Date").sort_index()

    # Fill calendar gaps, forward-fill, drop zeros
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range).ffill().dropna()
    if "Close" in daily.columns:
        daily = daily[daily["Close"] > 0]

    if len(daily) < 60:
        raise ValueError(
            f"Only {len(daily)} days after cleaning — need ≥60. Upload a longer CSV."
        )
    return daily


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame, price_col: str):
    """
    Build feature matrix for the XGBoost residual corrector.
    Features: time calendar, lags (1/7/14/30), rolling stats, RSI(14), OHLC spreads.
    Returns (feature_df, list_of_feature_names).
    """
    fe = df[[price_col]].copy()

    # Calendar
    fe["day_of_week"] = fe.index.dayofweek
    fe["month"]       = fe.index.month
    fe["quarter"]     = fe.index.quarter
    fe["day_of_year"] = fe.index.dayofyear

    # Lagged prices
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        if lag < len(fe):
            fe[f"lag_{lag}"] = fe[price_col].shift(lag)

    # Rolling statistics
    for w in [7, 14, 30]:
        if w < len(fe):
            fe[f"sma_{w}"] = fe[price_col].rolling(w).mean()
            fe[f"ema_{w}"] = fe[price_col].ewm(span=w).mean()
            fe[f"std_{w}"] = fe[price_col].rolling(w).std()
            fe[f"roc_{w}"] = fe[price_col].pct_change(w)

    # RSI(14) momentum
    if len(fe) > 14:
        delta = fe[price_col].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        fe["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # OHLC spread features
    if all(c in df.columns for c in ["Open","High","Low","Close"]):
        fe["hl_spread"] = (df["High"] - df["Low"]).shift(1)
        fe["oc_spread"] = (df["Close"] - df["Open"]).shift(1)
        fe["hl_pct"]    = (fe["hl_spread"] / df["Close"]).shift(1)

    fe = fe.dropna()
    feat_cols = [c for c in fe.columns if c != price_col]
    return fe, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_prophet(train, test, horizon, ci_pct):
    """Fit Prophet, walk-forward backtest on test, forecast horizon days."""
    from prophet import Prophet

    prophet_train = train.reset_index()
    prophet_train.columns = ["ds", "y"]

    model = Prophet(
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10,
        seasonality_mode="multiplicative",
        weekly_seasonality=True,
        yearly_seasonality=(len(train) > 365),
        daily_seasonality=False,
        interval_width=ci_pct / 100,
        uncertainty_samples=300,
    )
    if len(train) > 60:
        model.add_seasonality(name="monthly", period=30, fourier_order=10)
    model.fit(prophet_train)

    future   = model.make_future_dataframe(periods=len(test) + horizon, freq="D")
    forecast = model.predict(future)

    test_fc = forecast[forecast["ds"].isin(test.index)]
    y_pred  = test_fc["yhat"].values
    y_true  = test.reindex(pd.to_datetime(test_fc["ds"])).values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)

    future_rows = forecast[forecast["ds"] > train.index[-1]]
    return {
        "model_name": "Model 1: Prophet",
        "mae": mae, "rmse": rmse, "mape": mape,
        "test_dates":   pd.to_datetime(test_fc["ds"]).values,
        "test_pred":    y_pred,
        "test_lower":   test_fc["yhat_lower"].values,
        "test_upper":   test_fc["yhat_upper"].values,
        "future_dates": future_rows["ds"].values,
        "future_pred":  future_rows["yhat"].values,
        "future_lower": future_rows["yhat_lower"].values,
        "future_upper": future_rows["yhat_upper"].values,
    }


def run_hybrid(df, train, test, price_col, horizon, ci_pct):
    """Prophet trend + XGBoost residual correction hybrid model."""
    from prophet import Prophet
    import xgboost as xgb

    # ── Layer 1: Prophet trend ────────────────────────────────────────────────
    prophet_train = train.reset_index()
    prophet_train.columns = ["ds", "y"]
    m_p = Prophet(
        changepoint_prior_scale=0.15,
        seasonality_mode="multiplicative",
        weekly_seasonality=True,
        yearly_seasonality=(len(train) > 365),
        daily_seasonality=False,
        interval_width=ci_pct / 100,
        uncertainty_samples=0,
    )
    m_p.fit(prophet_train)

    train_fc  = m_p.predict(m_p.make_future_dataframe(periods=0))
    residuals = train.values - train_fc["yhat"].values

    # ── Layer 2: XGBoost residual corrector ───────────────────────────────────
    df_feat, feat_cols = engineer_features(df, price_col)
    common_idx = df_feat.index.intersection(train.index)
    res_series = pd.Series(residuals, index=train.index).reindex(common_idx).dropna()
    X_train    = df_feat.loc[res_series.index, feat_cols]
    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)

    m_xgb = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        random_state=42, verbosity=0,
    )
    m_xgb.fit(X_train_s, res_series)

    # Test-set hybrid prediction
    test_prophet_fc = m_p.predict(pd.DataFrame({"ds": test.index}))["yhat"].values
    test_feat_df    = df_feat.loc[df_feat.index.intersection(test.index)].dropna()
    X_test_s        = scaler.transform(test_feat_df[feat_cols])
    xgb_corr        = m_xgb.predict(X_test_s)

    hybrid_idx  = test_feat_df.index
    hybrid_pred = test_prophet_fc[:len(hybrid_idx)] + xgb_corr
    hybrid_true = test.reindex(hybrid_idx).values

    mae  = mean_absolute_error(hybrid_true, hybrid_pred)
    rmse = float(np.sqrt(mean_squared_error(hybrid_true, hybrid_pred)))
    mape = float(np.mean(np.abs((hybrid_true - hybrid_pred) / (hybrid_true + 1e-9))) * 100)

    # CI from bootstrapped residual std
    in_sample_std = np.std(residuals)
    z_val = {80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}.get(int(ci_pct), 1.960)
    ci_half = z_val * in_sample_std

    # Future forecast
    future_dates  = pd.date_range(test.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_fc     = m_p.predict(pd.DataFrame({"ds": future_dates}))["yhat"].values
    last_feat     = df_feat.loc[df_feat.index <= test.index[-1]].tail(1)
    xgb_fut_corr  = float(m_xgb.predict(scaler.transform(last_feat[feat_cols]))[0]) if len(last_feat) else 0.0
    future_pred   = future_fc + xgb_fut_corr

    return {
        "model_name": "Model 2: Prophet + XGBoost Hybrid",
        "mae": mae, "rmse": rmse, "mape": mape,
        "test_dates":   hybrid_idx.values,
        "test_pred":    hybrid_pred,
        "test_lower":   hybrid_pred - ci_half,
        "test_upper":   hybrid_pred + ci_half,
        "future_dates": future_dates.values,
        "future_pred":  future_pred,
        "future_lower": future_pred - ci_half,
        "future_upper": future_pred + ci_half,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

PAGES = [
    ("₿",  "Home",           "Upload data & dataset overview"),
    ("📊", "EDA",            "Charts, OHLC, volume & returns"),
    ("🔬", "Statistics",     "Stationarity, volatility & correlations"),
    ("🔮", "Forecasting",    "Train models & generate forecasts"),
    ("💡", "Model Insights", "Residuals, diagnostics & feature importance"),
]

with st.sidebar:
    st.markdown('<div class="btc-logo">₿ BTC</div>', unsafe_allow_html=True)
    st.markdown('<div class="btc-brand">TIME SERIES PROJECT</div>', unsafe_allow_html=True)
    st.markdown('<div class="btc-tagline">Analysis and Forecasting</div>', unsafe_allow_html=True)
    st.divider()

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    for icon, name, desc in PAGES:
        if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True):
            st.session_state.page = name
            st.rerun()

    # st.divider()
    # st.markdown(
    #     '<p style="color:#8888AA;font-size:0.72rem;text-align:center;line-height:1.5">'
    #     '⚠️ Educational use only.<br>Not financial advice.</p>',
    #     unsafe_allow_html=True,
    # )

PAGE = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

if PAGE == "Home":
    page_header("Bitcoin Price Forecasting",
                "Upload your dataset · Explore · Analyse · Forecast")

    uploaded = st.file_uploader("Upload a BTC/USD CSV file", type=["csv"],
                                  help="Kaggle daily OHLCV or Binance 1-min bars.")

    if uploaded is not None:
        try:
            with st.spinner("⛏  Loading and cleaning data…"):
                df = load_btc_csv(uploaded.read(), uploaded.name)
                st.session_state["df"] = df
            st.success(f"✅  Loaded **{len(df):,} daily bars** from `{uploaded.name}`")
        except Exception as e:
            st.error(f"**CSV Error:** {e}")
            st.info("Ensure your CSV has a date/time column and OHLCV columns.")
            st.stop()
    elif "df" in st.session_state:
        df = st.session_state["df"]
        st.info("📂 Using previously loaded dataset. Re-upload to refresh.")
    else:
        st.markdown("""
        <div class="upload-cta">
            <div class="big-icon">📂</div>
            <h2>Upload Your <span>BTC</span> Dataset</h2>
            <p>Drop a <code>.csv</code> file above — Kaggle daily OHLCV or Binance 1-min bars.<br>
            The app auto-detects the format and aggregates to daily bars automatically.</p>
        </div>""", unsafe_allow_html=True)

        section_title("Supported CSV Formats")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Standard Kaggle Daily OHLCV**\n```\nDate, Open, High, Low, Close, Volume\n2020-01-01, 7195.3, 7255.0, 7150.0, 7200.0, 12345\n```")
        with c2:
            st.markdown("**Binance 1-Minute Bars**\n```\nOpen time, Open, High, Low, Close, Volume\n1577836800000, 7195.3, 7198.0, 7190.0, 7195.8, 2.1\n```")
        st.stop()

    # KPIs
    price_col = "Close" if "Close" in df.columns else df.columns[0]
    latest = df[price_col].iloc[-1]
    prev   = df[price_col].iloc[-2] if len(df) > 1 else latest
    pct_chg = (latest - prev) / prev * 100
    ath     = df[price_col].max()
    avg_vol = df["Volume"].mean() if "Volume" in df.columns else None

    section_title("Dataset Overview")
    kpi_row([
        {"label": "Total Days",       "value": f"{len(df):,}",          "sub": f"{df.index.min().year}–{df.index.max().year}"},
        {"label": f"Latest {price_col}", "value": f"${latest:,.0f}",   "sub": f"{pct_chg:+.2f}% vs prev"},
        {"label": "All-Time High",    "value": f"${ath:,.0f}",          "sub": df[price_col].idxmax().strftime("%Y-%m-%d")},
        {"label": "Avg Daily Volume", "value": f"{avg_vol:,.0f}" if avg_vol else "N/A", "sub": "BTC / day"},
        {"label": "Date Range",       "value": f"{(df.index.max()-df.index.min()).days:,}d",
         "sub": f"{df.index.min().strftime('%b %Y')} → {df.index.max().strftime('%b %Y')}"},
    ])

    # Price chart with MAs
    section_title("Price History with Moving Averages")
    dma = df.copy()
    # dma["SMA_50"]  = dma[price_col].rolling(50).mean()
    # dma["SMA_200"] = dma[price_col].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dma.index, y=dma[price_col], name="Close",
                              line=dict(color=C_PRIMARY, width=1.6)))
    # fig.add_trace(go.Scatter(x=dma.index, y=dma["SMA_50"], name="SMA 50",
    #                           line=dict(color=C_SOFT_LAVENDER, width=1.2, dash="dot")))
    # fig.add_trace(go.Scatter(x=dma.index, y=dma["SMA_200"], name="SMA 200",
    #                           line=dict(color=C_LIGHT_PASTEL_PURPLE, width=1.2, dash="dash")))
    apply_layout(fig, "BTC/USD — Full Price History", height=420)
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                       yaxis_title="Price (USD)", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🗂  Raw Data (last 20 rows)"):
        pcols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        st.dataframe(
            df[pcols].tail(20).sort_index(ascending=False)
              .style.format({c: "${:,.2f}" for c in pcols if c != "Volume"}),
            use_container_width=True, height=380,
        )

    insight("<strong>Next step:</strong> Navigate to <strong>EDA</strong> for interactive charts, "
            "<strong>Statistics</strong> for stationarity tests, and <strong>Forecasting</strong> to generate predictions.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════

elif PAGE == "EDA":
    page_header("Exploratory Data Analysis",
                "OHLC candlesticks · Volume analysis · Return distribution · Volatility · Drawdown · Monthly heatmap")

    if "df" not in st.session_state:
        no_data_gate()
    df = st.session_state["df"]

    has_ohlc   = all(c in df.columns for c in ["Open","High","Low","Close"])
    has_volume = "Volume" in df.columns
    price_col  = "Close" if "Close" in df.columns else df.columns[0]

    # Lookback selector
    _, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        lookback = st.selectbox("Lookback period",
                                 ["All time","5 Years","3 Years","1 Year","6 Months"], index=0)
    cutoffs   = {"All time": None, "5 Years": 5*365, "3 Years": 3*365, "1 Year": 365, "6 Months": 183}
    days_back = cutoffs[lookback]
    df_view   = df.iloc[-days_back:] if days_back else df

    # ── 2.1  OHLC Candlestick + Volume ────────────────────────────────────────
    section_title("OHLC Candlestick Chart + Volume")

    rows    = 2 if has_volume else 1
    heights = [0.65, 0.35] if has_volume else [1.0]
    fig_cs  = make_subplots(rows=rows, cols=1, row_heights=heights,
                             subplot_titles=("BTC/USD Price","Volume") if has_volume else ("BTC/USD Price",),
                             vertical_spacing=0.15)

    if has_ohlc:
        fig_cs.add_trace(go.Candlestick(
            x=df_view.index, open=df_view["Open"], high=df_view["High"],
            low=df_view["Low"], close=df_view["Close"],
            increasing_line_color=C_GREEN, decreasing_line_color=C_RED,
            increasing_fillcolor=C_GREEN, decreasing_fillcolor=C_RED, name="OHLC",
        ), row=1, col=1)
    else:
        fig_cs.add_trace(go.Scatter(x=df_view.index, y=df_view[price_col],
                                     line=dict(color=C_PRIMARY), name=price_col), row=1, col=1)

    for w, color, dash in [(20, "#E4FF30", "solid"), (50, "#FF7D29", "dot")]:
        ma = df_view[price_col].rolling(w).mean()
        fig_cs.add_trace(go.Scatter(x=df_view.index, y=ma, name=f"SMA {w}",
                                     line=dict(color=color, width=1.3, dash=dash)), row=1, col=1)

    if has_volume:
        vol_colors = [C_GREEN if (has_ohlc and df_view["Close"].iloc[i] >= df_view["Open"].iloc[i])
                      else C_RED if has_ohlc else C_PRIMARY for i in range(len(df_view))]
        fig_cs.add_trace(go.Bar(x=df_view.index, y=df_view["Volume"],
                                 marker_color=vol_colors, opacity=0.55, name="Volume"), row=2, col=1)

    apply_layout(fig_cs, "", height=600)
    fig_cs.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_cs, use_container_width=True)

    # ── 2.2  Returns Distribution + Volatility ────────────────────────────────
    section_title("Log-Return Distribution & Rolling Volatility")

    dr = df_view.copy()
    dr["log_return"] = np.log(dr[price_col] / dr[price_col].shift(1))
    dr["volatility"] = dr["log_return"].rolling(30).std() * np.sqrt(252)
    returns_clean    = dr["log_return"].dropna()

    col1, col2 = st.columns(2)

    with col1:
        from scipy import stats as sp_stats
        mu, sigma = returns_clean.mean(), returns_clean.std()
        x_norm = np.linspace(returns_clean.min(), returns_clean.max(), 200)
        y_norm = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm-mu)/sigma)**2)

        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(x=returns_clean, nbinsx=80,
                                        histnorm="probability density",
                                        marker_color=C_PRIMARY, opacity=0.75, name="Log Returns"))
        fig_ret.add_trace(go.Scatter(x=x_norm, y=y_norm, name="Normal Dist",
                                      line=dict(color="#E4FF30", width=2)))
        apply_layout(fig_ret, "Daily Log-Return Distribution", height=340)
        st.plotly_chart(fig_ret, use_container_width=True)

        kurt = sp_stats.kurtosis(returns_clean)
        skew = sp_stats.skew(returns_clean)
        _, p_norm = sp_stats.normaltest(returns_clean)
        kpi_row([
            {"label": "Mean Return",  "value": f"{mu*100:.2f}%"},
            {"label": "Daily Std",    "value": f"{sigma*100:.2f}%"},
            {"label": "Excess Kurt",  "value": f"{kurt:.2f}", "sub": "fat tails" if kurt > 1 else "near-normal"},
            {"label": "Skewness",     "value": f"{skew:.2f}"},
        ])

    with col2:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=dr.index, y=dr["volatility"], fill="tozeroy",
            fillcolor="rgba(247,147,26,0.12)",
            line=dict(color=C_PRIMARY, width=1.6), name="Ann. Volatility",
        ))
        apply_layout(fig_vol, "30-Day Rolling Annualised Volatility (%)", height=340)
        fig_vol.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_vol, use_container_width=True)

        vol_now = dr["volatility"].dropna().iloc[-1]
        vol_avg = dr["volatility"].dropna().mean()
        vol_max = dr["volatility"].dropna().max()
        kpi_row([
            {"label": "Current Vol", "value": f"{vol_now:.1%}"},
            {"label": "Avg Vol",     "value": f"{vol_avg:.1%}"},
            {"label": "Peak Vol",    "value": f"{vol_max:.1%}"},
        ])

    if p_norm < 0.05:
        insight(f"<strong>Non-Normal Returns:</strong> Normality test p = {p_norm:.4f} — "
                f"excess kurtosis = {kurt:.2f}. Fat tails mean Gaussian-error models "
                "systematically underestimate tail risk.")

    # ── 2.3  Drawdown ─────────────────────────────────────────────────────────
    section_title("Drawdown from All-Time High")

    rolling_max = df_view[price_col].cummax()
    drawdown    = (df_view[price_col] - rolling_max) / rolling_max * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown,
                                 fill="tozeroy", fillcolor="rgba(255,69,96,0.18)",
                                 line=dict(color=C_RED, width=1.3), name="Drawdown %"))
    apply_layout(fig_dd, "Price Drawdown from All-Time High (%)", height=290)
    fig_dd.update_layout(yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    max_dd = drawdown.min()
    insight(f"<strong>Maximum Drawdown:</strong> {max_dd:.1f}% on "
            f"{drawdown.idxmin().strftime('%Y-%m-%d')}. "
            "Deep, extended drawdowns are a hallmark of crypto market cycles.")

    # ── 2.4  Monthly Returns Heatmap ──────────────────────────────────────────
    section_title("Monthly Returns Heatmap")

    monthly = df[price_col].resample("ME").last().pct_change().dropna() * 100
    mdf = pd.DataFrame({
        "Year":   monthly.index.year,
        "Month":  monthly.index.month,
        "Return": monthly.values,
    })
    pivot = mdf.pivot_table(index="Year", columns="Month", values="Return")
    
    # Create a full mapping dictionary
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }

    # Rename only the columns that actually exist in your pivot table
    pivot.columns = [month_names[m] for m in pivot.columns]

    if not pivot.empty:
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0.0, C_RED], [0.5, "#18181F"], [1.0, C_GREEN]],
            zmid=0,
            text=np.round(pivot.values, 1),
            texttemplate="%{text}%",
            colorbar=dict(title=dict(text="Return %", font=dict(color=C_MUTED))),
        ))
        apply_layout(fig_heat, "Monthly Returns Heatmap (%)",
                      height=max(280, len(pivot)*36 + 90))
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── 2.5  Yearly Performance Bar ───────────────────────────────────────────
    section_title("Yearly Price Performance (%)")

    yearly = df[price_col].resample("YE").last().pct_change().dropna() * 100
    fig_yr = go.Figure()
    fig_yr.add_trace(go.Bar(
        x=yearly.index.year.astype(str),
        y=yearly.values,
        marker_color=[C_GREEN if v > 0 else C_RED for v in yearly.values],
        text=[f"{v:+.1f}%" for v in yearly.values],
        textposition="outside",
        name="Yearly Return",
    ))
    apply_layout(fig_yr, "Annual BTC Return (%)", height=500)
    fig_yr.update_layout(yaxis_title="Return (%)", xaxis_title="Year")
    st.plotly_chart(fig_yr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

elif PAGE == "Statistics":
    page_header("Statistical Analysis",
                "Stationarity tests · ACF/PACF · Rolling statistics · Seasonal decomposition · Correlation")

    if "df" not in st.session_state:
        no_data_gate()
    df        = st.session_state["df"]
    price_col = "Close" if "Close" in df.columns else df.columns[0]

    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf

    price_series = df[price_col].dropna()
    log_returns  = np.log(price_series / price_series.shift(1)).dropna()

    # ── 3.1  Stationarity ─────────────────────────────────────────────────────
    section_title("Stationarity Tests — ADF & KPSS")

    @st.cache_data(show_spinner=False)
    def run_stationarity(values):
        v = np.array(values, dtype=float)
        a_s, a_p, _, _, a_cv, _ = adfuller(v, autolag="AIC")
        k_s, k_p, _, k_cv        = kpss(v, regression="c", nlags="auto")
        return a_s, a_p, a_cv, k_s, k_p, k_cv

    def stationarity_block(label, series):
        a_s, a_p, a_cv, k_s, k_p, k_cv = run_stationarity(series.values)
        adf_ok  = a_p < 0.05
        kpss_ok = k_p > 0.05
        verdict = ("✅ CONFIRMED STATIONARY" if (adf_ok and kpss_ok)
                   else "❌ NON-STATIONARY"   if (not adf_ok and not kpss_ok)
                   else "⚠️ INCONCLUSIVE")
        st.markdown(f"**{label}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            kpi_row([
                {"label": "ADF Statistic", "value": f"{a_s:.3f}", "sub": "✅ Stationary" if adf_ok else "❌ Unit root"},
                {"label": "ADF p-value",   "value": f"{a_p:.4f}", "sub": "reject H₀" if adf_ok else "fail to reject H₀"},
            ])
        with col2:
            kpi_row([
                {"label": "KPSS Statistic","value": f"{k_s:.3f}", "sub": "✅ Stationary" if kpss_ok else "❌ Non-stationary"},
                {"label": "KPSS p-value",  "value": f"{k_p:.4f}", "sub": "fail to reject H₀" if kpss_ok else "reject H₀"},
            ])
        with col3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Verdict</div>'
                        f'<div class="kpi-value" style="font-size:0.9rem;">{verdict}</div></div>',
                        unsafe_allow_html=True)

    stationarity_block("Price Level (Close)", price_series)
    st.markdown("<br>", unsafe_allow_html=True)
    stationarity_block("Log Returns (1st Difference)", log_returns)

    insight("<strong>Interpretation:</strong> BTC price levels are non-stationary (random walk). "
            "Log-returns are stationary. ARIMA requires differencing d≥1; Prophet and XGBoost "
            "work on raw prices and handle non-stationarity implicitly.")

    # ── 3.2  ACF / PACF ───────────────────────────────────────────────────────
    section_title("Autocorrelation — ACF & PACF (Log Returns)")

    max_lags = min(40, len(log_returns) // 2 - 1)
    if max_lags > 4:
        acf_res  = acf(log_returns,  nlags=max_lags, alpha=0.05)
        pacf_res = pacf(log_returns, nlags=max_lags, alpha=0.05)
        ci_bound = 1.96 / np.sqrt(len(log_returns))
        lags     = list(range(len(acf_res[0])))

        col1, col2 = st.columns(2)
        for col, arr, title in [
            (col1, acf_res[0],  "ACF — Log Returns"),
            (col2, pacf_res[0], "PACF — Log Returns"),
        ]:
            with col:
                colors = [C_PRIMARY if abs(v) > ci_bound else C_MUTED for v in arr]
                fig_ac = go.Figure()
                fig_ac.add_trace(go.Bar(x=lags, y=arr, marker_color=colors, name="Correlation"))
                fig_ac.add_hline(y=ci_bound,  line_dash="dash", line_color="#E4FF30", line_width=1)
                fig_ac.add_hline(y=-ci_bound, line_dash="dash", line_color="#E4FF30", line_width=1)
                fig_ac.add_hline(y=0, line_color="#2A2A3A", line_width=1)
                apply_layout(fig_ac, title, height=300)
                st.plotly_chart(fig_ac, use_container_width=True)

        insight("ACF significant spike at lag k → MA(k). PACF significant spike at lag p → AR(p). "
                "<strong>Orange bars</strong> exceed the 95% confidence interval (blue dashed lines).")
    else:
        st.warning(f"Only {len(log_returns)} return observations — not enough for ACF/PACF analysis.")

    # ── 3.3  Rolling Statistics ────────────────────────────────────────────────
    section_title("Rolling Statistics — Mean & Standard Deviation")

    _, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        window = st.slider("Window (days)", 7, 180, 30, key="roll_win")

    roll_mean = price_series.rolling(window).mean()
    roll_std  = price_series.rolling(window).std()

    fig_roll = make_subplots(rows=2, cols=1,
                              subplot_titles=(f"{window}-Day Rolling Mean",
                                              f"{window}-Day Rolling Std"),
                              vertical_spacing=0.15)
    fig_roll.add_trace(go.Scatter(x=price_series.index, y=price_series, name="Price",
                                   line=dict(color=C_MUTED, width=1), opacity=0.4), row=1, col=1)
    fig_roll.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean, name="Roll Mean",
                                   line=dict(color=C_PRIMARY, width=2)), row=1, col=1)
    fig_roll.add_trace(go.Scatter(x=roll_std.index, y=roll_std, name="Roll Std",
                                   fill="tozeroy", fillcolor="rgba(247,147,26,0.12)",
                                   line=dict(color=C_SOFT_LAVENDER, width=1.6)), row=2, col=1)
    apply_layout(fig_roll, "", height=600)
    st.plotly_chart(fig_roll, use_container_width=True)

    # ── 3.4  Seasonal Decomposition ───────────────────────────────────────────
    section_title("Seasonal Decomposition (Period = 365 days)")

    from statsmodels.tsa.seasonal import seasonal_decompose
    n_days = len(price_series)

    if n_days >= 365 * 2:
        @st.cache_data(hash_funcs={pd.core.indexes.datetimes.DatetimeIndex: lambda x: tuple(x)})
        def do_decomp(values, idx):
            s = pd.Series(values, index=idx)
            return seasonal_decompose(s, model="multiplicative", period=365)

        decomp = do_decomp(price_series.values, price_series.index)
        components = [
            (price_series,   "Observed", C_PRIMARY),
            (decomp.trend,   "Trend",    "#E4FF30"),
            (decomp.seasonal,"Seasonal", C_GREEN),
            (decomp.resid,   "Residual", C_RED),
        ]
        fig_dec = make_subplots(rows=4, cols=1,
                                 subplot_titles=[c[1] for c in components],
                                 vertical_spacing=0.1)
        for i, (series, label, color) in enumerate(components, 1):
            fig_dec.add_trace(go.Scatter(x=series.index, y=series,
                                          line=dict(color=color, width=1.3), name=label), row=i, col=1)
        apply_layout(fig_dec, "Multiplicative Seasonal Decomposition", height=900)
        st.plotly_chart(fig_dec, use_container_width=True)

        seas_strength = 1 - decomp.resid.var() / (decomp.seasonal + decomp.resid).var()
        insight(f"<strong>Seasonal Strength:</strong> {seas_strength:.3f} "
                f"({'strong' if seas_strength > 0.4 else 'weak'} 365-day cycle). "
                "Values > 0.4 suggest yearly seasonality should be included in the model.")
    else:
        st.warning(f"Only {n_days} days — seasonal decomposition requires ≥ 730 days (2 full years).")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FORECASTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

elif PAGE == "Forecasting":
    page_header("Forecasting Engine",
                "Configure · Train · Backtest · Generate future forecast")

    if "df" not in st.session_state:
        no_data_gate()
    df = st.session_state["df"]

    # Controls in sidebar
    with st.sidebar:
        st.divider()
        st.markdown("### 🔮 Forecast Config")
        price_col = st.selectbox(
            "Price Column",
            [c for c in ["Close","Open","High","Low"] if c in df.columns], index=0,
        )
        model_choice = st.selectbox("Model", [
            "Model 1 — Prophet",
            "Model 2 — Prophet + XGBoost Hybrid",
        ])
        horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
        ci_pct  = st.select_slider("Confidence Interval", [80, 90, 95, 99], value=95)
        run_btn = st.button("🚀  Generate Forecast")

    # Train / test split
    price_series = df[price_col].dropna()
    split_idx    = int(len(price_series) * 0.8)
    train        = price_series.iloc[:split_idx]
    test         = price_series.iloc[split_idx:]

    section_title("Train / Test Split (80/20 Temporal)")
    kpi_row([
        {"label": "Train Set",  "value": f"{len(train):,} days",
         "sub": f"{train.index[0].strftime('%Y-%m-%d')} → {train.index[-1].strftime('%Y-%m-%d')}"},
        {"label": "Test Set",   "value": f"{len(test):,} days",
         "sub": f"{test.index[0].strftime('%Y-%m-%d')} → {test.index[-1].strftime('%Y-%m-%d')}"},
        {"label": "Horizon",    "value": f"{horizon} days", "sub": "future forecast window"},
        {"label": "Confidence Interval",   "value": f"{ci_pct}%",      "sub": "uncertainty band"},
    ])

    fig_split = go.Figure()
    fig_split.add_trace(go.Scatter(x=train.index, y=train, name="Train",
                                    line=dict(color=C_PRIMARY, width=1.5)))
    fig_split.add_trace(go.Scatter(x=test.index, y=test, name="Test",
                                    line=dict(color=C_SOFT_LAVENDER, width=1.5)))
    x_value = test.index[0].timestamp() * 1000
    fig_split.add_vline(x=x_value, line_dash="dot", line_color=C_MUTED,
                         annotation_text=" Split", annotation_font_color=C_MUTED)
    apply_layout(fig_split, "Temporal Train / Test Split", height=270)
    st.plotly_chart(fig_split, use_container_width=True)

    if not run_btn:
        insight("Configure the model in the <strong>sidebar</strong> and click "
                "<strong>Generate Forecast</strong> to begin training.")
        st.stop()

    # Run model
    try:
        if "Model 1" in model_choice:
            with st.spinner("⛏ Training Prophet…  (15–30 s)"):
                result = run_prophet(train, test, horizon, ci_pct)
        else:
            with st.spinner("⛏ Training Prophet + XGBoost Hybrid…  (30–60 s)"):
                result = run_hybrid(df, train, test, price_col, horizon, ci_pct)

        st.session_state["last_result"]    = result
        st.session_state["last_train"]     = train
        st.session_state["last_test"]      = test
        st.session_state["last_price_col"] = price_col

    except Exception as e:
        st.error(f"**Model Error:** {e}")
        st.info("Ensure Prophet and XGBoost are installed. Upload ≥ 200 days for best results.")
        st.stop()

    # Metrics
    section_title("Backtesting Performance — Test Set")
    kpi_row([
        {"label": "MAE",  "value": f"${result['mae']:,.0f}",  "sub": "Mean Absolute Error"},
        {"label": "RMSE", "value": f"${result['rmse']:,.0f}", "sub": "Root Mean Squared Error"},
        {"label": "MAPE", "value": f"{result['mape']:.2f}%",  "sub": "Mean Abs Pct Error"},
    ])

    # Forecast chart
    section_title(f"Forecast — Next {horizon} Days")
    hist_w = train[-365:] if len(train) > 365 else train

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=hist_w.index, y=hist_w, name="Historical",
                                 line=dict(color=C_MUTED, width=1.2), opacity=0.6))
    fig_fc.add_trace(go.Scatter(x=test.index, y=test, name="Actual (Test)",
                                 line=dict(color=C_PRIMARY, width=2)))
    fig_fc.add_trace(go.Scatter(x=result["test_dates"], y=result["test_pred"],
                                 name="Backtest", line=dict(color="#FF7D29", width=1.8, dash="solid")))
    fig_fc.add_trace(go.Scatter(x=result["future_dates"], y=result["future_pred"],
                                 name=f"Forecast +{horizon}d", line=dict(color="#E4FF30", width=2.5)))
    fig_fc.add_trace(go.Scatter(
        x=np.concatenate([result["future_dates"], result["future_dates"][::-1]]),
        y=np.concatenate([result["future_upper"], result["future_lower"][::-1]]),
        fill="toself", fillcolor="rgba(247,147,26,0.12)",
        line=dict(width=0), name=f"{ci_pct}% CI",
    ))
    x_value = test.index[0].timestamp() * 1000
    fig_fc.add_vline(x=x_value, line_dash="dot", line_color=C_MUTED,
                      annotation_text=" Train/Test", annotation_font_color=C_MUTED)
    apply_layout(fig_fc, f"{result['model_name']} — BTC/{price_col} Forecast", height=520)
    fig_fc.update_layout(
        legend=dict(orientation="h", y=-0.13, x=0.5, xanchor="center"),
        yaxis_title="Price (USD)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast table + download
    with st.expander(f"📋 Forecast Table ({horizon} days)", expanded=False):
        fdf = pd.DataFrame({
            "Date":                         pd.to_datetime(result["future_dates"]).strftime("%Y-%m-%d"),
            f"Predicted {price_col} (USD)": np.round(result["future_pred"], 2),
            f"Lower {ci_pct}% CI":          np.round(result["future_lower"], 2),
            f"Upper {ci_pct}% CI":          np.round(result["future_upper"], 2),
        })
        st.dataframe(
            fdf.style.format({
                f"Predicted {price_col} (USD)": "${:,.2f}",
                f"Lower {ci_pct}% CI":          "${:,.2f}",
                f"Upper {ci_pct}% CI":          "${:,.2f}",
            }),
            use_container_width=True, height=360,
        )
        st.download_button("⬇️ Download Forecast CSV",
                            data=fdf.to_csv(index=False).encode(),
                            file_name=f"btc_forecast_{horizon}d.csv",
                            mime="text/csv")

    insight(f"<strong>Model:</strong> {result['model_name']} · "
            f"RMSE = <strong>${result['rmse']:,.0f}</strong> on the test set. "
            "Navigate to <strong>Model Insights</strong> for residual diagnostics.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif PAGE == "Model Insights":
    page_header("Model Insights",
                "Predicted vs actual · Residual diagnostics · Cumulative error · Architecture")

    if "df" not in st.session_state:
        no_data_gate()
    df = st.session_state["df"]

    if "last_result" not in st.session_state:
        insight("No forecast has been run yet. Go to <strong>Forecasting</strong> "
                "and click <em>Generate Forecast</em> first.")
        st.stop()

    result    = st.session_state["last_result"]
    train     = st.session_state["last_train"]
    test      = st.session_state["last_test"]
    price_col = st.session_state["last_price_col"]

    # Align predictions to test index
    test_dates  = pd.to_datetime(result["test_dates"])
    test_pred   = pd.Series(result["test_pred"], index=test_dates)
    test_actual = test.reindex(test_dates).dropna()
    test_pred   = test_pred.reindex(test_actual.index)
    residuals   = test_actual - test_pred

    # ── 5.1  Summary KPIs ────────────────────────────────────────────────────
    section_title("Performance Summary")
    kpi_row([
        {"label": "Model",  "value": result["model_name"].split(":")[-1].strip()[:25], "sub": "selected model"},
        {"label": "MAE",    "value": f"${result['mae']:,.0f}",  "sub": "mean absolute error"},
        {"label": "RMSE",   "value": f"${result['rmse']:,.0f}", "sub": "root mean sq error"},
        {"label": "MAPE",   "value": f"{result['mape']:.2f}%",  "sub": "mean abs pct error"},
        {"label": "Test N", "value": f"{len(test_actual)}",     "sub": "days evaluated"},
    ])

    # ── 5.2  Predicted vs Actual ─────────────────────────────────────────────
    section_title("Predicted vs. Actual (Test Set)")

    col1, col2 = st.columns(2)
    with col1:
        lim_lo = min(test_actual.min(), test_pred.min()) * 0.98
        lim_hi = max(test_actual.max(), test_pred.max()) * 1.02

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(x=test_actual, y=test_pred, mode="markers",
                                     marker=dict(color=C_PRIMARY, size=5, opacity=0.7),
                                     name="Data points"))
        fig_sc.add_trace(go.Scatter(x=[lim_lo, lim_hi], y=[lim_lo, lim_hi],
                                     mode="lines",
                                     line=dict(color=C_GREEN, dash="dash", width=1.5),
                                     name="Perfect fit"))
        apply_layout(fig_sc, "Predicted vs. Actual ($)", height=340)
        fig_sc.update_layout(xaxis_title="Actual Price ($)", yaxis_title="Predicted Price ($)")
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        fig_ot = go.Figure()
        fig_ot.add_trace(go.Scatter(x=test_actual.index, y=test_actual,
                                     name="Actual", line=dict(color=C_PRIMARY, width=2)))
        fig_ot.add_trace(go.Scatter(x=test_pred.index, y=test_pred,
                                     name="Predicted", line=dict(color=C_GREEN, width=2, dash="dash")))
        apply_layout(fig_ot, "Actual vs. Predicted Over Time", height=340)
        fig_ot.update_layout(yaxis_title="Price ($)")
        st.plotly_chart(fig_ot, use_container_width=True)

    # ── 5.3  Residual Diagnostics ─────────────────────────────────────────────
    section_title("Residual Diagnostics")

    from scipy import stats as sp_stats

    col1, col2, col3 = st.columns(3)
    with col1:
        fig_r1 = go.Figure()
        fig_r1.add_trace(go.Scatter(x=residuals.index, y=residuals,
                                     mode="lines+markers",
                                     line=dict(color=C_PRIMARY, width=1),
                                     marker=dict(size=3), name="Residuals"))
        fig_r1.add_hline(y=0, line_dash="dash", line_color=C_MUTED)
        apply_layout(fig_r1, "Residuals Over Time", height=290)
        st.plotly_chart(fig_r1, use_container_width=True)

    with col2:
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Histogram(x=residuals, nbinsx=30,
                                       marker_color=C_PRIMARY, opacity=0.8, name="Residuals"))
        apply_layout(fig_r2, "Residual Distribution", height=290)
        st.plotly_chart(fig_r2, use_container_width=True)

    with col3:
        (qq_x, qq_y), (slope, intercept, _) = sp_stats.probplot(residuals.dropna())
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=qq_x, y=qq_y, mode="markers",
                                     marker=dict(color=C_PRIMARY, size=4), name="Sample"))
        fig_qq.add_trace(go.Scatter(
            x=[min(qq_x), max(qq_x)],
            y=[slope*min(qq_x)+intercept, slope*max(qq_x)+intercept],
            mode="lines", line=dict(color=C_GREEN, dash="dash"), name="Normal",
        ))
        apply_layout(fig_qq, "Q-Q Plot (Normality)", height=290)
        st.plotly_chart(fig_qq, use_container_width=True)

    res_mean  = residuals.mean()
    res_std   = residuals.std()
    _, p_norm = sp_stats.normaltest(residuals.dropna())
    kpi_row([
        {"label": "Residual Mean", "value": f"${res_mean:,.0f}",  "sub": "should be ≈ 0"},
        {"label": "Residual Std",  "value": f"${res_std:,.0f}",   "sub": "spread of errors"},
        {"label": "Normality p",   "value": f"{p_norm:.4f}",      "sub": "p < 0.05 → not normal"},
        {"label": "Max Under",     "value": f"${residuals.min():,.0f}", "sub": "largest under-predict"},
        {"label": "Max Over",      "value": f"${residuals.max():,.0f}", "sub": "largest over-predict"},
    ])

    if abs(res_mean) < res_std * 0.1:
        insight("<strong>Unbiased model:</strong> Mean residual ≈ 0 — no systematic price bias detected.")
    else:
        direction = "over-predicting" if res_mean < 0 else "under-predicting"
        insight(f"<strong>Systematic bias:</strong> Mean residual = ${res_mean:,.0f} — "
                f"the model is <strong>{direction}</strong>. Consider recalibrating changepoint parameters.")

    # ── 5.4  Cumulative Absolute Error ────────────────────────────────────────
    section_title("Cumulative Absolute Error Over Test Period")

    cum_err = residuals.abs().cumsum()
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=cum_err.index, y=cum_err,
                                  fill="tozeroy", fillcolor="rgba(247,147,26,0.10)",
                                  line=dict(color=C_PRIMARY, width=1.8), name="Cumulative |Error|"))
    apply_layout(fig_cum, "Cumulative Absolute Error (USD)", height=270)
    fig_cum.update_layout(yaxis_title="Cumulative |Error| ($)")
    st.plotly_chart(fig_cum, use_container_width=True)

    # ── 5.5  Architecture Explainer ───────────────────────────────────────────
    section_title("Model Architecture")

    if "Hybrid" in result["model_name"]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**Layer 1 — Facebook Prophet**
- Piecewise linear trend with automatic changepoint detection
- Weekly seasonality (Fourier series order 3)
- Yearly seasonality (Fourier series order 10, if ≥1yr data)
- Handles BTC bull/bear regime shifts naturally
- Provides the long-range extrapolation component
            """)
        with col2:
            st.markdown("""
**Layer 2 — XGBoost Residual Corrector**
- Trained only on Prophet's in-sample prediction errors
- Features: lags (1/7/14/30), SMA/EMA/Std/ROC (7/14/30), RSI(14), OHLC spreads
- Captures non-linear momentum and mean-reversion patterns
- CIs derived from bootstrapped residual standard deviation × z-score
- Does not extrapolate — Prophet's job
            """)
        insight("<strong>Key insight:</strong> The hybrid delegates extrapolation to Prophet and correction "
                "to XGBoost. This exploits both models' strengths while avoiding their individual "
                "weaknesses — a classic ensemble stacking architecture.")
    else:
        st.markdown("""
**Facebook Prophet — Decomposable Additive Model**

`y(t) = trend(t) + seasonality(t) + holidays(t) + ε(t)`

| Component | Method | Relevance for BTC |
|---|---|---|
| Trend | Piecewise linear with changepoints | Captures bull/bear cycles |
| Weekly seasonality | Fourier series (K=3) | Weekend trading patterns |
| Yearly seasonality | Fourier series (K=10) | Halving-cycle effects |
| Uncertainty bands | Monte Carlo sampling (300 draws) | Quantifies prediction risk |
| Changepoints | Automatic via regularised regression | Adapts to market regime shifts |
        """)

