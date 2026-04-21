# ₿ BTC Forecast Lab

> A production-quality, multi-page Streamlit web application for **Bitcoin Price Forecasting** — built for an AI Engineering curriculum.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [App Pages](#app-pages)
4. [Models](#models)
   - [Model 1 — Prophet](#model-1--facebook-prophet)
   - [Model 2 — ARIMA / SARIMA](#model-2--arima--sarima)
   - [Model 3 — Prophet + XGBoost Hybrid](#model-3--prophet--xgboost-hybrid)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Supported CSV Formats](#supported-csv-formats)
9. [Tech Stack](#tech-stack)
10. [Design](#design)
11. [Disclaimer](#disclaimer)

---

## Overview

**BTC Forecast Lab** is a full-stack AI engineering application that takes a raw Bitcoin/USD CSV dataset and walks the user through the complete data science pipeline — from exploratory analysis and statistical testing all the way to training a forecasting model and interpreting its results.

The app is structured around two forecasting models drawn from a rigorous backtesting notebook:

- **Model 1** — Facebook Prophet (trend + seasonality decomposition)
- **Model 3** — Prophet + XGBoost Hybrid (two-layer ensemble)

All pages share a single uploaded dataset, held in Streamlit session state, so the user uploads once and navigates freely across all five views.

---

## Features

- **Smart CSV loader** — auto-detects Binance 1-minute bars and standard Kaggle daily OHLCV formats; aggregates to daily, fills calendar gaps, and validates price columns
- **Five interactive pages** with Plotly charts throughout
- **Temporal backtesting** — strict 80/20 train/test split that never leaks future data
- **Confidence intervals** — selectable at 80 / 90 / 95 / 99%
- **Forecast horizon** — configurable from 7 to 90 days
- **Downloadable forecast CSV** — export predictions with CI bands
- **Residual diagnostics** — time-series residuals, histogram, Q-Q plot, bias detection
- **Statistical tests** — ADF & KPSS stationarity, ACF/PACF, seasonal decomposition

---

## App Pages

### ₿ Page 1 — Home
The entry point. Upload a CSV and get an immediate snapshot of the dataset: total days, latest close price, all-time high, average volume, and a full price history chart with SMA 50 and SMA 200 overlays.

### 📊 Page 2 — EDA (Exploratory Data Analysis)
Deep visual exploration of the data:
- **OHLC Candlestick chart** with volume bars and moving averages (SMA 20, SMA 50)
- **Log-return distribution** vs normal distribution overlay
- **30-day rolling annualised volatility** chart
- **Drawdown analysis** — percentage drop from all-time high over time
- **Monthly returns heatmap** — colour-coded by positive/negative months
- **Yearly performance bar chart**

### 🔬 Page 3 — Statistics
Formal statistical analysis:
- **ADF & KPSS stationarity tests** on both price levels and log-returns, with verdict cards
- **ACF & PACF bar charts** with 95% confidence bands for ARIMA order identification
- **Rolling mean and standard deviation** with an adjustable window slider
- **Multiplicative seasonal decomposition** (Observed / Trend / Seasonal / Residual)
- **OHLCV Pearson correlation matrix** heatmap

### 🔮 Page 4 — Forecasting
The core engine:
- Sidebar controls: price column, model, forecast horizon, confidence interval
- Train/test split visualisation before training
- MAE, RMSE, and MAPE metric cards after training
- High-fidelity Plotly forecast chart: historical → backtest → future trend → shaded CI band
- Downloadable forecast table as a `.csv` file

### 💡 Page 5 — Model Insights
Post-hoc model interpretation:
- **Predicted vs. Actual scatter plot** with perfect-fit line
- **Actual vs. Predicted time-series overlay**
- **Residual diagnostics**: over-time plot, distribution histogram, Q-Q normality plot
- **Bias detection** — automatic flag if systematic over/under-prediction is present
- **Cumulative absolute error** chart over the test period
- **Architecture explainer** — table-format breakdown of each model component

---

## Models

### Model 1 — Facebook Prophet

Prophet is a decomposable additive time-series model:

```
y(t) = trend(t) + seasonality(t) + holidays(t) + ε(t)
```

| Component | Configuration |
|---|---|
| Trend | Piecewise linear with automatic changepoint detection |
| Weekly seasonality | Fourier series, order 3 |
| Monthly seasonality | Fourier series, order 10 (if > 60 days of data) |
| Yearly seasonality | Enabled if > 365 days of data |
| Uncertainty | Monte Carlo sampling (300 draws) |
| `changepoint_prior_scale` | `0.15` (flexible — suited to BTC regime shifts) |
| `seasonality_mode` | `multiplicative` (appropriate for growing variance) |

**Best for:** Long-horizon directional forecasts (30–90 days). Handles bull/bear regime changes gracefully through changepoints.

**Limitation:** Assumes piecewise-linear trend. Can underfit sharp, non-linear reversals.

---

### Model 2 — ARIMA / SARIMA

ARIMA (AutoRegressive Integrated Moving Average) is the classical statistical approach to time-series forecasting. It explicitly models three components:

```
ARIMA(p, d, q)

  p  →  AR:  Autoregressive order   — price depends on its own past p values
  d  →  I:   Integrated order       — number of differences needed for stationarity
  q  →  MA:  Moving average order   — model corrects using past q forecast errors
```

For seasonal data, this extends to **SARIMA(p, d, q)(P, D, Q)[m]**, where the uppercase terms capture the same structure at a seasonal period `m` (e.g. `m=7` for weekly patterns).

#### Order Selection — `auto_arima`

Rather than manually grid-searching over `(p, d, q)`, the notebook uses `pmdarima.auto_arima` which automatically selects the optimal order by minimising **AIC (Akaike Information Criterion)**:

```python
import pmdarima as pm

m_arima = pm.auto_arima(
    train_series,
    start_p=0, start_q=0,
    max_p=5,   max_q=5,
    d=None,           # auto-detect differencing order
    seasonal=False,   # set True + m=7 to enable SARIMA
    stepwise=True,    # stepwise search for speed
    information_criterion="aic",
    suppress_warnings=True,
    trace=True,
)
```

Setting `seasonal=False` fits a plain ARIMA; switching to `seasonal=True` with `m=7` enables full SARIMA with weekly seasonality — at the cost of significantly longer fitting time.

#### Walk-Forward Validation

The notebook evaluates ARIMA with a **walk-forward (online) backtesting** strategy rather than a single static split. At each step in the test set the model predicts one step ahead, then updates itself with the true observed value before moving to the next:

```python
for i in range(len(test_series)):
    forecast, conf_int = m_arima.predict(n_periods=1, return_conf_int=True)
    predictions.append(forecast[0])
    m_arima.update([test_series.iloc[i]])   # re-fit on new observation
```

This is the most honest evaluation protocol for ARIMA — it mimics real deployment where you retrain daily on the latest data point.

#### Residual Diagnostics

After fitting, three diagnostic checks verify model adequacy:

| Diagnostic | What it checks | Pass condition |
|---|---|---|
| Residuals over time | No visible structure or drift | Flat, zero-mean scatter |
| Residual histogram + Q-Q plot | Normality of errors | Bell-shaped / points on diagonal |
| Ljung-Box test (lag 10) | No remaining autocorrelation | p-value > 0.05 |

If the Ljung-Box test fails (p < 0.05), autocorrelation remains in the residuals, indicating the model is under-specified and a higher `(p, q)` order should be tried.

#### Configuration Summary

| Parameter | Value | Reason |
|---|---|---|
| `max_p`, `max_q` | 5 | Generous upper bound; AIC penalises over-fitting |
| `d` | auto | Detected via ADF test inside `auto_arima` |
| `seasonal` | `False` | SARIMA adds runtime; plain ARIMA sufficient for daily data |
| `information_criterion` | `"aic"` | Balances fit quality against model complexity |
| `stepwise` | `True` | Reduces search space from O(p×q) to stepwise path |

**Best for:** Short-horizon point forecasts (1–7 days). Linear and interpretable with well-understood uncertainty bounds. Low inference latency — suitable as an ensemble component.

**Limitation:** ARIMA is a **linear** model. It cannot capture the non-linear, fat-tailed, and volatility-clustered dynamics of crypto markets. Forecast uncertainty bands widen rapidly beyond 7–14 days, making it unreliable for long-horizon BTC forecasting. It also requires re-fitting as market regimes shift.

#### Why Model 2 Was Not Included in the App

ARIMA was evaluated in the notebook as the classical baseline but excluded from the Streamlit app for two practical reasons:

1. **Walk-forward refitting is slow** — iterating one step at a time over a 20% test set (potentially hundreds of days) takes minutes per run, making it unsuitable for an interactive UI.
2. **Short-horizon weakness** — the app targets 7–90 day forecast horizons where ARIMA degrades quickly, whereas Prophet and the hybrid remain competitive across the full range.

ARIMA remains the recommended choice for production pipelines that need a fast, interpretable 1–3 day-ahead signal to complement a longer-horizon ensemble.

---

### Model 3 — Prophet + XGBoost Hybrid

A two-layer stacking ensemble that combines the extrapolation strength of Prophet with the pattern-recognition power of XGBoost:

```
Final Prediction = Prophet(trend + seasonality) + XGBoost(residuals)
```

**Layer 1 — Prophet** fits the macro trend and seasonality (same configuration as Model 1, but with `uncertainty_samples=0` for speed).

**Layer 2 — XGBoost** is trained exclusively on Prophet's in-sample residuals, using an engineered feature matrix:

| Feature group | Features |
|---|---|
| Calendar | `day_of_week`, `month`, `quarter`, `day_of_year` |
| Lagged prices | `lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14`, `lag_21`, `lag_30` |
| Rolling statistics | `sma_7/14/30`, `ema_7/14/30`, `std_7/14/30`, `roc_7/14/30` |
| Momentum | `rsi_14` |
| OHLC-derived | `hl_spread`, `oc_spread`, `hl_pct` (if OHLC columns available) |

Confidence intervals are derived from the in-sample residual standard deviation scaled by the appropriate z-value for the selected CI percentage.

**Best for:** Datasets with ≥ 200 days of history. Captures non-linear momentum and mean-reversion patterns that Prophet misses.

**Limitation:** XGBoost cannot extrapolate beyond the training range — trend extrapolation is fully delegated to Prophet.

---

## Project Structure

```
BTC-Price-Forecast/
├── app.py               # Main Streamlit application (all 5 pages)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

The entire application lives in a single `app.py` file, organised into clearly commented sections:

```
Section 1  —  Data Loading       load_btc_csv()
Section 2  —  Feature Eng.       engineer_features()
Section 3  —  Model Runners      run_prophet() / run_hybrid()
Section 4  —  Sidebar Nav        Page routing via st.session_state
Pages      —  Home / EDA / Statistics / Forecasting / Model Insights
```

---

## Installation

**Prerequisites:** Python 3.9 or higher.

### 1. Clone or download the project

```bash
git clone https://github.com/Mostafa710/BTC-Price-Forecast.git
cd btc-forecast-lab
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Installing `prophet` can take several minutes as it compiles Stan models. If you encounter issues, ensure you have a C++ compiler available (`build-essential` on Linux, Xcode CLI tools on macOS).

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Usage

1. **Launch** the app with `streamlit run app.py`
2. **Navigate** to the **Home** page using the sidebar
3. **Upload** a BTC/USD CSV file (see supported formats below)
4. **Explore** the data on the **EDA** and **Statistics** pages
5. **Configure** your model on the **Forecasting** page (sidebar controls), then click **Generate Forecast**
6. **Inspect** results on the **Model Insights** page
7. **Download** the forecast table as a CSV from the Forecasting page

---

## Supported CSV Formats

The app auto-detects two formats:

### Standard Kaggle Daily OHLCV
```
Date,       Open,   High,   Low,    Close,  Volume
2020-01-01, 7195.3, 7255.0, 7150.0, 7200.0, 12345
2020-01-02, 7200.0, 7310.0, 7180.0, 7280.0, 14200
```
Required column: a date column + at least `Close`.

### Binance 1-Minute Bars
```
Open time,      Open,   High,   Low,    Close,  Volume, ...
1577836800000,  7195.3, 7198.0, 7190.0, 7195.8, 2.1
```
The app automatically aggregates minute bars to daily OHLCV using `first/max/min/last/sum`.

**Recommended dataset:** [Bitcoin Historical Data — Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

**Minimum requirement:** 60 days of history. 200+ days recommended for the hybrid model.

---

## Tech Stack

| Library | Version | Role |
|---|---|---|
| `streamlit` | ≥ 1.32 | Web framework and UI |
| `pandas` | ≥ 2.0 | Data loading, cleaning, manipulation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `plotly` | ≥ 5.18 | Interactive charts |
| `prophet` | ≥ 1.1.5 | Model 1 — trend + seasonality |
| `xgboost` | ≥ 2.0 | Model 3 — residual corrector |
| `scikit-learn` | ≥ 1.3 | Metrics (MAE, RMSE), StandardScaler |
| `statsmodels` | ≥ 0.14 | ADF/KPSS tests, seasonal decomposition, ACF/PACF |
| `scipy` | ≥ 1.11 | Return distribution stats, Q-Q plot, normality test |

---

## Design

The UI follows a **NixtNode-inspired deep purple / near-black** aesthetic:

| Token | Hex | Usage |
|---|---|---|
| Accent violet | `#9B6FFF` | Primary highlight, borders, section titles |
| Soft lavender | `#C4A7FF` | Secondary text, chart annotations |
| Near-black | `#0A0A10` | App background |
| Dark card | `#0F0F1A` | Card and sidebar surfaces |
| Bullish green | `#5BFFA0` | Positive candles, up metrics |
| Bearish red | `#FF5B7A` | Negative candles, down metrics |
| Muted grey | `#7878A0` | Labels, captions |

**Font:** [Public Sans](https://fonts.google.com/specimen/Public+Sans) — a clean, neutral grotesque designed for interfaces.

A fixed radial purple glow sits behind all content, giving the app depth and atmosphere without distracting from data.

---

## Disclaimer

> ⚠️ **This application is built for educational purposes only.**
>
> All forecasts are generated from historical price data using statistical and machine learning models. They do **not** account for on-chain metrics, macroeconomic factors, regulatory events, sentiment, or any other real-world drivers of Bitcoin price.
>
> **Do not use this tool to make financial or trading decisions.**
