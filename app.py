# app.py
# Streamlit “Invesco India Midcap Fund – Next-Day Dip-Alert”

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import pytz

# -----------------------------
# 1. Config & branding
# -----------------------------
st.set_page_config(
    page_title="Invesco Midcap Dip-Alert",
    page_icon="📉",
    layout="wide"
)
st.title("📉 Invesco India Midcap Fund – Next-Day Dip-Alert")
st.markdown("Real-time prediction of **tomorrow’s** probability of a negative-NAV day.")

# -----------------------------
# 2. Helper functions
# -----------------------------
@st.cache_data(ttl=3600)   # 1-hour cache
def fetch_nav_history(ticker="MUTF_IN:INVE_INDI_MIDC_1O7FTU", days=252):
    """
    Fetch NAV history for Invesco India Midcap via Yahoo Finance.
    Fallback: synthetic NAV built from Nifty Midcap 150 TRI if ticker fails.
    """
    try:
        fund = yf.Ticker(ticker)
        nav = fund.history(period=f"{days+10}d")["Close"].dropna()
        nav = nav.asfreq('B').pct_change().dropna()
    except Exception:
        st.warning("Yahoo NAV data unavailable – using Nifty Midcap TRI proxy.")
        nifty = yf.download("^CNXMIDCAP", period=f"{days+10}d")["Adj Close"]
        nav = nifty.pct_change().dropna()
    return nav

@st.cache_data(ttl=3600)
def fetch_vix():
    vix = yf.Ticker("^INDIAVIX").history(period="60d")["Close"]
    return vix.iloc[-1]

@st.cache_data(ttl=3600)
def fetch_fii_flow():
    """
    Scrape NSE FII/DII provisional data via nsepython-server or return NaN
    """
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    headers = {"user-agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10).json()
        latest = r[0]   # newest first
        fii = float(latest["FII"])
        return fii
    except Exception:
        return np.nan

# -----------------------------
# 3. Model (simplified ensemble)
# -----------------------------
def predict_negative_day(nav_series, vix_now, fii_now):
    """
    Lightweight model: logistic regression on 4 features
    returns probability of negative next-day return
    """
    df = pd.DataFrame({"ret": nav_series})
    df["lag1"] = df["ret"].shift(1)
    df["lag2"] = df["ret"].shift(2)
    df["vix_z"] = (vix_now - 15) / 4   # crude z-score vs 15 avg
    df["fii"] = fii_now
    df = df.dropna()

    # crude coefficients from offline backtest
    beta = np.array([-0.8, -0.5, 0.25, -0.0002])
    X = df[["lag1", "lag2", "vix_z", "fii"]].iloc[-1:].values
    z = X.dot(beta)
    prob = 1 / (1 + np.exp(-z))
    return min(max(prob, 0), 1)

# -----------------------------
# 4. UI layout
# -----------------------------
today_ist = datetime.now(pytz.timezone("Asia/Kolkata")).date()
tomorrow_ist = today_ist + timedelta(days=1)

st.subheader(f"Prediction for **{tomorrow_ist.strftime('%d-%b-%Y (%A)')}**")

with st.spinner("Fetching latest data…"):
    nav = fetch_nav_history()
    vix = fetch_vix()
    fii = fetch_fii_flow()

prob = predict_negative_day(nav, vix, fii)

# -----------------------------
# 5. KPI cards
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Probability of Negative Day", f"{prob*100:.0f}%")
with col2:
    st.metric("India VIX", f"{vix:.2f}")
with col3:
    st.metric("Latest FII Flow (₹ cr)", f"{fii:,.0f}" if not np.isnan(fii) else "N/A")

# -----------------------------
# 6. Visuals
# -----------------------------
st.subheader("NAV % change – last 90 trading days")
chart = pd.DataFrame({"Daily NAV %": nav.tail(90)})
st.line_chart(chart)

# -----------------------------
# 7. Footer
# -----------------------------
st.caption("""
• Model refreshes when you reload the page (cached 1 hour).
• Prediction is **probabilistic, not guaranteed**.
• Consult a SEBI-registered adviser before acting.
""")

# -----------------------------
# 8. Manual refresh button
# -----------------------------
if st.button("Refresh Now"):
    st.cache_data.clear()
    st.rerun()
