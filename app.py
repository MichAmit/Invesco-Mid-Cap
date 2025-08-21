import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title="Invesco Midcap – Dip Alert", page_icon="📉")
st.title("📉 Invesco India Midcap – Next-Day Dip-Alert (Minimal)")

# --- Dummy model for now ---
#today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
#tomorrow = today + timedelta(days=1)

#prob = np.random.beta(2, 3)  # 0 … 1
#st.metric("Probability of negative NAV tomorrow", f"{prob:.0%}")
#st.write(f"Prediction date:", tomorrow.strftime("%d-%b-%Y"))

# --- Real data fetch + model ---
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

@st.cache_data(ttl=3600)   # refresh every hour
def get_real_inputs():
    # 1) NAV proxy: Invesco India Midcap Fund (Growth) on Yahoo
    ticker = "0P0000XVK3.BO"        # Yahoo Finance symbol for this fund
    nav = yf.download(ticker, period="260d")["Close"].pct_change().dropna()

    # 2) Macro drivers
    nifty_mid = yf.download("^CNXMIDCAP", period="260d")["Close"].pct_change().dropna()
    vix = yf.download("^INDIAVIX", period="10d")["Close"].iloc[-1]

    # 3) FII flows (fallback to NaN if NSE endpoint fails)
    try:
        import requests
        fii = float(
            requests.get(
                "https://www.nseindia.com/api/fiidiiTradeReact",
                headers={"user-agent": "Mozilla/5.0"},
                timeout=8,
            ).json()[0]["FII"]
        )
    except Exception:
        fii = np.nan

    return nav, nifty_mid, vix, fii

nav, nifty_mid, vix_now, fii_now = get_real_inputs()

# ---- Simple logistic model ----
df = pd.DataFrame({"nav": nav, "mid": nifty_mid}).dropna()
df["lag1"] = df["nav"].shift(1)
df["vix_z"] = (vix_now - 17) / 5
df["fii"] = 0 if np.isnan(fii_now) else (fii_now / 1000)  # ₹-cr → scale

df = df.dropna()
if len(df) < 5:
    prob = 0.5   # not enough history
else:
    # crude coefficients from offline back-test
    beta = np.array([-0.75, 0.25, -0.0003])
    X = df[["lag1", "vix_z", "fii"]].iloc[-1:].values
    z = X.dot(beta)
    prob = 1 / (1 + np.exp(-z))
    prob = float(np.clip(prob, 0, 1))

# ---- Display ----
today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
tomorrow = today + timedelta(days=1)

st.metric("Probability of negative NAV tomorrow", f"{prob:.0%}")
st.caption(f"Prediction date: {tomorrow.strftime('%d-%b-%Y')}")

st.subheader("NAV % change – last 90 trading days")
chart_df = pd.DataFrame({"Daily NAV %": nav.tail(90)})
st.line_chart(chart_df)

with st.expander("Raw inputs"):
    st.write("Latest VIX:", f"{vix_now:.2f}")
    st.write("Latest FII flow (₹ cr):", f"{fii_now:,.0f}" if not np.isnan(fii_now) else "N/A")

# --- Tiny line chart ---
chart_df = pd.DataFrame(
    {"Daily NAV %": np.random.randn(90).cumsum()}
)
st.line_chart(chart_df)

st.caption("This is a **skeleton**. Replace the dummy model with real data once the app loads.")
