import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

st.set_page_config(page_title="Invesco Midcap – Dip Alert", page_icon="📉")
st.title("📉 Invesco India Midcap – Next-Day Dip-Alert (Minimal)")

# --- Dummy model for now ---
today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
tomorrow = today + timedelta(days=1)

prob = np.random.beta(2, 3)  # 0 … 1
st.metric("Probability of negative NAV tomorrow", f"{prob:.0%}")
st.write(f"Prediction date:", tomorrow.strftime("%d-%b-%Y"))

# --- Tiny line chart ---
chart_df = pd.DataFrame(
    {"Daily NAV %": np.random.randn(90).cumsum()}
)
st.line_chart(chart_df)

st.caption("This is a **skeleton**. Replace the dummy model with real data once the app loads.")
