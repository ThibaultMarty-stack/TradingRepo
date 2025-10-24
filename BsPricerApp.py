import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import blackscholes as bs

import streamlit as st

# Function definitions
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price.

    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to expiration in years
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying stock
    option_type : str : 'call' or 'put'

    Returns:
    float : Option price
    """
    from scipy.stats import norm

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

# ---- Streamlit page configuration ----
st.set_page_config(
    page_title="Black-Scholes Option Pricer",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for parameters
st.sidebar.header("📊 Pricer Parameters")
asset_price = st.sidebar.number_input("Asset Price (S)", value=100.0, step=1.0)
#asset = st.sidebar.selectbox("Underlying", ["AAPL"])
strike_price = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
time_to_maturity = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1)
volatility = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)

# Define parameters for the pricer
S = asset_price
K = strike_price
T = time_to_maturity
sigma = volatility
r = risk_free_rate

st.sidebar.divider()

st.sidebar.header("⚙️ Heatmap Parameters")

min_vol = st.sidebar.number_input("Min Volatility", value=0.1, step=0.01, min_value=0.0, max_value=2.0)
max_vol = st.sidebar.number_input("Max Volatility", value=0.5, step=0.01, min_value=0.0, max_value=2.0)
min_spot = st.sidebar.number_input("Min Spot Price", value=50.0, step=1.0, min_value=0.0, max_value=3*S)
max_spot = st.sidebar.number_input("Max Spot Price", value=150.0, step=1.0, min_value=0.0, max_value=3*S)

volatilities = np.linspace(min_vol, max_vol, 10)
spots = np.linspace(min_spot, max_spot, 10)

call_value = black_scholes(S, K, T, r, sigma, option_type='call')
put_value = black_scholes(S, K, T, r, sigma, option_type='put')

CALLS = np.zeros((len(volatilities), len(spots)))
PUTS = np.zeros((len(volatilities), len(spots)))

for i, vol in enumerate(volatilities):
    for j, spot in enumerate(spots):
        CALLS[i, j] = black_scholes(spot, K, T, r, vol, option_type='call')
        PUTS[i, j] = black_scholes(spot, K, T, r, vol, option_type='put')


st.title("Black-Scholes Option Pricer")

df = pd.DataFrame([{
    "Spot Price": S,
    "Volatility": sigma,
    "Strike": K,
    "Maturity (Years)": T,
    "Risk-Free Rate": r
}])
st.dataframe(
    df.style.format("{:.4f}").set_table_styles([
        {'selector': 'thead th', 'props': [('font-size', '16px'), ('text-align', 'center')]},
        {'selector': 'tbody td', 'props': [('font-size', '15px'), ('text-align', 'center')]}
    ]),use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"""
        <div style="background-color:#90EE90; padding:4px; border-radius:4px; text-align:center;">
            <p style="color:black; font-size:20px; margin:0;">Call Value</p>
            <p style="color:black; font-size:32px; margin:0;">
            <b>${call_value:.2f}<b></p> 
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="background-color:#FFB6C1; padding:4px; border-radius:4px; text-align:center;">
            <p style="color:black; font-size:20px; margin:0;">
            PUT Value
            </p>
            <p style="color:black; font-size:32px; margin:0;">
            <b>${put_value:.2f}<b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.title("Spot-Vol Heatmap - Interactive")

# ---- Create two Streamlit columns ----
col1, col2 = st.columns(2)

# ---- Common layout styling ----
common_layout = dict(
    plot_bgcolor='white',
    paper_bgcolor='black',
    xaxis=dict(
        title="Spot Price",
        tickmode='array',
        tickvals=np.arange(len(spots)),
        ticktext=[f"{s:.2f}" for s in spots],
        side='bottom',
        mirror=True,
        showline=True,
        linecolor='black'
    ),
    yaxis=dict(
        title="Volatility",
        tickmode='array',
        tickvals=np.arange(len(volatilities)),
        ticktext=[f"{v:.2f}" for v in volatilities],
        mirror=True,
        showline=True,
        linecolor='black',
        autorange='reversed'  # keeps top-to-bottom increasing volatility like seaborn
    ),
    font=dict(size=13),
    margin=dict(l=40, r=40, t=50, b=40),
    autosize=True,
)

# ---- Interactive CALL Heatmap ----
fig_call = px.imshow(
    CALLS,
    color_continuous_scale='viridis',
    text_auto='.2f',
)
fig_call.update_layout(
    title="CALL Prices",
    coloraxis_colorbar_title="Value",
    **common_layout
)
fig_call.update_yaxes(scaleanchor="x", scaleratio=1)  # ensure square cells

col1.plotly_chart(fig_call, use_container_width=True)

# ---- Interactive PUT Heatmap ----
fig_put = px.imshow(
    PUTS,
    color_continuous_scale='viridis',
    text_auto='.2f',
)
fig_put.update_layout(
    title="PUT Prices",
    coloraxis_colorbar_title="Value",
    **common_layout
)
fig_put.update_yaxes(scaleanchor="x", scaleratio=1)  # ensure square cells

col2.plotly_chart(fig_put, use_container_width=True)