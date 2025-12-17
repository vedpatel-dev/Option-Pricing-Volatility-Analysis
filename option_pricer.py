import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import yfinance as yf
import time

st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon='📈',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
            
/* 1. THE CONTAINER (Layout Engine) */
.metric-container {
    display: flex;              /* turns on "Flexible Box" mode. To make aligning items easier. */
    justify-content: center;    /* Centers the content horizontally (Left to Right). */
    align-items: center;        /* Centers the content vertically (Top to Bottom). */
    padding: 8px;               /* Adds 8 pixels of internal cushion inside the box to avoid text touching edges */
    width: auto;                /* Don't stretch to fill the screen. Just be as wide as you need. */
    margin: 0 auto;             /* For Box, 0 = Top/Bottom margin, auto = Left/Right margin (automatic centering). */
}
            
/* 2. THE CALL OPTION (GREEN BOX) */
.metric-call {
    background-color: #90ee90;  /* Sets Green background */  
    color: black;               /* Sets text color black*/
    margin-right: 10px;         /* Pushes the next box away */
    border-radius: 10px;        /* Rounds the corners from shrap square */
}

/* 3. THE PUT OPTION (RED BOX) */
.metric-put {
    background-color: #ff0000;      /* Sets Red background */
    color: black;
    border-radius: 10px;    
}

/* 4. NUMBER FORMATTING */
.metric-value {
    font-size: 1.5rem;      /* Makes text 1.5x bigger */
    font-weight: bold;
    margin: 0;              /* Removes extra space */
}
            
/* 5. LABEL FORMATTING */
.metric-label {
    font-size: 1rem;        /* Sets normal text size */
    margin-bottom: 4px;     /* Adds space below the label */
}
</style>
""", unsafe_allow_html=True) # to enable us write HTML/CSS code, Without this, the code printed on your website as text.

class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity    # Time remaining in years (T)
        self.strike = strike                        # The deal price (K)
        self.current_price = current_price          # The market price (S)
        self.volatility = volatility                # How crazy the market is (sigma)
        self.interest_rate = interest_rate          # Risk-free bank rate (r)

    def calculate_prices(self):
        # If time is up (0) or the market isn't moving (vol=0), the formula breaks (divide by zero).
        # So we manually calculate the "Intrinsic Value" (Simple Profit).
        if self.time_to_maturity <= 0 or self.volatility <= 0:
            self.call_price = max(self.current_price - self.strike, 0)      # Call Value = Stock Price - Strike Price (or 0 if negative)
            self.put_price = max(self.strike - self.current_price, 0)
            self.call_delta = self.put_delta = self.call_gamma = self.put_gamma = 0     # Greeks are zero because there is no time left for things to change.
            return self.call_price, self.put_price

        # d1: The 'probability score' for the stock price movement, d1 = {ln(S/K) + (r + {sigma^2}/2).t} / {sigma.sqrt(t)}
        d1 = (log(self.current_price / self.strike) + (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (self.volatility * sqrt(self.time_to_maturity))
        # d2: The probability score adjusted for volatility drag. d2 = d1 - self.volatility * sqrt(self.time_to_maturity)
        d2 = d1 - self.volatility * sqrt(self.time_to_maturity)
        
        # C = S.N(d1) - K.e^{-rt}.N(d2),    norm.cdf(x) -> area under the curve from -infty to x
        self.call_price = self.current_price * norm.cdf(d1) - self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        # P = K.e^{-rt}.N(-d2) - S.N(-d1)
        self.put_price = self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2) - self.current_price * norm.cdf(-d1)

        self.call_delta = norm.cdf(d1)
        self.put_delta = 1 - norm.cdf(d1)
        # Gamma = {N'(d1)}/{S.sigma.sqrt(t)}, norm.pdf(x) -> "Slope" Formula, calculates height of bell curve at point x.
        self.call_gamma = self.put_gamma = norm.pdf(d1) / (self.current_price * self.volatility * sqrt(self.time_to_maturity))

        return self.call_price, self.put_price

with st.sidebar:
    st.title("Black Scholes Model")
    st.markdown(f'''
    <a href="https://www.linkedin.com/in/ved-rajeshkumar-patel-vrp/" target="_blank" style="text-decoration: none; display: inline-flex; align-items: center; gap: 8px;">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20" height="20" />
        <span style="font-size: 16px; font-weight: normal;">Ved Patel</span>
    </a>
    ''', unsafe_allow_html=True)

    ticker_symbol = st.text_input("Enter Stock Ticker Symbol (e.g: MSFT, AAPL, NVDA):", value="NVDA")
    use_real_price = st.checkbox("Use Real-Time Price", value=True)
    
    st.markdown("---") 
    show_advanced = st.checkbox("Show Advanced Plots 📈", value=True, help="Toggle to show/hide Volatility Surfaces and Historical Analysis")

    ticker = None
    try:
        ticker = yf.Ticker(ticker_symbol)
        current_price_live = ticker.history(period="1d")['Close'].iloc[-1]
        if use_real_price:
            st.success(f"Latest price for {ticker_symbol}: ${current_price_live:.2f}")
    except Exception:
        ticker = None
        st.warning("Failed to fetch data for the ticker. Using manual input.")
        current_price_live = 100.0

    current_price = st.number_input("Current Asset Price($S_{t}$)", value=float(current_price_live) if use_real_price else 100.0)
    strike = st.number_input("Strike Price (K)", value=round(current_price * 0.95, 2))
    time_to_maturity = st.number_input("Time to Maturity (t)(Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate (r)", value=0.05)
    call_purchase_price = st.number_input("Purchase Price of CALL Option", value=5.0)
    put_purchase_price = st.number_input("Purchase Price of PUT Option", value=5.0)

    st.markdown('----')
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', 0.01, 1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', 0.01, 1.0, value=volatility*1.5, step=0.01)

    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

def plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, call_price_paid, put_price_paid):
    call_pnl = np.zeros((len(vol_range), len(spot_range)))
    put_pnl = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            temp_model = BlackScholes(bs_model.time_to_maturity, strike, spot, vol, bs_model.interest_rate)
            temp_model.calculate_prices()
            call_pnl[i, j] = temp_model.call_price - call_price_paid
            put_pnl[i, j] = temp_model.put_price - put_price_paid

    cmap = mcolors.LinearSegmentedColormap.from_list("pnl_map", ['red', 'green'], N=100)

    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_pnl, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2),
                annot=True, fmt=".2f", cmap=cmap, ax=ax_call)
    ax_call.set_title("CALL P&L")

    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_pnl, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2),
                annot=True, fmt=".2f", cmap=cmap, ax=ax_put)
    ax_put.set_title("PUT P&L")

    return fig_call, fig_put

def plot_pricing_error_heatmap(time_to_maturity, interest_rate, strike, spot_range, vol_range, market_call_price):
    pricing_errors = np.zeros((len(vol_range), len(spot_range)))
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            model = BlackScholes(time_to_maturity, strike, spot, vol, interest_rate)
            model_call, _ = model.calculate_prices()
            pricing_errors[i, j] = model_call - market_call_price

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = mcolors.LinearSegmentedColormap.from_list("error_map", ["red", "green"], N=100)
    sns.heatmap(pricing_errors, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap=cmap, ax=ax)
    ax.set_title("Pricing Error Heatmap (Model - Market)")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")
    return fig

def plot_greeks_surface(time_to_maturity, interest_rate, strike, spot_range, vol_range):
    delta_surface = np.zeros((len(vol_range), len(spot_range)))
    gamma_surface = np.zeros((len(vol_range), len(spot_range)))
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            model = BlackScholes(time_to_maturity, strike, spot, vol, interest_rate)
            model.calculate_prices()
            delta_surface[i, j] = model.call_delta
            gamma_surface[i, j] = model.call_gamma

    fig_delta, ax_delta = plt.subplots(figsize=(10, 8))
    sns.heatmap(delta_surface, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="Blues", ax=ax_delta)
    ax_delta.set_title("Call Delta Surface")
    ax_delta.set_xlabel("Spot Price")
    ax_delta.set_ylabel("Volatility")

    fig_gamma, ax_gamma = plt.subplots(figsize=(10, 8))
    sns.heatmap(gamma_surface, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".4f", cmap="Oranges", ax=ax_gamma)
    ax_gamma.set_title("Call Gamma Surface")
    ax_gamma.set_xlabel("Spot Price")
    ax_gamma.set_ylabel("Volatility")

    return fig_delta, fig_gamma

def plot_digital_pnl(strike, time_to_maturity, volatility, interest_rate, option_type='Call'):
    # 1. Create a range of Spot Prices (from -20% to +20%)
    spot_range = np.linspace(strike * 0.8, strike * 1.2, 100)
    
    # 2. Calculate "P&L at Expiry" (Intrinsic Value)
    # If Call: Max(Spot - Strike, 0)
    # If Put:  Max(Strike - Spot, 0)
    if option_type == 'Call':
        pnl_expiry = np.maximum(spot_range - strike, 0)
    else:
        pnl_expiry = np.maximum(strike - spot_range, 0)
        
    # 3. Calculate "P&L Today" (Black-Scholes Value)
    pnl_today = []
    for spot in spot_range:
        # Create a temp model for each spot price
        temp_model = BlackScholes(time_to_maturity, strike, spot, volatility, interest_rate)
        call, put = temp_model.calculate_prices()
        
        if option_type == 'Call':
            pnl_today.append(call)
        else:
            pnl_today.append(put)
            
    # 4. Build the Interactive Plotly Chart
    fig = go.Figure()

    # Line 1: The "Hockey Stick" (Expiration)
    fig.add_trace(go.Scatter(
        x=spot_range, 
        y=pnl_expiry, 
        mode='lines', 
        name='Value at Expiry (Intrinsic)',
        line=dict(color='red', width=2, dash='dash') # Dashed red line
    ))

    # Line 2: The "Curve" (Today)
    fig.add_trace(go.Scatter(
        x=spot_range, 
        y=pnl_today, 
        mode='lines', 
        name='Value Today (Black-Scholes)',
        line=dict(color='green', width=3) # Solid green line
    ))

    # Add specific "Time Value" shading
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=pnl_today,
        fill='tonexty', # Fills space between this line and the previous one
        fillcolor='rgba(0, 255, 0, 0.1)', # Light green transparent fill
        name='Time Value (Theta)'
    ))

    # Formatting to look "Pro"
    fig.update_layout(
        title=f'{option_type} Option Payoff Diagram',
        xaxis_title='Spot Price ($)',
        yaxis_title='Option Value ($)',
        template='plotly_dark', # Use Dark Mode to match your new theme
        hovermode='x unified'
    )
    
    return fig

st.title("Black-Scholes Pricing Model")
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price, put_price = bs_model.calculate_prices()

st.table(pd.DataFrame({
    'Current Asset Price': [current_price],
    'Strike Price': [strike],
    'Time to Maturity (Years)': [time_to_maturity],
    'Volatility (σ)': [volatility],
    'Risk-Free Interest Rate': [interest_rate]
}))

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-container metric-call">
        <div>
            <div class="metric-label">CALL Value</div>
            <div class="metric-value">${call_price:.2f}</div>
        </div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-container metric-put">
        <div>
            <div class="metric-label">PUT Value</div>
            <div class="metric-value">${put_price:.2f}</div>
        </div>
    </div>""", unsafe_allow_html=True)


st.title("Interactive P&L Visualizer")
selected_option_type = st.radio("Select Option Type", ["Call", "Put"], horizontal=True)
fig_pnl = plot_digital_pnl(strike, time_to_maturity, volatility, interest_rate, selected_option_type)
st.plotly_chart(fig_pnl, width='stretch')


st.title("Profit & Loss Heatmap")
if st.button("Generate P&L Heatmap"):
    fig_call, fig_put = plot_pnl_heatmap(bs_model, spot_range, vol_range, strike, call_purchase_price, put_purchase_price)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_call)
    with col2:
        st.pyplot(fig_put)

if ticker_symbol:
    try:
        time.sleep(2)
        if ticker is None:
            raise RuntimeError("Ticker object not available")
        option_chain = ticker.option_chain()
        st.subheader("Sample Option Chain - Calls")
        # 1. DATA CLEANING (Crucial for Real World Data)
        calls = option_chain.calls
        # Filter for strikes within 50% of spot price to focus on relevant data
        calls = calls[(calls['strike'] > current_price_live * 0.5) & (calls['strike'] < current_price_live * 1.5)]
        calls = calls[calls['impliedVolatility'] > 0] # Remove invalid data
        
        # 2. PLOTTING THE VOLATILITY SMILE
        fig_iv = go.Figure()
        fig_iv.add_trace(go.Scatter(
            x=calls['strike'], y=calls['impliedVolatility'],
            mode='lines+markers', name='Implied Volatility',
            line=dict(color='cyan', width=2), marker=dict(size=4)
        ))
        fig_iv.add_vline(x=current_price_live, line_dash="dash", line_color="white", annotation_text="Spot")
        fig_iv.update_layout(title=f'Implied Volatility Skew for {ticker_symbol}', xaxis_title='Strike', yaxis_title='Implied Vol (σ)', template='plotly_dark')
        st.plotly_chart(fig_iv, width='stretch')
        # Pricing error comparison
        calls = option_chain.calls.copy()
        calls['strike_diff'] = abs(calls['strike'] - strike)
        closest_call = calls.sort_values(by='strike_diff').iloc[0]
        market_price = closest_call['lastPrice']
        market_iv = closest_call['impliedVolatility']
        market_strike = closest_call['strike']

        model_market_bs = BlackScholes(time_to_maturity, market_strike, current_price, market_iv, interest_rate)
        model_market_call, _ = model_market_bs.calculate_prices()
        pricing_error = model_market_call - market_price

        st.subheader("Pricing Error vs Market")
        col1, col2, col3 = st.columns(3)
        col1.metric("Market Strike", f"${market_strike:.2f}")
        col2.metric("Market Call Price", f"${market_price:.2f}")
        col3.metric("Model Call Price", f"${model_market_call:.2f}")
        st.markdown(f"**Pricing Error**: ${pricing_error:.2f} (Model - Market)")

        error_pct = 100 * pricing_error / market_price
        st.markdown(f"**Pricing Error %**: ({error_pct:+.1f}%)")
    
        if 'market_price' in locals():
            st.subheader("Pricing Error Heatmap")
            fig_error = plot_pricing_error_heatmap(time_to_maturity, interest_rate, market_strike, spot_range, vol_range, market_price)
            st.pyplot(fig_error)

            if show_advanced:
                st.subheader("Delta / Gamma Surfaces")
                fig_delta, fig_gamma = plot_greeks_surface(time_to_maturity, interest_rate, market_strike, spot_range, vol_range)
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig_delta)
                with col2:
                    st.pyplot(fig_gamma)
            
            # --- NEW SECTION: HISTORICAL VOLATILITY ---
        if show_advanced:
            st.title("Realized Volatility Analysis")
        
            history = ticker.history(period="1y")
            if not history.empty:
                # Calculate Log Returns and Rolling Standard Deviation (Annualized)
                history['Log Returns'] = np.log(history['Close'] / history['Close'].shift(1))
                history['Realized Volatility'] = history['Log Returns'].rolling(window=30).std() * np.sqrt(252)
            
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=history.index, y=history['Realized Volatility'], name='30-Day Realized Vol', line=dict(color='orange')))
                fig_vol.update_layout(title=f'Historical Volatility (30-Day Rolling)', xaxis_title='Date', yaxis_title='Volatility', template='plotly_dark')
                st.plotly_chart(fig_vol, width='stretch')
    except Exception as e:
        st.error(f"Option chain fetch failed: {e}")
