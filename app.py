# app.py - Open-Source BESS Optimization Dashboard (Streamlit)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBRegressor

# --- Page Configuration & Dark Theme Setup ---
st.set_page_config(
    page_title="Open BESS Optimizer | Quantitative Energy",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Dark Theme UI & Readable Legend/Boxes
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .sidebar .sidebar-content { background-color: #16192b; }
    h1, h2, h3 { color: #00ffcc !important; }
    .metric-card { background-color: #1e1e2f; padding: 15px; border-radius: 10px; border: 1px solid #262730; }
    .legendtext { color: #111111 !important; }
    </style>
""", unsafe_allow_html=True)

# --- Core Logic Classes (Integrated) ---
class BESSPricePredictor:
    def __init__(self):
        self.model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)

    def forecast(self, df_history: pd.DataFrame, price_col: str = 'price_eur_mwh') -> pd.DataFrame:
        data = df_history.copy()
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek
        data['rolling_mean_4h'] = data[price_col].shift(1).rolling(window=4).mean()
        data['lag_1h'] = data[price_col].shift(4)
        data.dropna(inplace=True)
        
        features = ['hour', 'dayofweek', 'rolling_mean_4h', 'lag_1h']
        X = data[features]
        y = data[price_col]
        self.model.fit(X, y)
        data['forecasted_price'] = self.model.predict(X)
        return data[['price_eur_mwh', 'forecasted_price']]

class BESSParametricOptimizer:
    def __init__(self, capacity_mwh: float, max_power_mw: float, 
                 min_soc_limit: float, max_soc_limit: float, 
                 roundtrip_efficiency: float, rolling_window: int):
        self.capacity = capacity_mwh
        self.max_power = max_power_mw
        self.min_soc = min_soc_limit
        self.max_soc = max_soc_limit
        self.eta = np.sqrt(roundtrip_efficiency)
        self.rolling_window = rolling_window

    def optimize_dispatch(self, df_prices: pd.DataFrame, price_col: str = 'forecasted_price') -> pd.DataFrame:
        df = df_prices.copy()
        df['p50'] = df[price_col].rolling(window=self.rolling_window, min_periods=4).quantile(0.50)
        df['p90'] = df[price_col].rolling(window=self.rolling_window, min_periods=4).quantile(0.90)
        df['p50'] = df['p50'].bfill().ffill()
        df['p90'] = df['p90'].bfill().ffill()
        
        soc = self.min_soc * self.capacity
        soc_history = []
        actions = []
        revenue_cost = []

        for idx, row in df.iterrows():
            price = row[price_col]
            p50 = row['p50']
            p90 = row['p90']
            
            action = 0
            power_flow = 0.0

            if price <= p50 and soc < (self.max_soc * self.capacity):
                action = 1
                power_flow = self.max_power
                energy_added = power_flow * 0.25 * self.eta
                if soc + energy_added <= self.max_soc * self.capacity:
                    soc += energy_added
                else:
                    power_flow = (self.max_soc * self.capacity - soc) / (0.25 * self.eta)
                    soc = self.max_soc * self.capacity

            elif price >= p90 and soc > (self.min_soc * self.capacity):
                action = -1
                power_flow = -self.max_power
                energy_removed = (abs(power_flow) * 0.25) / self.eta
                if soc - energy_removed >= self.min_soc * self.capacity:
                    soc -= energy_removed
                else:
                    power_flow = -(soc - self.min_soc * self.capacity) * self.eta / 0.25
                    soc = self.min_soc * self.capacity

            actions.append(action)
            revenue_cost.append(-power_flow * price * 0.25)
            soc_history.append(soc / self.capacity)

        df['bess_action'] = actions
        df['soc_level'] = soc_history
        df['cash_flow_eur'] = revenue_cost
        return df

# --- Sidebar Configuration ---
st.sidebar.header("⚡ BESS Asset Parameters")
capacity = st.sidebar.number_input("Capacity (MWh)", value=4.0, step=0.5)
max_power = st.sidebar.number_input("Max Power (MW)", value=2.0, step=0.5)
min_soc = st.sidebar.slider("Min SoC Limit", 0.0, 0.3, 0.1, 0.05)
max_soc = st.sidebar.slider("Max SoC Limit", 0.7, 1.0, 0.9, 0.05)
efficiency = st.sidebar.slider("Roundtrip Efficiency", 0.70, 0.95, 0.85, 0.01)
rolling_window = st.sidebar.slider("Rolling Threshold Window (15-min intervals)", 24, 192, 96, 12)

# --- Main Dashboard Header ---
st.title("Open-Source BESS Quantitative Optimization")
st.markdown("Transparent, lender-grade dispatch modeling for European electricity markets (EPEX / ENTSO-E).")

# --- Data Generation & Pipeline Execution ---
@st.cache_data
def run_pipeline(cap, pwr, min_s, max_s, eff, window):
    time_index = pd.date_range(start="2026-09-01 00:00:00", periods=672, freq="15min")
    np.random.seed(42)
    mock_prices = np.random.uniform(30, 120, size=672)
    mock_prices[600:615] = 585.0  # Spike simulation
    
    df_history = pd.DataFrame({'price_eur_mwh': mock_prices}, index=time_index)
    
    predictor = BESSPricePredictor()
    df_forecasted = predictor.forecast(df_history)
    
    optimizer = BESSParametricOptimizer(
        capacity_mwh=cap, max_power_mw=pwr, 
        min_soc_limit=min_s, max_soc_limit=max_s, 
        roundtrip_efficiency=eff, rolling_window=window
    )
    return optimizer.optimize_dispatch(df_forecasted, price_col='forecasted_price')

results_df = run_pipeline(capacity, max_power, min_soc, max_soc, efficiency, rolling_window)

# --- Key Metrics Overview ---
total_revenue = results_df[results_df['cash_flow_eur'] > 0]['cash_flow_eur'].sum()
total_cost = results_df[results_df['cash_flow_eur'] < 0]['cash_flow_eur'].sum()
net_profit = total_revenue + total_cost

col1, col2, col3 = st.columns(3)
col1.metric("Gross Revenue (Discharge)", f"€{total_revenue:,.2f}")
col2.metric("Total Charging Cost", f"€{abs(total_cost):,.2f}")
col3.metric("Net Financial Yield", f"€{net_profit:,.2f}", delta="Optimal Yield")

st.markdown("---")

# --- Interactive Plotly Charts (Dark Theme with light legend box & dark text) ---
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Price Forecasting & Dynamic Rolling Thresholds", "Battery State of Charge (SoC) & Dispatch Flow")
)

# Price and Thresholds
fig.add_trace(go.Scatter(x=results_df.index, y=results_df['forecasted_price'], name="Forecasted Price", line=dict(color="#00ffcc", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=results_df.index, y=results_df['p90'], name="P90 Threshold (Discharge)", line=dict(color="#ff4444", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=results_df.index, y=results_df['p50'], name="P50 Threshold (Charge)", line=dict(color="#ffbb00", dash="dot")), row=1, col=1)

# Battery SoC
fig.add_trace(go.Scatter(x=results_df.index, y=results_df['soc_level'] * 100, name="SoC Level (%)", line=dict(color="#ff007f", width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=results_df.index, y=results_df.index.to_series().apply(lambda x: max_soc * 100), name="Max Limit", line=dict(color="gray", dash="dot")), row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    height=750,
    margin=dict(l=20, r=20, t=90, b=20),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.08,
        xanchor="right",
        x=1,
        bgcolor="#e0e0e0",       # Light background for clear visibility
        bordercolor="#cccccc",
        borderwidth=1,
        font=dict(color="#111111", size=12)  # Dark, high-contrast text color
    )
)

st.plotly_chart(fig, use_container_width=True)

# --- Data Table Preview ---
with st.expander("🔍 View Raw Pipeline Dispatch Table"):
    st.dataframe(results_df.tail(50), use_container_width=True)
