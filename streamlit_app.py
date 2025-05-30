pip install streamlit pypfopt pandas matplotlib

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

st.set_page_config(page_title="ETF Portfolio Dashboard", layout="wide")

st.title("📈 ETF Portfolio Analysis & Optimization")

# Load data
df = pd.read_csv("data/raw/etf_prices.csv", parse_dates=["Date"], index_col="Date")

st.subheader("ETF Price History")
st.line_chart(df)

# Compute daily returns
returns = df.pct_change().dropna()

# Show stats
st.subheader("Statistical Summary")
st.dataframe(returns.describe().T.style.format("{:.2%}"))

# Portfolio Optimization
st.subheader("📊 Optimized Portfolio Allocation")
mu = mean_historical_return(df)
S = CovarianceShrinkage(df).ledoit_wolf()
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
perf = ef.portfolio_performance(verbose=False)

# Display weights and metrics
st.markdown("**Weights:**")
st.json(cleaned_weights)
st.markdown(f"**Expected Annual Return:** {perf[0]:.2%}")
st.markdown(f"**Annual Volatility:** {perf[1]:.2%}")
st.markdown(f"**Sharpe Ratio:** {perf[2]:.2f}")

# Discrete Allocation
latest_prices = get_latest_prices(df)
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.lp_portfolio()
st.subheader("💼 Discrete Allocation (for $10,000 investment)")
st.write(allocation)
st.markdown(f"**Unallocated Funds:** ${leftover:.2f}")
