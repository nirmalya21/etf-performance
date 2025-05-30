# Portfolio Optimization using PyPortfolioOpt

import pandas as pd
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Load ETF price data
df = pd.read_csv('https://github.com/nirmalya21/etf-performance/blob/main/data/raw/etf_prices.csv', parse_dates=['Date'], index_col='Date')

# Calculate expected annual returns and sample covariance matrix
mu = mean_historical_return(df)
S = CovarianceShrinkage(df).ledoit_wolf()

# Optimize for max Sharpe Ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print("Optimized Weights:", cleaned_weights)
ef.portfolio_performance(verbose=True)

# Discrete allocation of $10,000 portfolio
latest_prices = get_latest_prices(df)
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
# Code for portfolio optimization using PyPortfolioOpt
