import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('https://github.com/nirmalya21/etf-performance/blob/main/data/raw/etf_prices.csv', parse_dates=['Date'], index_col='Date')
df.head()
     

# Plot price history
df.plot(figsize=(12, 6), title='ETF Price History')
plt.ylabel('Price')
plt.grid(True)
plt.show()
     

# Calculate daily returns
returns = df.pct_change().dropna()
returns.describe()
     

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(returns.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation of Daily Returns')
plt.show()
     

# Cumulative returns
cumulative_returns = (1 + returns).cumprod()
cumulative_returns.plot(figsize=(12, 6), title='Cumulative Returns')
plt.grid(True)
plt.show()
