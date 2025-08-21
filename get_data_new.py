import pandas as pd
import numpy as np

# file_path = 'data/raw/2025 KLCI 30 index stock price.xlsx'
file_path = 'data/raw/2025 FTSE 100 index stock price.xlsx'
ANNUAL_RF_RATE = 0.0291
SEMI_ANNUAL_RF_RATE = (1 + ANNUAL_RF_RATE) ** 0.5 - 1  # same scale as 6m return

# minimum data sanity
MIN_DAYS = 60

# Read the raw price file
df = pd.read_excel(file_path, sheet_name='Sheet1', header=0)
df = df.rename(columns={'Exchange Date': 'Date'})
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').sort_index(ascending=True)  # ascending for slicing

results = []

for stock in df.columns:
    # "series" holds the clean price history for one stock
    # "dropna" is used to remove any missing prices that would break our return computations.
    series = df[stock].dropna()
    if series.empty:
        continue

    # Take the last 6 calendar months for THIS stock
    end = series.index.max()
    start = end - pd.DateOffset(months=6)
    window = series.loc[start:end].dropna()

    # Require enough points to compute a reasonable daily std
    if window.shape[0] < MIN_DAYS:
        continue

     # Semi-annual (6m) return: compound growth from the first to the last price in the window.
    semi_annual_return = window.iloc[-1] / window.iloc[0] - 1

    # Daily returns in the window
    daily = window.pct_change().dropna()
    if daily.empty:
        continue

    # Semi-annual risk (σ_6m):
    # 1) Compute sample stdev of daily simple returns over the last 6 months.
    # 2) Scale by sqrt(N) to get the standard deviation of a 6-month return
    #    (square-root-of-time rule; N = number of trading days in the window).
    risk_sa = daily.std(ddof=1) * np.sqrt(len(daily))

    # Latest price & min buy in
    latest_price = series.iloc[-1]
    min_buy_in = latest_price * 100

    # Semi-annual Sharpe: excess 6m return / 6m risk 
    sharpe_sa = (semi_annual_return - SEMI_ANNUAL_RF_RATE) / risk_sa if risk_sa != 0 else np.nan

    results.append({
        'Stock': stock,
        'Latest Price': round(latest_price, 4),
        'Min Buy In (RM)': round(min_buy_in, 2),
        'Semi-Annual Return': round(semi_annual_return, 4),
        'Semi-Annual Risk': round(risk_sa, 4),
        'Sharpe Ratio': round(sharpe_sa, 4),
    })

results_df = pd.DataFrame(results).sort_values('Stock')
# output_file = 'data/processed/stock analysis 2025 KLCI 30 index.xlsx'
output_file = 'data/processed/stock analysis 2025 FTSE 100 index.xlsx'
results_df.to_excel(output_file, index=False)

print(f"Success! Saved {len(results_df)} stocks → {output_file}")
print(f"Risk-free (annual): {ANNUAL_RF_RATE:.4%}  →  Semi-annual: {SEMI_ANNUAL_RF_RATE:.4%}")
print(results_df.head().to_string(index=False))
