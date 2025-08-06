import pandas as pd
import numpy as np

# Load and process data
file_path = 'data/raw/2025 KLCI 30 index stock price.xlsx'
# file_path = 'data/raw/2025 FTSE 100 index stock price.xlsx'
# file_path = 'data/raw/2021-2024 KLCI 30 index stock price.xlsx'
# file_path = 'data/raw/2021-2024 FTSE 100 index stock price.xlsx'

# Set risk-free rate 
ANNUAL_RF_RATE = 0.0291  

try:
    # Read Excel file
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=0)
    
    # Convert and sort dates (latest first)
    df = df.rename(columns={'Exchange Date': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index(ascending=False)
    
    # Calculate semi-annual risk-free rate
    # (1 + annual_rate)^(1/2) - 1
    SEMI_ANNUAL_RF_RATE = (1 + ANNUAL_RF_RATE) ** 0.5 - 1
    
    # Initialize results storage
    results = []
    
    # Process each stock
    for stock in df.columns:
        # Remove ALL nulls regardless of position
        series = df[stock].dropna()
        
        if len(series) == 0:
            continue  # Skip if no data remains after cleaning
            
        # Get latest price (first value in descending-sorted data)
        latest_price = series.iloc[0]
        min_buy_in = latest_price * 100
        
        # Prepare data for resampling (requires ascending order)
        series_asc = series.sort_index(ascending=True)
        
        # Resample semi-annually, ignoring periods with no data
        semi_annual = series_asc.resample('6M').last()
        
        # Skip if insufficient data points after resampling
        if len(semi_annual) < 3:  # Need at least 3 points for 2 returns
            continue
            
        # Calculate returns and filter insufficient data
        returns = semi_annual.pct_change().dropna()
        if len(returns) < 2:  # Need at least 2 returns for meaningful stats
            continue
            
        # Calculate performance metrics
        avg_return = returns.mean()
        risk = returns.std()
        
        # Calculate Sharpe ratio
        sharpe_ratio = (avg_return - SEMI_ANNUAL_RF_RATE) / risk if risk != 0 else np.nan
        
        results.append({
            'Stock': stock,
            'Latest Price': latest_price,
            'Min Buy In (RM)': min_buy_in,
            'Semi-Annual Return': avg_return,
            'Semi-Annual Risk': risk,
            'Sharpe Ratio': sharpe_ratio
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Format numerical columns
    results_df['Latest Price'] = results_df['Latest Price'].round(4)
    results_df['Min Buy In (RM)'] = results_df['Min Buy In (RM)'].round(2)
    results_df['Semi-Annual Return'] = results_df['Semi-Annual Return'].round(4)
    results_df['Semi-Annual Risk'] = results_df['Semi-Annual Risk'].round(4)
    results_df['Sharpe Ratio'] = results_df['Sharpe Ratio'].round(4)
    
    # Save to Excel
    output_file = 'data/processed/stock analysis 2025 KLCI 30 index.xlsx'
    # output_file = 'data/processed/stock analysis 2025 FTSE 100 index.xlsx'
    # output_file = 'data/processed/stock analysis 2021-2024 KLCI 30 index.xlsx'
    # output_file = 'data/processed/stock analysis 2021-2024 FTSE 100 index.xlsx'
    results_df.to_excel(output_file, index=False)
    
    print(f"Success! Results saved to '{output_file}'")
    print(f"Analyzed {len(results_df)} stocks with sufficient data")
    print(f"Risk-free rate used: {ANNUAL_RF_RATE*100:.2f}% p.a. → Semi-annual: {SEMI_ANNUAL_RF_RATE*100:.4f}%")
    print("\nSample results:")
    print(results_df.head().to_string(index=False))

except Exception as e:
    print(f"Error: {str(e)}")
    print("Make sure the input file exists and has the correct format")