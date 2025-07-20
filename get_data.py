import pandas as pd
import numpy as np

# Load and process data
file_path = 'malaysia stock price 3Y.xlsx'

try:
    # Read Excel file with single header row
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=0)
    
    # Set date column as index and convert to datetime
    df = df.rename(columns={'Exchange Date': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # Initialize results storage
    results = []
    
    # Process each stock
    for stock in df.columns:
        series = df[stock].dropna()
        
        # Resample to semi-annual periods (end of June/December)
        semi_annual = series.resample('6M').last()
        
        if len(semi_annual) < 2:
            continue  # Skip stocks with insufficient data
            
        # Calculate period-to-period returns
        returns = semi_annual.pct_change().dropna()
        
        if len(returns) == 0:
            continue
            
        # Calculate metrics
        avg_return = returns.mean()
        risk = returns.std()
        
        results.append({
            'Stock': stock,
            'Semi-Annual Return': avg_return,
            'Semi-Annual Risk': risk
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to Excel
    output_file = 'stock_returns_risk.xlsx'
    results_df.to_excel(output_file, index=False)
    
    print(f"Success! Results saved to '{output_file}'")
    print(f"\nSample results:")
    print(results_df.head())

except Exception as e:
    print(f"Error: {str(e)}")
    print("Make sure the input file exists and has the correct format")