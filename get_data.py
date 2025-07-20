import pandas as pd
import numpy as np

# Load and process data
file_path = 'malaysia stock price 3Y.xlsx'

try:
    # Read Excel file
    df = pd.read_excel(file_path, sheet_name='Sheet1', header=0)
    
    # Set date column as index
    df = df.rename(columns={'Exchange Date': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # Initialize results storage
    results = []
    
    # Process each stock
    for stock in df.columns:
        series = df[stock].dropna()
        semi_annual = series.resample('6M').last()  # Semi-annual resampling
        
        if len(semi_annual) < 2:
            continue  # Skip insufficient data
            
        returns = semi_annual.pct_change().dropna()
        
        if len(returns) == 0:
            continue
            
        avg_return = returns.mean()
        risk = returns.std()
        
        results.append({
            'Stock': stock,
            'Semi-Annual Return': avg_return,
            'Semi-Annual Risk': risk
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # 🔑 KEY MODIFICATION: Round to 4 decimal places
    results_df['Semi-Annual Return'] = results_df['Semi-Annual Return'].round(4)
    results_df['Semi-Annual Risk'] = results_df['Semi-Annual Risk'].round(4)
    
    # Save to Excel
    output_file = 'stock_returns_risk.xlsx'
    results_df.to_excel(output_file, index=False)
    
    print(f"Success! Results saved to '{output_file}'")
    print("\nSample results (4 decimal places):")
    print(results_df.head().to_string(float_format='{:,.4f}'.format))  # 👉 Display 4 decimals

except Exception as e:
    print(f"Error: {str(e)}")
    print("Make sure the input file exists and has the correct format")