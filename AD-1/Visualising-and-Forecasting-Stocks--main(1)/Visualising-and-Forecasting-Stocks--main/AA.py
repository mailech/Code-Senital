import yfinance as yf

stock = "AAPL"  # Change this to your stock symbol
df = yf.download(stock, period="6mo")

if df.empty:
    print(f"Error: No data found for {stock}")
else:
    print(df.head())  # Check if data is retrieved
