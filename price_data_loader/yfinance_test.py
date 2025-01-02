



# Example: Get detailed info for a single company
ticker = yf.Ticker("AAPL")
info = ticker.info
print(info)

# Example info fields for base table
base_table = {
    "ticker": info["symbol"],
    "name": info["shortName"],
    "sector": info.get("sector", "N/A"),
    "industry": info.get("industry", "N/A"),
    "exchange": info["exchange"],
    "market_cap": info.get("marketCap", "N/A"),
    "ipo_date": info.get("ipoDate", "N/A")
}
print(base_table)
