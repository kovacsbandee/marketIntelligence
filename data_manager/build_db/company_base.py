import pandas as pd
import yfinance as yf

from data_manager.build_db.db_adapters import PostgresAdapter  # Import your existing PostgresAdapter class

def fetch_company_fundamentals(ticker):
    """
    Fetch company fundamentals using yfinance for a given ticker.
    Args:
        ticker (str): Ticker symbol.
    Returns:
        dict: Dictionary containing company fundamentals.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant fields for the company base table
        return {
            "ticker": info.get("symbol", ticker),
            "name": info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "exchange": info.get("exchange", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "ipo_date": info.get("ipoDate", None)
        }
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return None

def load_company_base(adapter: PostgresAdapter, tickers: List[str]):
    """
    Fetch and load company fundamentals for NASDAQ tickers into the company_base table.
    Args:
        adapter (PostgresAdapter): Instance of PostgresAdapter for database interaction.
        tickers (List[str]): List of NASDAQ tickers.
    """
    company_data = []
    
    # Fetch data for each ticker
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data = fetch_company_fundamentals(ticker)
        if data:
            company_data.append(data)
    
    # Insert the data into the company_base table
    try:
        adapter.insert_data(table_name="company_base", rows=company_data)
        print("Company base table loaded successfully.")
    except Exception as e:
        print(f"Error loading company base table: {e}")

if __name__ == "__main__":
    # Initialize the PostgresAdapter
    adapter = PostgresAdapter()
    
    # Load the list of NASDAQ tickers from a CSV file
    tickers = pd.read_csv("/home/bandee/projects/stockAnalyzer/configs/nasdaq_tickers.csv")["ticker"].tolist()
    
    # Load the company base table
    load_company_base(adapter, tickers)
