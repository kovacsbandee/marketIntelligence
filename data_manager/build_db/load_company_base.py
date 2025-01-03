from typing import List
import pandas as pd
import yfinance as yf

from db_objects import Company
from db_adapters import PostgresAdapter  # Import your existing PostgresAdapter class

def fetch_company_fundamentals():
    """
    Fetch company fundamentals using yfinance for a given ticker.
    Args:
        ticker (str): Ticker symbol.
    Returns:
        dict: Dictionary containing company fundamentals.
    """
    try:
        rows = pd.read_csv("/home/bandee/projects/stockAnalyzer/configs/nasdaq_screener.csv")
        rows["ipo_year"] = rows["ipo_year"].astype("Int64")
        rows = rows.to_dict(orient="records")
        return rows
    except Exception as e:
        print(f"Failed to load fundamentals csv from configs: {e}")
        return None

def load_company_base(adapter: PostgresAdapter, fundamentals: List):
    """
    Fetch and load company fundamentals for NASDAQ tickers into the company_base table.
    Args:
        adapter (PostgresAdapter): Instance of PostgresAdapter for database interaction.
        tickers (List[str]): List of NASDAQ tickers.
    """
    
    # Insert the data into the company_base table
    try:
        adapter.insert_data(table=Company, rows=fundamentals)
        print("Company base table loaded successfully.")
    except Exception as e:
        print(f"Error loading company base table: {e}")


if __name__ == "__main__":
    # Initialize the PostgresAdapter
    adapter = PostgresAdapter()
    
    # Load the list of NASDAQ tickers from a CSV file
    fundamentals = fetch_company_fundamentals()
    
    # Load the company base table
    load_company_base(adapter, fundamentals)
