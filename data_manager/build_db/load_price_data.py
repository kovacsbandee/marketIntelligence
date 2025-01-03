import time
import logging

import pandas as pd
import yfinance as yf
from db_adapters import PostgresAdapter

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def fetch_and_store_price_data(adapter, symbols, period="5d", interval="1m"):
    """
    Fetch price data using yfinance and store it in the database.

    Args:
        adapter (PostgresAdapter): The database adapter.
        symbols (list): List of stock symbols to fetch data for.
        period (str): The period for fetching data (e.g., "1mo").
        interval (str): The interval for fetching data (e.g., "1d").
    """
    for symbol in symbols:
        try:
            logger.info(f"Processing {symbol}...")

            # Step 1: Create or retrieve the table for the symbol
            table_class = adapter.create_candlestick_table(symbol)

            # Step 2: Fetch historical data using yfinance
            stock_data = yf.download(symbol, period=period, interval=interval)
            print(stock_data.columns)
            print(stock_data.index)
            print(stock_data.dtypes)
            print(stock_data.info())
            print(stock_data)
            stock_data.columns = stock_data.columns.get_level_values(0)
            stock_data.drop(["Adj Close"], axis=1, inplace=True)
            if stock_data.empty:
                logger.warning(f"No data available for {symbol}. Skipping.")
                continue
            # Debugging: Before resetting index
            logger.debug(f"Data before resetting index:\n{stock_data.head()}")

            # Step 3: Reset index and rename columns
            stock_data.reset_index(inplace=True)
            stock_data.rename(columns={
                "Datetime": "time",  # Ensure the index becomes the `time` column
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)

            # Debugging: After resetting index
            logger.debug(f"Data after resetting index:\n{stock_data.head()}")
            print(stock_data)

            # Handle rows with null values in the 'time' column
            '''
            if stock_data["time"].isnull().any():
                logger.warning(f"Null values detected in the 'time' column for {symbol}. Handling them...")
                stock_data["time"].fillna(method="ffill", inplace=True)  # Forward-fill missing values
                stock_data["time"].fillna(method="bfill", inplace=True)  # Backward-fill missing values
                stock_data.dropna(subset=["time"], inplace=True)
            '''

            # Convert the `time` column to a datetime object if it's not already
            stock_data["time"] = pd.to_datetime(stock_data["time"], errors="coerce")

            # Add the symbol column dynamically
            stock_data["symbol"] = symbol

            # Debugging: Before insertion
            logger.debug(f"Data before insertion:\n{stock_data.head()}")

            print(stock_data)
            # Prepare data for insertion
            data = stock_data.to_dict(orient="records")
            # Step 4: Insert data into the database
            adapter.insert_data(table_class, data)

            logger.info(f"Successfully processed {symbol}.")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

# def fetch_and_store_price_data(adapter, symbols, period="1mo", interval="1d"):
#     """
#     Fetch price data using yfinance and store it in the database.

#     Args:
#         adapter (PostgresAdapter): The database adapter.
#         symbols (list): List of stock symbols to fetch data for.
#         period (str): The period for fetching data (e.g., "1mo").
#         interval (str): The interval for fetching data (e.g., "1d").
#     """
#     for symbol in symbols:
#         try:
#             logger.info(f"Processing {symbol}...")

#             # Step 1: Create or retrieve the table for the symbol
#             table_class = adapter.create_candlestick_table(symbol)

#             # Step 2: Fetch historical data using yfinance
#             stock_data = yf.download(symbol, period=period, interval=interval)
#             print(stock_data["time"])
#             if stock_data.empty:
#                 logger.warning(f"No data available for {symbol}. Skipping.")
#                 continue
#             print(stock_data["time"])
#             # Step 3: Reset index and rename columns
#             stock_data.reset_index(inplace=True)
#             stock_data.rename(columns={
#                 "Date": "time",  # Ensure the index becomes the `time` column
#                 "Open": "open",
#                 "High": "high",
#                 "Low": "low",
#                 "Close": "close",
#                 "Volume": "volume"
#             }, inplace=True)

#             # Handle rows with null values in the 'time' column
#             if stock_data["time"].isnull().any():
#                 logger.warning(f"Null values detected in the 'time' column for {symbol}. Handling them...")
#                 # Fill missing `time` values by interpolation or a placeholder
#                 stock_data["time"].fillna(method="ffill", inplace=True)  # Forward-fill missing values
#                 # Optionally, log how many rows were dropped
#                 logger.info(f"Dropped {stock_data.isnull().sum().sum()} rows due to null 'time' values.")

#             stock_data.dropna(subset=["time"], inplace=True)
#             stock_data["time"] = pd.to_datetime(stock_data["time"], errors="coerce")

#             # Add the symbol column dynamically
#             stock_data["symbol"] = symbol

#             # Prepare data for insertion
#             data = stock_data.to_dict(orient="records")

#             # Step 4: Insert data into the database
#             adapter.insert_data(table_class, data)

#             logger.info(f"Successfully processed {symbol}.")
#         except Exception as e:
#             logger.error(f"Error processing {symbol}: {e}")


if __name__ == "__main__":
    # Initialize the database adapter
    adapter = PostgresAdapter()

    # List of NASDAQ symbols (replace with actual symbols)
    nasdaq_symbols = ["AAPL"] #, "MSFT", "GOOG", "AMZN", "TSLA"]

    # Fetch and store price data for each symbol
    fetch_and_store_price_data(adapter, nasdaq_symbols)

