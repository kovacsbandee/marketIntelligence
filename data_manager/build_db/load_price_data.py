import time
import logging

import pandas as pd
import yfinance as yf
from db_adapters import PostgresAdapter

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_and_store_price_data(adapter, symbols, interval="1m", start=None, end=None):
    """
    Fetch price data using yfinance and store it in the database, appending only missing data.

    Args:
        adapter (PostgresAdapter): The database adapter.
        symbols (list): List of stock symbols to fetch data for.
        period (str): The period for fetching data (e.g., "1mo").
        interval (str): The interval for fetching data (e.g., "1d").
        start (str): Start date for the data (e.g., "2023-01-01").
        end (str): End date for the data (e.g., "2023-12-31").
    """
    # Convert start and end to pandas.Timestamp
    start = pd.Timestamp(start) if start else None
    end = pd.Timestamp(end) if end else None

    for symbol in symbols:
        try:
            logger.info(f"Processing {symbol}...")

            # Step 1: Check if the table exists
            table_name = symbol.lower()
            if not adapter.table_exists(table_name):
                logger.info(f"Table for {symbol} does not exist. Creating...")
                table_class = adapter.create_candlestick_table(symbol)
            else:
                table_class = adapter.create_candlestick_table(symbol)

            # Step 2: Get the existing time range in the database
            db_start, db_end = adapter.get_time_range(table_class)
            logger.info(f"Existing time range in database for {symbol}: {db_start} to {db_end}")

            # Step 3: Determine missing time ranges
            missing_ranges = []

            # Case 1: Missing range before the existing data
            # if start < db_start and end < db_end:
            #     missing_ranges.append()
            if start and (not db_start or start < db_start):
                missing_ranges.append((start, db_start - pd.Timedelta(minutes=1) if db_start else end))

            # Case 2: Missing range after the existing data
            if end and (not db_end or end > db_end):
                missing_ranges.append((db_end + pd.Timedelta(minutes=1) if db_end else start, end))

            # If no missing ranges, skip processing
            if not missing_ranges:
                logger.info(f"No missing data to fetch for {symbol}. Skipping...")
                continue

            logger.info(f"Missing ranges for {symbol}: {missing_ranges}")
            # Step 4: Fetch and insert data for each missing range
            for range_start, range_end in missing_ranges:
                logger.info(f"Fetching data for {symbol} from {range_start} to {range_end}...")
                stock_data = yf.download(symbol, start=range_start, end=range_end, interval=interval)
                print(stock_data)
                stock_data.columns = stock_data.columns.get_level_values(0)
                stock_data.drop(["Adj Close"], axis=1, inplace=True)
                if stock_data.empty:
                    logger.warning(f"No data available for {symbol} in range {range_start} to {range_end}. Skipping.")
                    continue
                stock_data.reset_index(inplace=True)
                stock_data.rename(columns={
                    "Datetime": "time",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                }, inplace=True)
                stock_data["time"] = pd.to_datetime(stock_data["time"], errors="coerce")
                stock_data["symbol"] = symbol
                print(stock_data)
                # Prepare data for insertion
                data = stock_data.to_dict(orient="records")

                # Insert only new data into the database
                adapter.insert_new_data(table_class, data)

                logger.info(f"Successfully processed range {range_start} to {range_end} for {symbol}.")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    # Initialize the database adapter
    adapter = PostgresAdapter()

    # List of NASDAQ symbols (replace with actual symbols)
    nasdaq_symbols = ["IBM"]

    # Define start and end dates
    start_date = "2000-12-05"
    end_date = "2024-12-10"

    fetch_and_store_price_data(adapter, nasdaq_symbols, interval="1d", start=start_date, end=end_date)
    # Fetch and store price data for each symbol, downloading only missing data
    #fetch_and_store_price_data(adapter, nasdaq_symbols, period="5d", interval="1m", start=start_date, end=end_date)
