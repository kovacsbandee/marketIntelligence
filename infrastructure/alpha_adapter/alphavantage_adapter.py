"""
AlphaLoader module: Fetches, standardizes, and persists financial datasets from Alpha Vantage.

Features:
    - Retrieves daily time series, company fundamentals, financial statements,
      insider transactions, stock splits, and dividends for a given stock symbol.
    - Supports both PostgreSQL (via SQLAlchemy ORM) and local CSV persistence.
    - Uses data_manager.etl_jobs.transform_utils for data standardization and cleanup.
    - Handles API, connection, and data errors robustly.
    - Designed for ETL and data science pipelines.

Usage:
    Instantiate AlphaLoader with a symbol and desired modes, then call desired methods.
"""

import logging
import requests
import pandas as pd
import datetime
import os
import shutil
import json

from configs.config import ALPHA_API_KEY
from infrastructure.databases.company.postgre_manager.postgre_manager import CompanyDataManager
from infrastructure.databases.company.postgre_manager.postgre_objects import (
    DailyTimeSeries, CompanyFundamentalsTable, AnnualIncomeStatement,
    QuarterlyIncomeStatement, AnnualBalanceSheetTable, QuarterlyBalanceSheetTable,
    AnnualCashFlowTable, QuarterlyCashFlowTable, AnnualEarningsTable, QuarterlyEarningsTable,
    InsiderTransactions, StockSplit, DividendsTable
)

from infrastructure.alpha_adapter.transform_utils import (
    preprocess_daily_timeseries,
    preprocess_company_fundamentals,
    preprocess_annual_income_statement,
    preprocess_quarterly_income_statement,
    preprocess_annual_balance_sheet,
    preprocess_quarterly_balance_sheet,
    preprocess_annual_cash_flow,
    preprocess_quarterly_cash_flow,
    preprocess_annual_earnings,
    preprocess_quarterly_earnings,
    preprocess_insider_transactions,
    preprocess_stock_splits,
    preprocess_dividends,
)

from infrastructure.alpha_adapter.column_maps import (
    ANNUAL_BALANCE_SHEET_INT_COLUMNS,
    ANNUAL_BALANCE_SHEET_FLOAT_COLUMNS,
    QUARTERLY_BALANCE_SHEET_INT_COLUMNS,
    QUARTERLY_BALANCE_SHEET_FLOAT_COLUMNS,
    INCOME_STATEMENT_NUMERIC_COLUMNS,
    COMPANY_FUNDAMENTALS_INT_COLUMNS,
    COMPANY_FUNDAMENTALS_FLOAT_COLUMNS,
)


# TODO: ADD  ROW FOR EACH DATA HANDLER TO CHECK FOR DATABASE COMPLIANCE AND DATA QUALITY
# e.g. missing values, and add a logger for these stuff!


def dump_api_response(symbol, table, api_response):
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"debug_data/input/{now_str}_{symbol}_{table}_api.json"
    with open(fname, "w") as f:
        json.dump(api_response, f, indent=2)

def dump_dataframe(symbol, table, df):
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fname = f"debug_data/output/{now_str}_{symbol}_{table}_df.csv"
    df.to_csv(fname, index=False)


class AlphaLoader:

    """
        AlphaLoader: Data Extraction and Storage Utility for Alpha Vantage API

        This module provides the `AlphaLoader` class for fetching a variety of financial data 
        from the Alpha Vantage API, including daily time series, company fundamentals, financial statements, 
        insider transactions, stock splits, and dividends for a specified company symbol.

        The loader supports two main data persistence modes:
        - **Database Mode:** Stores data directly in a PostgreSQL database via SQLAlchemy adapters and ORM classes.
        - **Local Store Mode:** Saves data as CSV files in a local directory.

        Key Features:
        - Robust API querying with error handling for all supported data types.
        - Data standardization and cleanup before storage, using custom utility functions.
        - Modular design to facilitate data quality checks and compliance (future enhancements).
        - Optional local storage for offline or backup use.
        - Easy integration with ETL pipelines.

        Note: Database table definitions and transformation utilities are imported from the `data_manager` package.
    """

    def __init__(self, symbol: str, db_mode: bool = False, local_store_mode: bool = False, verbose_data_logging: bool = False):
        self.symbol = symbol.upper()
        self.db_mode = db_mode
        self.local_store_mode = local_store_mode
        self.verbose_data_logging = verbose_data_logging
        self.base_url = "https://www.alphavantage.co/query?function="
        self.local_store_path = "/home/bandee/projects/marketIntelligence/dev_data/jsons"
        self.logger = logging.getLogger(self.__class__.__name__)

    # PATCH: log_exception now accepts table_name and uses it for debug file naming
    def log_exception(self, e, exc_type="Exception", api_response=None, is_warning=False, table_name=None):
        """
        Logs error or warning, and stores failing DataFrame and API response if verbosity is enabled.
        Writes out all debug data files for each error/warning (does NOT clean the folder).
        """
        msg_type = "Warning" if is_warning else "Exception"
        self.logger.error("An unexpected %s occurred: %s", msg_type, e)
        if self.verbose_data_logging:
            debug_dir = "logs/management/debug_data"
            os.makedirs(debug_dir, exist_ok=True)
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            table = table_name or self._get_table_name()
            # Store DataFrames
            if hasattr(self, "last_df") and self.last_df is not None:
                fname = f"{now_str}_{self.symbol}_{table}.csv"
                self.last_df.to_csv(os.path.join(debug_dir, fname), index=False)
                self.logger.error("Failing DataFrame written to: %s", os.path.join(debug_dir, fname))
            if hasattr(self, "last_df_quarterly") and self.last_df_quarterly is not None:
                fname = f"{now_str}_{self.symbol}_{table}_quarterly.csv"
                self.last_df_quarterly.to_csv(os.path.join(debug_dir, fname), index=False)
                self.logger.error("Failing Quarterly DataFrame written to: %s", os.path.join(debug_dir, fname))
            # Store API response (even if None)
            fname = f"{now_str}_{self.symbol}_{table}_api_response.json"
            with open(os.path.join(debug_dir, fname), "w") as f:
                json.dump(api_response, f, indent=2)
            self.logger.error("API response written to: %s", os.path.join(debug_dir, fname))
        else:
            if hasattr(self, "last_df") and self.last_df is not None:
                self.logger.error("Failing DataFrame for %s: %s", self.symbol, self.last_df)
            if hasattr(self, "last_df_quarterly") and self.last_df_quarterly is not None:
                self.logger.error("Failing Quarterly DataFrame for %s: %s", self.symbol, self.last_df_quarterly)
            self.logger.error("API response for %s: %s", self.symbol, api_response)

    def _get_table_name(self, quarterly=False):
        # Helper to infer table name from context (for file naming)
        import inspect
        stack = inspect.stack()
        for frame in stack:
            if "self" in frame.frame.f_locals:
                method = frame.function
                if method.startswith("get_"):
                    base = method.replace("get_", "")
                    if quarterly:
                        return f"{base}_quarterly"
                    return base
        return "unknown"

    # PATCH: Pass explicit table_name for each data handler
    def get_daily_timeseries(self):
        url = f"{self.base_url}TIME_SERIES_DAILY&outputsize=full&symbol={self.symbol}&apikey={ALPHA_API_KEY}"
        self.logger.info("Fetching data for %s...", self.symbol)
        data = None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            dump_api_response(self.symbol, "daily_time_series", data)  # <-- Dump API response

            if "Time Series (Daily)" not in data:
                self.logger.error("Error in API response for %s: %s", self.symbol, data.get('Note') or data)
                return

            data_df = pd.DataFrame.from_dict(data=data["Time Series (Daily)"], orient='index')
            self.last_df = data_df
            data_df = preprocess_daily_timeseries(data_df, self.symbol)
            dump_dataframe(self.symbol, "daily_time_series", data_df)  # <-- Dump DataFrame

            if data_df is None or data_df.empty:
                self.logger.info(f"Skipped {self.symbol}: No valid daily timeseries data to insert.")
                return

            if self.db_mode:
                adapter = CompanyDataManager()
                adapter.insert_new_data(table=DailyTimeSeries, rows=data_df.to_dict(orient="records"))
                self.logger.info("Candlestick data for %s loaded into the database.", self.symbol)

            if self.local_store_mode:
                output_path = f"{self.local_store_path}/{self.symbol}_daily_time_series.csv"
                data_df.to_csv(output_path, index=False)
                self.logger.info("Data saved locally: %s", output_path)

            if not self.db_mode and not self.local_store_mode:
                self.logger.warning('Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed for %s: %s", self.symbol, e)
        except Exception as e:
            self.log_exception(e, api_response=data, table_name="daily_time_series")

    def get_company_base(self):
        url = f'{self.base_url}OVERVIEW&symbol={self.symbol}&apikey={ALPHA_API_KEY}'
        self.logger.info("Fetching company base data for %s...", self.symbol)
        data = None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            dump_api_response(self.symbol, "company_fundamentals", data)  # <-- Dump API response

            if "Error Message" in data or "Note" in data:
                self.logger.error("Error in API response for %s: %s", self.symbol, data.get('Note') or data)
                return

            data_df = pd.DataFrame([data])
            self.last_df = data_df
            data_df = preprocess_company_fundamentals(data_df, self.symbol)
            dump_dataframe(self.symbol, "company_fundamentals", data_df)  # <-- Dump DataFrame

            if data_df is None or data_df.empty:
                self.logger.info(f"Skipped {self.symbol}: No valid company fundamentals data to insert.")
                return

            if self.db_mode:
                adapter = CompanyDataManager()
                adapter.insert_new_data(table=CompanyFundamentalsTable, rows=data_df.to_dict(orient="records"))
                self.logger.info("Company base data for %s loaded into the database.", self.symbol)

            if self.local_store_mode:
                latest_quarter = data_df.get("latest_quarter", ["unknown"])[0]
                output_path = f"{self.local_store_path}/{self.symbol}_company_fundamentals_lat_quart_{latest_quarter}.csv"
                data_df.to_csv(output_path, index=False)
                self.logger.info("Company base data saved locally: %s", output_path)

            if not self.db_mode and not self.local_store_mode:
                self.logger.warning('Choose a place where you want to store the company base data!')

        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed for %s: %s", self.symbol, e)
        except Exception as e:
            self.log_exception(e, api_response=data, table_name="company_fundamentals")

    def get_financials(self, function: str):
        url = f'{self.base_url}{function}&symbol={self.symbol}&apikey={ALPHA_API_KEY}'
        self.logger.info("Fetching %s data for %s...", function, self.symbol)
        data = None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            dump_api_response(self.symbol, function.lower(), data)  # <-- Dump API response

            if function == 'EARNINGS':
                annual_key = "annualEarnings"
                quarterly_key = "quarterlyEarnings"
                if annual_key not in data or quarterly_key not in data:
                    self.logger.error("Error in API response for %s: %s", self.symbol, data.get('Note') or data)
                    return
                annual_df = pd.DataFrame(data[annual_key])
                quarterly_df = pd.DataFrame(data[quarterly_key])
            else:
                if "annualReports" not in data or "quarterlyReports" not in data:
                    self.logger.error("Error in API response for %s: %s", self.symbol, data.get('Note') or data)
                    return
                annual_df = pd.DataFrame(data["annualReports"])
                quarterly_df = pd.DataFrame(data["quarterlyReports"])

            # Preprocess by table type
            if function == 'INCOME_STATEMENT':
                annual_df = preprocess_annual_income_statement(annual_df, self.symbol)
                quarterly_df = preprocess_quarterly_income_statement(quarterly_df, self.symbol)
                AnnualTable = AnnualIncomeStatement
                QuarterlyTable = QuarterlyIncomeStatement
            elif function == 'BALANCE_SHEET':
                annual_df = preprocess_annual_balance_sheet(annual_df, self.symbol)
                quarterly_df = preprocess_quarterly_balance_sheet(quarterly_df, self.symbol)
                AnnualTable = AnnualBalanceSheetTable
                QuarterlyTable = QuarterlyBalanceSheetTable
            elif function == 'CASH_FLOW':
                annual_df = preprocess_annual_cash_flow(annual_df, self.symbol)
                quarterly_df = preprocess_quarterly_cash_flow(quarterly_df, self.symbol)
                AnnualTable = AnnualCashFlowTable
                QuarterlyTable = QuarterlyCashFlowTable
            elif function == 'EARNINGS':
                annual_df = preprocess_annual_earnings(annual_df, self.symbol)
                quarterly_df = preprocess_quarterly_earnings(quarterly_df, self.symbol)
                AnnualTable = AnnualEarningsTable
                QuarterlyTable = QuarterlyEarningsTable
            else:
                self.logger.error("Unknown function '%s'", function)
                return

            self.last_df = annual_df
            self.last_df_quarterly = quarterly_df

            dump_dataframe(self.symbol, f"{function.lower()}_annual", annual_df)      # <-- Dump annual DataFrame
            dump_dataframe(self.symbol, f"{function.lower()}_quarterly", quarterly_df) # <-- Dump quarterly DataFrame

            # PATCH: skip DB/local insert if None or empty
            if annual_df is None or annual_df.empty:
                self.logger.info(f"Skipped {self.symbol}: No valid annual {function.lower()} data to insert.")
            else:
                if self.db_mode:
                    adapter = CompanyDataManager()
                    adapter.insert_new_data(table=AnnualTable, rows=annual_df.to_dict(orient="records"))
                    self.logger.info("%s annual data for %s loaded into the database.", function, self.symbol)
                if self.local_store_mode:
                    annual_df.to_csv(f"{self.local_store_path}/{self.symbol}_{function.lower()}_annual.csv", index=False)
                    self.logger.info("%s annual data saved locally for %s.", function, self.symbol)

            if quarterly_df is None or quarterly_df.empty:
                self.logger.info(f"Skipped {self.symbol}: No valid quarterly {function.lower()} data to insert.")
            else:
                if self.db_mode:
                    adapter = CompanyDataManager()
                    adapter.insert_new_data(table=QuarterlyTable, rows=quarterly_df.to_dict(orient="records"))
                    self.logger.info("%s quarterly data for %s loaded into the database.", function, self.symbol)
                if self.local_store_mode:
                    quarterly_df.to_csv(f"{self.local_store_path}/{self.symbol}_{function.lower()}_quarterly.csv", index=False)
                    self.logger.info("%s quarterly data saved locally for %s.", function, self.symbol)

            if not self.db_mode and not self.local_store_mode:
                self.logger.warning('Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed for %s: %s", self.symbol, e)
        except Exception as e:
            self.log_exception(e, api_response=data, table_name=function.lower())

    def get_insider_transactions(self):
        url = f"{self.base_url}INSIDER_TRANSACTIONS&symbol={self.symbol}&apikey={ALPHA_API_KEY}"
        self.logger.info("Fetching INSIDER_TRANSACTIONS data for %s...", self.symbol)
        data = None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            dump_api_response(self.symbol, "insider_transactions", data)  # <-- Dump API response

            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            df = pd.DataFrame(data)
            self.last_df = df
            df = preprocess_insider_transactions(df, self.symbol)
            dump_dataframe(self.symbol, "insider_transactions", df)  # <-- Dump DataFrame

            if df is None or df.empty:
                self.logger.info(f"Skipped {self.symbol}: No valid insider transactions data to insert.")
                return

            if self.db_mode:
                adapter = CompanyDataManager()
                adapter.insert_new_data(table=InsiderTransactions, rows=df.to_dict(orient="records"))
                self.logger.info("Insider transaction data for %s loaded into the database.", self.symbol)

            if self.local_store_mode:
                df.to_csv(f"{self.local_store_path}/{self.symbol}_insider_transactions.csv", index=False)
                self.logger.info("Insider transaction data saved locally for %s.", self.symbol)

            if not self.db_mode and not self.local_store_mode:
                self.logger.warning('Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed for %s: %s", self.symbol, e)
        except Exception as e:
            self.log_exception(e, api_response=data, table_name="insider_transactions")

    def get_stock_splits(self):
        url = f"{self.base_url}SPLITS&symbol={self.symbol}&apikey={ALPHA_API_KEY}"
        self.logger.info("Fetching STOCK_SPLITS data for %s...", self.symbol)
        data = None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            dump_api_response(self.symbol, "stock_splits", data)  # <-- Dump API response

            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            df = pd.DataFrame(data)
            self.last_df = df
            df = preprocess_stock_splits(df, self.symbol)
            dump_dataframe(self.symbol, "stock_splits", df)  # <-- Dump DataFrame

            if df is None or df.empty:
                self.logger.info(f"Skipped {self.symbol}: No valid stock splits data to insert.")
                return

            if self.db_mode:
                adapter = CompanyDataManager()
                adapter.insert_new_data(table=StockSplit, rows=df.to_dict(orient="records"))
                self.logger.info("Stock split data for %s loaded into the database.", self.symbol)

            if self.local_store_mode:
                df.to_csv(f"{self.local_store_path}/{self.symbol}_stock_splits.csv", index=False)
                self.logger.info("Stock split data saved locally for %s.", self.symbol)

            if not self.db_mode and not self.local_store_mode:
                self.logger.warning('Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed for %s: %s", self.symbol, e)
        except Exception as e:
            self.log_exception(e, api_response=data, table_name="stock_splits")

    def get_dividends(self):
        url = f"{self.base_url}DIVIDENDS&symbol={self.symbol}&apikey={ALPHA_API_KEY}"
        self.logger.info("Fetching DIVIDENDS data for %s...", self.symbol)
        data = None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            dump_api_response(self.symbol, "dividends", data)  # <-- Dump API response

            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            df = pd.DataFrame(data)
            self.last_df = df
            df = preprocess_dividends(df, self.symbol)
            dump_dataframe(self.symbol, "dividends", df)  # <-- Dump DataFrame

            if df is None or df.empty:
                self.logger.info(f"Skipped {self.symbol}: No valid dividends data to insert.")
                return

            if self.db_mode:
                adapter = CompanyDataManager()
                adapter.insert_new_data(table=DividendsTable, rows=df.to_dict(orient="records"))
                self.logger.info("Dividend data for %s loaded into the database.", self.symbol)

            if self.local_store_mode:
                df.to_csv(f"{self.local_store_path}/{self.symbol}_dividends.csv", index=False)
                self.logger.info("Dividend data saved locally for %s.", self.symbol)

            if not self.db_mode and not self.local_store_mode:
                self.logger.warning('Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed for %s: %s", self.symbol, e)
        except Exception as e:
            self.log_exception(e, api_response=data, table_name="dividends")


# TODO: there is intraday data in the alphavantage

# def get_time_series_intraday(month: str, symbol: str = SYMBOL, interval: str='1min'):
#     """
#         month is in YYYY-MM format
#     """

#     url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&month={month}&outputsize=full&apikey={ALPHA_API_KEY}'
#     r = requests.get(url)
#     data = r.json()
#     data_df = pd.DataFrame.from_dict(data=data["Time Series (1min)"], orient='index')
#     data_df.columns = ['open', 'high', 'low', 'close', 'volume']
#     data_df.index = pd.to_datetime(data_df.index)
#     data_df.sort_index(ascending=True, axis=0, inplace=True)
#     data_df.to_csv(f"/home/bandee/projects/stockAnalyzer/dev_data/{symbol}_intraday_{month}.csv", index=True)
#     print(f"{symbol} for {month} was successfully written out to trash_data!")
