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
    InsiderTransactionTable, StockSplit, DividendsTable
)

from infrastructure.alpha_adapter.transform_utils import (
    standardize_company_fundamentals_columns,
    standardize_annual_income_statement_columns,
    standardize_quarterly_income_statement_columns,
    standardize_annual_balance_sheet_columns,
    standardize_quarterly_balance_sheet_columns,
    standardize_annual_cash_flow_columns,
    standardize_quarterly_cash_flow_columns,
    standardize_annual_earnings_columns,
    standardize_quarterly_earnings_columns,
    standardize_insider_transaction_columns,
    standardize_stock_split_columns,
    standardize_dividends_columns,
    clean_numeric_types_for_db,
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

    def log_exception(self, e, exc_type="Exception", api_response=None, is_warning=False):
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
            # Store DataFrames
            if hasattr(self, "last_df") and self.last_df is not None:
                fname = f"{now_str}_{self.symbol}_{self._get_table_name()}.csv"
                self.last_df.to_csv(os.path.join(debug_dir, fname), index=False)
                self.logger.error("Failing DataFrame written to: %s", os.path.join(debug_dir, fname))
            if hasattr(self, "last_df_quarterly") and self.last_df_quarterly is not None:
                fname = f"{now_str}_{self.symbol}_{self._get_table_name(quarterly=True)}.csv"
                self.last_df_quarterly.to_csv(os.path.join(debug_dir, fname), index=False)
                self.logger.error("Failing Quarterly DataFrame written to: %s", os.path.join(debug_dir, fname))
            # Store API response (even if None)
            fname = f"{now_str}_{self.symbol}_{self._get_table_name()}_api_response.json"
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

    def get_daily_timeseries(self):
        """
            Fetch daily time series data for the symbol from Alpha Vantage.
            Saves the data to a database or as a CSV file, based on configuration.
            Handles API errors and ensures proper data formatting.
        """
        url = f"{self.base_url}TIME_SERIES_DAILY&outputsize=full&symbol={self.symbol}&apikey={ALPHA_API_KEY}"
        self.logger.info("Fetching data for %s...", self.symbol)
        data = None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            data = r.json()
            # data = json.load(open(f'{self.local_store_path}/{self.symbol}_daily_time_series.json'))
            # Handle API limit or error response
            if "Time Series (Daily)" not in data:
                self.logger.error(
                    "Error in API response for %s: %s", self.symbol, data.get('Note') or data)
                return

            data_df = pd.DataFrame.from_dict(
                data=data["Time Series (Daily)"], orient='index')
            self.last_df = data_df  # PATCH: Store DataFrame for error logging
            data_df.columns = ['open', 'high', 'low', 'close', 'volume']
            data_df.index = pd.to_datetime(data_df.index)
            data_df.sort_index(ascending=True, axis=0, inplace=True)
            data_df.dropna(inplace=True)
            data_df.reset_index(inplace=True, names=["date"])
            data_df["symbol"] = self.symbol

            # Reorder columns
            data_df = data_df[["date", "symbol", "open",
                               "high", "low", "close", "volume"]]

            # Clean numeric types for volume
            data_df["volume"] = pd.to_numeric(data_df["volume"], errors="coerce").round().astype("Int64")
            data_df = data_df.where(pd.notnull(data_df), None)

            if self.db_mode:
                adapter = CompanyDataManager()
                data_df_rows = data_df.to_dict(orient="records")
                adapter.insert_new_data(
                    table=DailyTimeSeries, rows=data_df_rows)
                self.logger.info(
                    "Candlestick data for %s loaded into the database.", self.symbol)

            if self.local_store_mode:
                output_path = f"{self.local_store_path}/{self.symbol}_daily_time_series.csv"
                data_df.to_csv(output_path, index=False)
                self.logger.info("Data saved locally: %s", output_path)

            if not self.db_mode and not self.local_store_mode:
                self.logger.warning('Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed for %s: %s", self.symbol, e)
        except Exception as e:
            self.log_exception(e, api_response=data)

    def get_company_base(self):
        """
        Fetch company base information from Alpha Vantage API and store it.
        This could be updated when a report is arriving for a company...
        """
        url = f'{self.base_url}OVERVIEW&symbol={self.symbol}&apikey={ALPHA_API_KEY}'
        self.logger.info("Fetching company base data for %s...", self.symbol)
        data = None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            data = r.json()

            # Handle error in API response
            if "Error Message" in data or "Note" in data:
                self.logger.error(
                    "Error in API response for %s: %s", self.symbol, data.get('Note') or data)
                return

            # Convert the response into a DataFrame
            data_df = pd.DataFrame([data])
            self.last_df = data_df  # PATCH: Store DataFrame for error logging

            # converting the datatypes:
            data_df.replace("None", pd.NA, inplace=True)
            data_df = data_df.apply(pd.to_numeric, errors='ignore')

            data_df = standardize_company_fundamentals_columns(data_df)

            # Clean numeric types for company fundamentals
            data_df = clean_numeric_types_for_db(
                data_df,
                int_columns=COMPANY_FUNDAMENTALS_INT_COLUMNS,
                float_columns=COMPANY_FUNDAMENTALS_FLOAT_COLUMNS
            )

            # --- Add updater fields ---
            data_df["data_state"] = ""  # Leave empty for now
            data_df["last_update"] = datetime.datetime.now()

            # Save data into the database if db_mode is enabled
            if self.db_mode:
                adapter = CompanyDataManager()
                data_df_rows = data_df.to_dict(orient="records")
                for d in data_df_rows:
                    for k, v in d.items():
                        if pd.isna(v):
                            d[k] = None
                adapter.insert_new_data(
                    table=CompanyFundamentalsTable, rows=data_df_rows)
                self.logger.info(
                    "Company base data for %s loaded into the database.", self.symbol)

            # Save data locally as a CSV if local_store_mode is enabled
            if self.local_store_mode:
                if "latest_quarter" in data_df.columns and not data_df["latest_quarter"].empty:
                    latest_quarter = data_df["latest_quarter"].iloc[0]
                else:
                    latest_quarter = "unknown"
                output_path = f"{self.local_store_path}/{self.symbol}_company_fundamentals_lat_quart_{latest_quarter}.csv"
                data_df.to_csv(output_path, index=False)
                self.logger.info("Company base data saved locally: %s", output_path)

            # If neither db_mode nor local_store_mode is enabled, prompt the user
            if not self.db_mode and not self.local_store_mode:
                self.logger.warning('Choose a place where you want to store the company base data!')

        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed for %s: %s", self.symbol, e)
        except Exception as e:
            self.log_exception(e, api_response=data)

    def get_financials(self, function: str):
        """
        Fetch financial data (INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW, or EARNINGS) for the given symbol.
        """
        url = f'{self.base_url}{function}&symbol={self.symbol}&apikey={ALPHA_API_KEY}'
        self.logger.info("Fetching %s data for %s...", function, self.symbol)
        data = None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            data = r.json()

            # === 1. Build DataFrames ===
            if function == 'EARNINGS':
                # EARNINGS uses different top-level keys!
                annual_key = "annualEarnings"
                quarterly_key = "quarterlyEarnings"
                if annual_key not in data or quarterly_key not in data:
                    self.logger.error(
                        "Error in API response for %s: %s", self.symbol, data.get('Note') or data)
                    return
                annual_df = pd.DataFrame(data[annual_key])
                quarterly_df = pd.DataFrame(data[quarterly_key])
                # Attach symbol if not present
                annual_df["symbol"] = data["symbol"]
                quarterly_df["symbol"] = data["symbol"]
            else:
                if "annualReports" not in data or "quarterlyReports" not in data:
                    self.logger.error(
                        "Error in API response for %s: %s", self.symbol, data.get('Note') or data)
                    return
                annual_df = pd.DataFrame(data["annualReports"])
                annual_df["symbol"] = data["symbol"]

                quarterly_df = pd.DataFrame(data["quarterlyReports"])
                quarterly_df["symbol"] = data["symbol"]

            # === 2. Standardize and type-cast columns ===
            if function == 'INCOME_STATEMENT':
                annual_df = standardize_annual_income_statement_columns(annual_df)
                quarterly_df = standardize_quarterly_income_statement_columns(quarterly_df)
                annual_df = clean_numeric_types_for_db(
                    annual_df,
                    int_columns=[],  # Add integer columns if needed
                    float_columns=INCOME_STATEMENT_NUMERIC_COLUMNS
                )
                quarterly_df = clean_numeric_types_for_db(
                    quarterly_df,
                    int_columns=[],  # Add integer columns if needed
                    float_columns=INCOME_STATEMENT_NUMERIC_COLUMNS
                )
                annual_pk = "fiscal_date_ending"
                quarterly_pk = "fiscal_date_ending"
                AnnualTable = AnnualIncomeStatement
                QuarterlyTable = QuarterlyIncomeStatement
            elif function == 'BALANCE_SHEET':
                annual_df = standardize_annual_balance_sheet_columns(annual_df)
                quarterly_df = standardize_quarterly_balance_sheet_columns(quarterly_df)
                annual_df = clean_numeric_types_for_db(
                    annual_df,
                    int_columns=ANNUAL_BALANCE_SHEET_INT_COLUMNS,
                    float_columns=ANNUAL_BALANCE_SHEET_FLOAT_COLUMNS
                )
                quarterly_df = clean_numeric_types_for_db(
                    quarterly_df,
                    int_columns=QUARTERLY_BALANCE_SHEET_INT_COLUMNS,
                    float_columns=QUARTERLY_BALANCE_SHEET_FLOAT_COLUMNS
                )
                annual_pk = "fiscal_date_ending"
                quarterly_pk = "fiscal_date_ending"
                AnnualTable = AnnualBalanceSheetTable
                QuarterlyTable = QuarterlyBalanceSheetTable
            elif function == 'CASH_FLOW':
                annual_df = standardize_annual_cash_flow_columns(annual_df)
                quarterly_df = standardize_quarterly_cash_flow_columns(quarterly_df)
                annual_df = clean_numeric_types_for_db(
                    annual_df,
                    int_columns=[],  # Add integer columns if needed
                    float_columns=[col for col in annual_df.columns if col not in ['symbol', 'fiscal_date_ending', 'reported_currency']]
                )
                quarterly_df = clean_numeric_types_for_db(
                    quarterly_df,
                    int_columns=[],  # Add integer columns if needed
                    float_columns=[col for col in quarterly_df.columns if col not in ['symbol', 'fiscal_date_ending', 'reported_currency']]
                )
                annual_pk = "fiscal_date_ending"
                quarterly_pk = "fiscal_date_ending"
                AnnualTable = AnnualCashFlowTable
                QuarterlyTable = QuarterlyCashFlowTable
            elif function == 'EARNINGS':
                annual_df = standardize_annual_earnings_columns(annual_df)
                quarterly_df = standardize_quarterly_earnings_columns(quarterly_df)
                annual_df = clean_numeric_types_for_db(
                    annual_df,
                    int_columns=[],  # Add integer columns if needed
                    float_columns=['reported_eps']
                )
                quarterly_df = clean_numeric_types_for_db(
                    quarterly_df,
                    int_columns=[],  # Add integer columns if needed
                    float_columns=['reported_eps', 'estimated_eps', 'surprise', 'surprise_percentage']
                )
                annual_pk = "fiscal_date_ending"
                quarterly_pk = "fiscal_date_ending"
                AnnualTable = AnnualEarningsTable
                QuarterlyTable = QuarterlyEarningsTable
            else:
                self.logger.error("Unknown function '%s'", function)
                return

            # === 3. Final cleanup (drop rows with null PKs, reindex) ===
            for df, pk in [(annual_df, annual_pk), (quarterly_df, quarterly_pk)]:
                if pk in df.columns:
                    df.dropna(subset=[pk], inplace=True)

            # PATCH: Store annual and quarterly DataFrames
            self.last_df = annual_df
            self.last_df_quarterly = quarterly_df

            # === 4. Save to DB if enabled ===
            if self.db_mode:
                adapter = CompanyDataManager()
                adapter.insert_new_data(
                    table=AnnualTable, rows=annual_df.to_dict(orient="records"))
                adapter.insert_new_data(
                    table=QuarterlyTable, rows=quarterly_df.to_dict(orient="records"))
                self.logger.info(
                    "%s data for %s loaded into the database.", function, self.symbol)

            # === 5. Save as CSV locally if enabled ===
            if self.local_store_mode:
                annual_df.to_csv(
                    f"{self.local_store_path}/{self.symbol}_{function.lower()}_annual.csv", index=False)
                quarterly_df.to_csv(
                    f"{self.local_store_path}/{self.symbol}_{function.lower()}_quaterly.csv", index=False)
                self.logger.info("%s data saved locally for %s.", function, self.symbol)

            # === 6. Warn if nothing is enabled ===
            if not self.db_mode and not self.local_store_mode:
                self.logger.warning('Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed for %s: %s", self.symbol, e)
        except Exception as e:
            self.log_exception(e, api_response=data)

    def get_insider_transactions(self):
        """
        Fetch and store insider transactions for the given symbol.
        """
        url = f"{self.base_url}INSIDER_TRANSACTIONS&symbol={self.symbol}&apikey={ALPHA_API_KEY}"
        self.logger.info("Fetching INSIDER_TRANSACTIONS data for %s...", self.symbol)
        data = None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "data" in data:
                data = data["data"]

            df = pd.DataFrame(data)
            self.last_df = df  # PATCH: Store DataFrame for error logging
            df["symbol"] = self.symbol

            df = standardize_insider_transaction_columns(df)
            # Clean numeric types for shares and share_price
            if "shares" in df.columns:
                df["shares"] = pd.to_numeric(df["shares"], errors="coerce").round().astype("Int64")
            if "share_price" in df.columns:
                df["share_price"] = pd.to_numeric(df["share_price"], errors="coerce")
            df = df.where(pd.notnull(df), None)

            if self.db_mode:
                adapter = CompanyDataManager()
                adapter.insert_new_data(
                    table=InsiderTransactionTable, rows=df.to_dict(orient="records"))
                self.logger.info(
                    "Insider transaction data for %s loaded into the database.", self.symbol)

            if self.local_store_mode:
                df.to_csv(
                    f"{self.local_store_path}/{self.symbol}_insider_transactions.csv", index=False)
                self.logger.info(
                    "Insider transaction data saved locally for %s.", self.symbol)

            if not self.db_mode and not self.local_store_mode:
                self.logger.warning(
                    'Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed for %s: %s", self.symbol, e)
        except Exception as e:
            self.log_exception(e, api_response=data)

    def get_stock_splits(self):
        """
        Fetch and store stock split data for the given symbol.
        Always inserts a row for each symbol, even if no data is available.
        """
        url = f"{self.base_url}SPLITS&symbol={self.symbol}&apikey={ALPHA_API_KEY}"
        self.logger.info("Fetching STOCK_SPLITS data for %s...", self.symbol)
        data = None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            self.logger.debug(data)
            # If the response is a dict with the splits in a key
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            # Attach the symbol to each row if missing
            for row in data:
                if "symbol" not in row:
                    row["symbol"] = self.symbol

            # PATCH: If no data, create a dummy row
            if not data or len(data) == 0:
                self.logger.warning("No stock split data found for %s. Inserting dummy row.", self.symbol)
                dummy_row = {
                    "symbol": self.symbol,
                    "effective_date": None,
                    "split_factor": 0.0
                }
                df = pd.DataFrame([dummy_row])
            else:
                df = pd.DataFrame(data)
                required_cols = ["symbol", "effective_date", "split_factor"]
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = None if col == "effective_date" else 0.0 if col == "split_factor" else self.symbol
                df = df[required_cols]

            df = standardize_stock_split_columns(df)
            self.last_df = df

            # Clean numeric types for split_factor
            if "split_factor" in df.columns:
                df["split_factor"] = pd.to_numeric(df["split_factor"], errors="coerce")
            df = df.where(pd.notnull(df), None)
            if "effective_date" in df.columns:
                df["effective_date"] = df["effective_date"].apply(
                    lambda x: "1900-01-01" if pd.isna(x) or x is None else x
                )

            # No dropna, always keep at least one row
            if self.db_mode:
                adapter = CompanyDataManager()
                adapter.insert_new_data(
                    table=StockSplit, rows=df.to_dict(orient="records"))
                self.logger.info(
                    "Stock split data for %s loaded into the database.", self.symbol)

            if self.local_store_mode:
                df.to_csv(
                    f"{self.local_store_path}/{self.symbol}_stock_splits.csv", index=False)
                self.logger.info("Stock split data saved locally for %s.", self.symbol)

            if not self.db_mode and not self.local_store_mode:
                self.logger.warning(
                    'Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed for %s: %s", self.symbol, e)
        except Exception as e:
            self.log_exception(e, api_response=data)

    def get_dividends(self):
        """
        Fetch and store dividend data for the given symbol.
        Always inserts a row for each symbol, even if no data is available.
        """
        url = f"{self.base_url}DIVIDENDS&symbol={self.symbol}&apikey={ALPHA_API_KEY}"
        self.logger.info("Fetching DIVIDENDS data for %s...", self.symbol)
        data = None
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()

            # If the response is a dict with dividends in a key
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            # Attach symbol if missing
            for row in data:
                if "symbol" not in row:
                    row["symbol"] = self.symbol

            # PATCH: If no data, create a dummy row
            if not data or len(data) == 0:
                self.logger.warning("No dividend data found for %s. Inserting dummy row.", self.symbol)
                dummy_row = {
                    "symbol": self.symbol,
                    "ex_dividend_date": None,  # Date type, None is accepted
                    "amount": 0.0,             # Float, 0.0 indicates missing
                    "declaration_date": None,  # Date type, None is accepted
                    "record_date": None,       # Date type, None is accepted
                    "payment_date": None       # Date type, None is accepted
                }
                df = pd.DataFrame([dummy_row])
            else:
                df = pd.DataFrame(data)
                # PATCH: If required column missing, fill with dummy values
                required_cols = ["symbol", "ex_dividend_date", "amount", "declaration_date", "record_date", "payment_date"]
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = None if "date" in col else 0.0 if col == "amount" else self.symbol if col == "symbol" else None
                df = df[required_cols]

            # Always standardize
            df = standardize_dividends_columns(df)
            self.last_df = df

            # Clean numeric types for amount
            if "amount" in df.columns:
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            df = df.where(pd.notnull(df), None)
            for date_col in ["ex_dividend_date", "declaration_date", "record_date", "payment_date"]:
                if date_col in df.columns:
                    df[date_col] = df[date_col].apply(
                        lambda x: "1900-01-01" if pd.isna(x) or x is None else x
                    )

            # No dropna, always keep at least one row
            if self.db_mode:
                adapter = CompanyDataManager()
                adapter.insert_new_data(
                    table=DividendsTable, rows=df.to_dict(orient="records"))
                self.logger.info(
                    "Dividend data for %s loaded into the database.", self.symbol)

            if self.local_store_mode:
                df.to_csv(
                    f"{self.local_store_path}/{self.symbol}_dividends.csv", index=False)
                self.logger.info("Dividend data saved locally for %s.", self.symbol)

            if not self.db_mode and not self.local_store_mode:
                self.logger.warning(
                    'Choose a place where you want to store the data from the API!')

        except requests.exceptions.RequestException as e:
            self.logger.error("Request failed for %s: %s", self.symbol, e)
        except Exception as e:
            self.log_exception(e, api_response=data)


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
