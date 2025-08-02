"""
Standardization utilities for Alpha Vantage data frames.

This module provides a set of functions to:
    - Rename, clean, and type-cast columns of data frames returned by Alpha Vantage API endpoints,
      including company fundamentals, income statements, balance sheets, cash flows, earnings,
      insider transactions, stock splits, and dividends.
    - Ensure column names and types conform to a standardized schema compatible with downstream
      database ingestion.
    - Handle missing values, inconsistent types, and duplicate records in a consistent way.
    - Designed to be reusable in ETL and data engineering workflows.

Note:
    If a function is defined multiple times in this script, only the **last occurrence** will be
    available when the module is imported elsewhere.
"""


import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# transform_utils.py

from .column_maps import (
    COMPANY_FUNDAMENTALS_MAP, 
    ANNUAL_INCOME_STATEMENT_MAP, INCOME_STATEMENT_NUMERIC_COLUMNS, QUARTERLY_INCOME_STATEMENT_MAP, 
    ANNUAL_BALANCE_SHEET_MAP, ANNUAL_BALANCE_SHEET_FLOAT_COLUMNS, ANNUAL_BALANCE_SHEET_INT_COLUMNS,
    QUARTERLY_BALANCE_SHEET_MAP, QUARTERLY_BALANCE_SHEET_FLOAT_COLUMNS, QUARTERLY_BALANCE_SHEET_INT_COLUMNS,
    ANNUAL_CASH_FLOW_MAP, QUARTERLY_CASH_FLOW_MAP,
    ANNUAL_EARNINGS_MAP,
    QUARTERLY_EARNINGS_MAP,
    INSIDER_TRANSACTION_MAP,
    STOCK_SPLIT_MAP,
    DIVIDENDS_MAP,
    COMPANY_FUNDAMENTALS_INT_COLUMNS,
    COMPANY_FUNDAMENTALS_FLOAT_COLUMNS
)


def clean_numeric_types_for_db(df, int_columns=None, float_columns=None):
    """
    Cleans a DataFrame before DB insert:
    - Rounds and casts integer columns to Int64, missing values become None.
    - Casts float columns to float64, missing values become None.
    - Converts pd.NA, np.nan, NaT to None for all columns.

    Args:
        df (pd.DataFrame): DataFrame to clean.
        int_columns (list): List of columns expected to be integer.
        float_columns (list): List of columns expected to be float.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for DB insert.
    """
    df = df.copy()
    if int_columns:
        for col in int_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')
    if float_columns:
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    # Replace all pd.NA, np.nan, NaT with None for DB compatibility
    df = df.where(pd.notnull(df), None)
    return df


def standardize_company_fundamentals_columns(df):
    """
    Rename columns from Alpha Vantage company fundamentals to match the database schema,
    and clean placeholder values.
    """
    df = df.rename(columns=COMPANY_FUNDAMENTALS_MAP)
    df.replace(
        to_replace=["None", "none", "NaN", "nan", "", "-"],
        value=np.nan,
        inplace=True
    )
    df = df.infer_objects(copy=False)
    df = df.where(pd.notnull(df), None)
    return df

def standardize_annual_income_statement_columns(df):
    """
    Rename and clean Alpha Vantage annual income statement columns for DB insert.
    Ensures numeric columns are properly cast and missing values are handled.
    """
    df = df.rename(columns=ANNUAL_INCOME_STATEMENT_MAP)
    df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan, inplace=True)
    df = df.infer_objects(copy=False)
    for col in INCOME_STATEMENT_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors="coerce")
    df = df.where(pd.notnull(df), None)
    return df


def standardize_quarterly_income_statement_columns(df):
    """
    Rename and clean Alpha Vantage quarterly income statement columns for DB insert.
    Ensures numeric columns are properly cast and missing values are handled.
    """
    df = df.rename(columns=QUARTERLY_INCOME_STATEMENT_MAP)
    df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan,
        inplace=True
    )
    df = df.infer_objects(copy=False)  # Avoids future warning
    for col in INCOME_STATEMENT_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors="coerce")
    df = df.where(pd.notnull(df), None)
    return df


def standardize_annual_balance_sheet_columns(df):
    """
    Rename and type-cast columns from Alpha Vantage annual balance sheet to match the database schema.
    Ensures integer columns are properly cast and missing values are handled.
    """
    df = df.rename(columns=ANNUAL_BALANCE_SHEET_MAP)
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')
    # Cast float columns
    for col in ANNUAL_BALANCE_SHEET_FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Cast integer columns
    for col in ANNUAL_BALANCE_SHEET_INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')
    # Replace missing values with None
    df = df.where(pd.notnull(df), None)
    return df


def standardize_quarterly_balance_sheet_columns(df):
    """
    Rename and type-cast columns from Alpha Vantage quarterly balance sheet to match the database schema.
    Ensures integer columns are properly cast and missing values are handled.
    """
    df = df.rename(columns=QUARTERLY_BALANCE_SHEET_MAP)
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')
    for col in QUARTERLY_BALANCE_SHEET_FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in QUARTERLY_BALANCE_SHEET_INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')
    df = df.where(pd.notnull(df), None)
    return df


def standardize_annual_cash_flow_columns(df):
    """Rename and type-cast Alpha Vantage annual cash flow columns to DB schema."""
    df = df.rename(columns=ANNUAL_CASH_FLOW_MAP)
    df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan, inplace=True)
    df = df.infer_objects(copy=False)
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')
    float_cols = [col for col in df.columns if col not in ['symbol', 'fiscal_date_ending', 'reported_currency']]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.where(pd.notnull(df), None)
    return df


def standardize_quarterly_cash_flow_columns(df):
    """Rename and type-cast Alpha Vantage quarterly cash flow columns to DB schema."""
    df = df.rename(columns=QUARTERLY_CASH_FLOW_MAP)
    df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan, inplace=True)
    df = df.infer_objects(copy=False)
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')
    float_cols = [col for col in df.columns if col not in ['symbol', 'fiscal_date_ending', 'reported_currency']]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.where(pd.notnull(df), None)
    return df


def standardize_annual_earnings_columns(df):
    """
    Standardizes annual earnings columns from Alpha Vantage data for database compatibility.
    """
    df = df.rename(columns=ANNUAL_EARNINGS_MAP)
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')
    if 'reported_eps' in df.columns:
        df['reported_eps'] = pd.to_numeric(df['reported_eps'], errors='coerce')
    if 'symbol' in df.columns and 'symbol' in ANNUAL_EARNINGS_MAP.values():
        df['symbol'] = df['symbol'].astype(str)
    df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan, inplace=True)
    df = df.infer_objects(copy=False)
    df = df.where(pd.notnull(df), None)
    return df

def standardize_quarterly_earnings_columns(df):
    """
    Standardizes quarterly earnings columns from Alpha Vantage data for database compatibility.
    """
    df = df.rename(columns=QUARTERLY_EARNINGS_MAP)
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')
    if 'reported_date' in df.columns:
        df['reported_date'] = pd.to_datetime(df['reported_date'], errors='coerce')
    float_cols = ['reported_eps', 'estimated_eps', 'surprise', 'surprise_percentage']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'report_time' in df.columns:
        df['report_time'] = df['report_time'].astype(str)
    if 'symbol' in df.columns and 'symbol' in QUARTERLY_EARNINGS_MAP.values():
        df['symbol'] = df['symbol'].astype(str)
    df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan, inplace=True)
    df = df.infer_objects(copy=False)
    df = df.where(pd.notnull(df), None)
    return df

def standardize_insider_transaction_columns(df):
    """
    Standardizes Alpha Vantage insider transaction data for database ingestion.

    - Renames columns.
    - Parses date and numeric columns.
    - Handles missing values and duplicates.
    - Warns if required columns are missing.

    Args:
        df (pd.DataFrame): Raw DataFrame of insider transactions.

    Returns:
        pd.DataFrame: Standardized DataFrame.
    """
    df = df.rename(columns=INSIDER_TRANSACTION_MAP)
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    for col in ["shares", "share_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan)
    df = df.infer_objects(copy=False)
    df = df.where(pd.notnull(df), None)
    required_cols = ["transaction_date", "symbol"]
    for c in required_cols:
        if c not in df.columns:
            logger.warning("Required column '%s' is missing after standardization!", c)
    df.drop_duplicates(
        subset=[
            "transaction_date",
            "symbol",
            "executive",
            "security_type",
            "acquisition_or_disposal"
        ],
        inplace=True)
    return df

def standardize_stock_split_columns(df):
    """
    Standardizes Alpha Vantage stock split data for database ingestion.

    - Renames columns.
    - Parses date and numeric columns.
    - Handles nulls and placeholder values.
    - Removes duplicates.

    Args:
        df (pd.DataFrame): Raw DataFrame of stock splits.

    Returns:
        pd.DataFrame: Standardized DataFrame.
    """
    df = df.rename(columns=STOCK_SPLIT_MAP)
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df["split_factor"] = pd.to_numeric(df["split_factor"], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)
    df = df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan)
    df = df.infer_objects(copy=False)
    df = df.where(pd.notnull(df), None)
    df.drop_duplicates(subset=["symbol", "effective_date"], inplace=True)
    return df

def standardize_dividends_columns(df):
    """
    Standardizes Alpha Vantage dividends data for database compatibility.

    - Renames columns.
    - Parses all date columns to Python date objects or None.
    - Converts amount to numeric and symbol to string.
    - Handles nulls and placeholder values.
    - Removes duplicates.

    Args:
        df (pd.DataFrame): Raw DataFrame containing dividend data.

    Returns:
        pd.DataFrame: Cleaned and standardized DataFrame ready for database insertion.
    """
    df = df.rename(columns=DIVIDENDS_MAP)
    df["ex_dividend_date"] = pd.to_datetime(df["ex_dividend_date"], errors="coerce")
    df["declaration_date"] = pd.to_datetime(df["declaration_date"], errors="coerce")
    df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")
    df["payment_date"] = pd.to_datetime(df["payment_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)
    df = df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan)
    for col in ["ex_dividend_date", "declaration_date", "record_date", "payment_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").apply(lambda x: x.date() if pd.notnull(x) else None)
            df[col] = df[col].apply(lambda x: None if pd.isnull(x) or x is pd.NaT or x == "NaT" else x)
    return df

def preprocess_daily_timeseries(df, symbol):
    """
    Preprocesses daily timeseries data for DB insert.
    Handles missing data, column standardization, type conversion, metadata, and errors.
    """
    expected_columns = ['open', 'high', 'low', 'close', 'volume']
    if df is None or df.empty or not all(col in df.columns for col in expected_columns):
        logger.warning(f"Daily timeseries data missing or malformed for symbol {symbol}. Inserting dummy row.")
        df = pd.DataFrame([{
            "date": pd.Timestamp("1900-01-01"),
            "symbol": symbol,
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": None,
            "data_state": "dummy",
            "last_update": pd.Timestamp.now()
        }])
        return df

    df = df.copy()
    df.columns = expected_columns
    df.index = pd.to_datetime(df.index, errors="coerce")
    df.sort_index(ascending=True, axis=0, inplace=True)
    df.reset_index(inplace=True, names=["date"])
    df["symbol"] = symbol
    df["data_state"] = ""
    df["last_update"] = pd.Timestamp.now()
    df = df[["date", "symbol", "open", "high", "low", "close", "volume", "data_state", "last_update"]]
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").round().astype("Int64")
    df.dropna(subset=["date"], inplace=True)
    df = df.where(pd.notnull(df), None)
    return df

def preprocess_company_fundamentals(df, symbol):
    """
    Preprocesses company fundamentals for DB insert.
    Handles missing data, column standardization, type conversion, metadata, and errors.
    """
    if df is None or df.empty:
        logger.warning(f"Company fundamentals missing for symbol {symbol}. Inserting dummy row.")
        df = pd.DataFrame([{
            "symbol": symbol,
            "dividend_date": "1900-01-01",
            "ex_dividend_date": "1900-01-01",
            "data_state": "dummy",
            "last_update": pd.Timestamp.now()
        }])
        return df

    df = standardize_company_fundamentals_columns(df)
    df = clean_numeric_types_for_db(
        df,
        int_columns=COMPANY_FUNDAMENTALS_INT_COLUMNS,
        float_columns=COMPANY_FUNDAMENTALS_FLOAT_COLUMNS
    )
    df["symbol"] = symbol
    df["data_state"] = ""
    df["last_update"] = pd.Timestamp.now()

    # Fix date columns: convert NaN/float to None or valid date string
    for col in ["dividend_date", "ex_dividend_date"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: None if pd.isna(x) or str(x).lower() == "nan" else str(x)
            )
    df = df.where(pd.notnull(df), None)
    return df

def preprocess_annual_income_statement(df, symbol):
    if df is None or df.empty:
        logger.warning(f"Annual income statement missing for symbol {symbol}. Inserting dummy row.")
        df = pd.DataFrame([{
            "fiscal_date_ending": pd.Timestamp("1900-01-01"),
            "symbol": symbol,
            "data_state": "dummy",
            "last_update": pd.Timestamp.now()
        }])
        return df
    df = standardize_annual_income_statement_columns(df)
    df = clean_numeric_types_for_db(df, int_columns=None, float_columns=INCOME_STATEMENT_NUMERIC_COLUMNS)
    df["symbol"] = symbol
    df["data_state"] = ""
    df["last_update"] = pd.Timestamp.now()
    df.dropna(subset=["fiscal_date_ending"], inplace=True)
    df = df.where(pd.notnull(df), None)
    return df

def preprocess_quarterly_income_statement(df, symbol):
    if df is None or df.empty:
        logger.warning(f"Quarterly income statement missing for symbol {symbol}. Inserting dummy row.")
        df = pd.DataFrame([{
            "fiscal_date_ending": pd.Timestamp("1900-01-01"),
            "symbol": symbol,
            "data_state": "dummy",
            "last_update": pd.Timestamp.now()
        }])
        return df
    df = standardize_quarterly_income_statement_columns(df)
    df = clean_numeric_types_for_db(df, int_columns=None, float_columns=INCOME_STATEMENT_NUMERIC_COLUMNS)
    df["symbol"] = symbol
    df["data_state"] = ""
    df["last_update"] = pd.Timestamp.now()
    df.dropna(subset=["fiscal_date_ending"], inplace=True)
    df = df.where(pd.notnull(df), None)
    return df

def preprocess_annual_balance_sheet(df, symbol):
    if df is None or df.empty:
        logger.warning(f"Annual balance sheet missing for symbol {symbol}. Inserting dummy row.")
        df = pd.DataFrame([{
            "fiscal_date_ending": pd.Timestamp("1900-01-01"),
            "symbol": symbol,
            "data_state": "dummy",
            "last_update": pd.Timestamp.now()
        }])
        return df
    df = standardize_annual_balance_sheet_columns(df)
    df = clean_numeric_types_for_db(
        df,
        int_columns=ANNUAL_BALANCE_SHEET_INT_COLUMNS,
        float_columns=ANNUAL_BALANCE_SHEET_FLOAT_COLUMNS
    )
    df["symbol"] = symbol
    df["data_state"] = ""
    df["last_update"] = pd.Timestamp.now()
    df.dropna(subset=["fiscal_date_ending"], inplace=True)
    df = df.where(pd.notnull(df), None)
    return df

def preprocess_quarterly_balance_sheet(df, symbol):
    if df is None or df.empty:
        logger.warning(f"Quarterly balance sheet missing for symbol {symbol}. Inserting dummy row.")
        df = pd.DataFrame([{
            "fiscal_date_ending": pd.Timestamp("1900-01-01"),
            "symbol": symbol,
            "data_state": "dummy",
            "last_update": pd.Timestamp.now()
        }])
        return df
    df = standardize_quarterly_balance_sheet_columns(df)
    df = clean_numeric_types_for_db(
        df,
        int_columns=QUARTERLY_BALANCE_SHEET_INT_COLUMNS,
        float_columns=QUARTERLY_BALANCE_SHEET_FLOAT_COLUMNS
    )
    df["symbol"] = symbol
    df["data_state"] = ""
    df["last_update"] = pd.Timestamp.now()
    df.dropna(subset=["fiscal_date_ending"], inplace=True)
    df = df.where(pd.notnull(df), None)
    return df

def preprocess_annual_cash_flow(df, symbol):
    if df is None or df.empty:
        logger.warning(f"Annual cash flow missing for symbol {symbol}. Inserting dummy row.")
        df = pd.DataFrame([{
            "fiscal_date_ending": pd.Timestamp("1900-01-01"),
            "symbol": symbol,
            "data_state": "dummy",
            "last_update": pd.Timestamp.now()
        }])
        return df
    df = standardize_annual_cash_flow_columns(df)
    df["symbol"] = symbol
    df["data_state"] = ""
    df["last_update"] = pd.Timestamp.now()
    df.dropna(subset=["fiscal_date_ending"], inplace=True)
    df = df.where(pd.notnull(df), None)
    return df

def preprocess_quarterly_cash_flow(df, symbol):
    if df is None or df.empty:
        logger.warning(f"Quarterly cash flow missing for symbol {symbol}. Inserting dummy row.")
        df = pd.DataFrame([{
            "fiscal_date_ending": pd.Timestamp("1900-01-01"),
            "symbol": symbol,
            "data_state": "dummy",
            "last_update": pd.Timestamp.now()
        }])
        return df
    df = standardize_quarterly_cash_flow_columns(df)
    df["symbol"] = symbol
    df["data_state"] = ""
    df["last_update"] = pd.Timestamp.now()
    df.dropna(subset=["fiscal_date_ending"], inplace=True)
    df = df.where(pd.notnull(df), None)
    return df

def preprocess_annual_earnings(df, symbol):
    if df is None or df.empty:
        logger.warning(f"Annual earnings missing for symbol {symbol}. Inserting dummy row.")
        df = pd.DataFrame([{
            "fiscal_date_ending": pd.Timestamp("1900-01-01"),
            "symbol": symbol,
            "reported_eps": None,
            "data_state": "dummy",
            "last_update": pd.Timestamp.now()
        }])
        return df
    df = standardize_annual_earnings_columns(df)
    df["symbol"] = symbol
    df["data_state"] = ""
    df["last_update"] = pd.Timestamp.now()
    df.dropna(subset=["fiscal_date_ending"], inplace=True)
    df = df.where(pd.notnull(df), None)
    return df

def preprocess_quarterly_earnings(df, symbol):
    if df is None or df.empty:
        logger.warning(f"Quarterly earnings missing for symbol {symbol}. Inserting dummy row.")
        df = pd.DataFrame([{
            "fiscal_date_ending": pd.Timestamp("1900-01-01"),
            "symbol": symbol,
            "reported_eps": None,
            "data_state": "dummy",
            "last_update": pd.Timestamp.now()
        }])
        return df
    df = standardize_quarterly_earnings_columns(df)
    df["symbol"] = symbol
    df["data_state"] = ""
    df["last_update"] = pd.Timestamp.now()
    df.dropna(subset=["fiscal_date_ending"], inplace=True)
    df = df.where(pd.notnull(df), None)
    return df

def preprocess_insider_transactions(df, symbol):
    required_cols = [
        "transaction_date", "symbol", "executive", "executive_title",
        "security_type", "acquisition_or_disposal", "shares", "share_price"
    ]
    # If no data, create a dummy row with correct types
    if df is None or df.empty:
        logger.warning(f"Insider transactions missing for symbol {symbol}. Inserting dummy row.")
        dummy_row = {
            "transaction_date": datetime.date(1900, 1, 1),
            "symbol": symbol,
            "executive": "",
            "executive_title": "",
            "security_type": "",
            "acquisition_or_disposal": "",
            "shares": 0.0,
            "share_price": 0.0
        }
        df = pd.DataFrame([dummy_row])
    else:
        # Standardize columns
        df = standardize_insider_transaction_columns(df)
        df["symbol"] = symbol
        # Ensure all required columns exist
        for col in required_cols:
            if col not in df.columns:
                df[col] = "" if col in ["executive", "executive_title", "security_type", "acquisition_or_disposal"] else 0.0 if col in ["shares", "share_price"] else symbol if col == "symbol" else datetime.date(1900, 1, 1)
        # Clean types
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce").dt.date.fillna(datetime.date(1900, 1, 1))
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
        df["share_price"] = pd.to_numeric(df["share_price"], errors="coerce").fillna(0.0)
        for col in ["executive", "executive_title", "security_type", "acquisition_or_disposal"]:
            df[col] = df[col].fillna("").astype(str)
        df = df.where(pd.notnull(df), None)
    # Only keep required columns, in order
    df = df[required_cols]
    return df

def preprocess_stock_splits(df, symbol):
    required_cols = ["symbol", "effective_date", "split_factor", "data_state", "last_update"]
    if df is None or df.empty:
        logger.warning(f"Stock splits missing for symbol {symbol}. Inserting dummy row.")
        dummy_row = {
            "symbol": symbol,
            "effective_date": "1900-01-01",  # string, not Timestamp
            "split_factor": 0.0,
            "data_state": "dummy",
            "last_update": pd.Timestamp.now()
        }
        df = pd.DataFrame([dummy_row])
    else:
        df = standardize_stock_split_columns(df)
        df["symbol"] = symbol
        df["data_state"] = ""
        df["last_update"] = pd.Timestamp.now()
        for col in required_cols:
            if col not in df.columns:
                df[col] = "" if col == "data_state" else 0.0 if col == "split_factor" else "1900-01-01" if col == "effective_date" else pd.Timestamp.now() if col == "last_update" else symbol
        df["split_factor"] = pd.to_numeric(df["split_factor"], errors="coerce").fillna(0.0)
        df["effective_date"] = df["effective_date"].apply(lambda x: str(x) if pd.notnull(x) else "1900-01-01")
        df = df.where(pd.notnull(df), None)
    # Ensure column order and shape
    df = df[required_cols]
    return df

def preprocess_dividends(df, symbol):
    if df is None or df.empty:
        logger.warning(f"Dividends missing for symbol {symbol}. Inserting dummy row.")
        df = pd.DataFrame([{
            "symbol": symbol,
            "ex_dividend_date": pd.Timestamp("1900-01-01").date(),
            "amount": None,
            "data_state": "dummy",
            "last_update": pd.Timestamp.now()
        }])
        return df
    df = standardize_dividends_columns(df)
    df["symbol"] = symbol
    df["data_state"] = ""
    df["last_update"] = pd.Timestamp.now()
    df.dropna(subset=["ex_dividend_date"], inplace=True)
    df = df.where(pd.notnull(df), None)
    return df
