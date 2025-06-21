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
    DIVIDENDS_MAP
    )


def standardize_company_fundamentals_columns(df):
    """
    Rename columns from Alpha Vantage company fundamentals to match the database schema.

    Args:
        df (pandas.DataFrame): Raw DataFrame from Alpha Vantage.

    Returns:
        pandas.DataFrame: DataFrame with columns renamed for downstream processing/storage.
    """
    return df.rename(columns=COMPANY_FUNDAMENTALS_MAP)


def standardize_annual_income_statement_columns(df):
    """
    Rename and clean Alpha Vantage annual income statement columns for DB insert.
    """
    df = df.rename(columns=ANNUAL_INCOME_STATEMENT_MAP)
    # Clean and convert values
    df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan,
        inplace=True
    )
    df = df.infer_objects(copy=False)
    for col in INCOME_STATEMENT_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(
            df['fiscal_date_ending'], errors="coerce")
    df = df.where(pd.notnull(df), None)
    return df


def standardize_quarterly_income_statement_columns(df):
    """
    Rename and clean Alpha Vantage quarterly income statement columns for DB insert.
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
    """
    df = df.rename(columns=ANNUAL_BALANCE_SHEET_MAP)
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(
            df['fiscal_date_ending'], errors='coerce')
    for col in ANNUAL_BALANCE_SHEET_FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ANNUAL_BALANCE_SHEET_INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan, inplace=True)
    return df


def standardize_quarterly_balance_sheet_columns(df):
    """
    Rename and type-cast columns from Alpha Vantage quarterly balance sheet to match the database schema.
    """
    df = df.rename(columns=QUARTERLY_BALANCE_SHEET_MAP)
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(
            df['fiscal_date_ending'], errors='coerce')
    for col in QUARTERLY_BALANCE_SHEET_FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in QUARTERLY_BALANCE_SHEET_INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan, inplace=True)
    return df

def standardize_annual_cash_flow_columns(df):
    """Rename and type-cast Alpha Vantage annual cash flow columns to DB schema."""
    df = df.rename(columns=ANNUAL_CASH_FLOW_MAP)
    df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan, inplace=True)
    df = df.infer_objects(copy=False)

    # Date conversion
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')

    # Float columns (all except symbol, date, currency)
    float_cols = [col for col in df.columns if col not in ['symbol', 'fiscal_date_ending', 'reported_currency']]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def standardize_quarterly_cash_flow_columns(df):
    """Rename and type-cast Alpha Vantage quarterly cash flow columns to DB schema."""
    df = df.rename(columns=QUARTERLY_CASH_FLOW_MAP)
    df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan, inplace=True)
    df = df.infer_objects(copy=False)

    # Date conversion
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')

    # Float columns (all except symbol, date, currency)
    float_cols = [col for col in df.columns if col not in ['symbol', 'fiscal_date_ending', 'reported_currency']]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def standardize_annual_earnings_columns(df):
    """
    Standardizes annual earnings columns from Alpha Vantage data for database compatibility.

    - Renames columns.
    - Parses date and numeric columns.
    - Converts symbol to string.
    - Handles nulls and placeholder values.

    Args:
        df (pd.DataFrame): Raw DataFrame of annual earnings data.

    Returns:
        pd.DataFrame: Standardized DataFrame.
    """
    df = df.rename(columns=ANNUAL_EARNINGS_MAP)
    if 'fiscal_date_ending' in df.columns:
        df['fiscal_date_ending'] = pd.to_datetime(df['fiscal_date_ending'], errors='coerce')
    if 'reported_eps' in df.columns:
        df['reported_eps'] = pd.to_numeric(df['reported_eps'], errors='coerce')
    if 'symbol' in df.columns and 'symbol' in ANNUAL_EARNINGS_MAP.values():
        df['symbol'] = df['symbol'].astype(str)
    df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan,
        inplace=True
    )
    df = df.infer_objects(copy=False)
    return df

def standardize_quarterly_earnings_columns(df):
    """
    Standardizes quarterly earnings columns from Alpha Vantage data for database compatibility.

    - Renames columns.
    - Parses date and numeric columns.
    - Converts symbol and report_time to string.
    - Handles nulls and placeholder values.

    Args:
        df (pd.DataFrame): Raw DataFrame of quarterly earnings data.

    Returns:
        pd.DataFrame: Standardized DataFrame.
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
    df.replace(
        to_replace=["None", "none", "NaN", "nan", ""],
        value=np.nan,
        inplace=True
    )
    df = df.infer_objects(copy=False)
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
            print(f"❌ Warning: Required column '{c}' is missing after standardization!")
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
    df.drop_duplicates(subset=["symbol", "ex_dividend_date"], inplace=True)
    return df
