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

from .column_maps import (
    COMPANY_FUNDAMENTALS_MAP, DIVIDENDS_MAP, STOCK_SPLIT_MAP, INSIDER_TRANSACTION_MAP,
    ANNUAL_BALANCE_SHEET_MAP, QUARTERLY_BALANCE_SHEET_MAP,
    ANNUAL_CASH_FLOW_MAP, QUARTERLY_CASH_FLOW_MAP,
    ANNUAL_EARNINGS_MAP, QUARTERLY_EARNINGS_MAP,
    ANNUAL_INCOME_STATEMENT_MAP, QUARTERLY_INCOME_STATEMENT_MAP, DAILY_TIMESERIES_MAP
)

from infrastructure.databases.company.postgre_manager.postgre_objects import (
    CompanyFundamentals, DailyTimeSeries, AnnualBalanceSheet, QuarterlyBalanceSheet,
    AnnualCashFlow, QuarterlyCashFlow, AnnualEarnings, QuarterlyEarnings,
    AnnualIncomeStatement, QuarterlyIncomeStatement, InsiderTransactions, StockSplit, Dividends
)

import pandas as pd
import numpy as np
import logging
import datetime
from datetime import date

logger = logging.getLogger(__name__)

def standardize_and_clean(
    df,
    column_map=None,
    date_cols=None,
    float_cols=None,
    int_cols=None,
    symbol=None,
    dropna_col=None,
    always_string_cols=None,
    always_float_cols=None,
    always_date_cols=None,
    orm_columns=None,
    dummy_row=None
):
    """
    Standardize and clean a DataFrame for database ingestion.
    """
    if df is None or df.empty:
        return pd.DataFrame([dummy_row]) if dummy_row else pd.DataFrame()

    logger.info(f"Raw columns: {list(df.columns)}")
    if column_map:
        logger.info(f"Column map: {column_map}")
        df = df.rename(columns=column_map)
        logger.info(f"Columns after renaming: {list(df.columns)}")

    # Replace common placeholders with np.nan
    df.replace(to_replace=["None", "none", "NaN", "nan", "", "-", "null", "NULL"], value=np.nan, inplace=True)
    df = df.infer_objects(copy=False)

    # --- PATCH: Improved type conversion ---
    # Use ORM schema if provided, otherwise use explicit lists
    if orm_columns:
        # Get column names and types from ORM
        orm_col_objs = orm_columns if hasattr(orm_columns[0], "type") else None
        orm_col_names = [col.name if orm_col_objs else col for col in orm_columns]
        for idx, col in enumerate(orm_col_names):
            # Determine type
            if orm_col_objs:
                col_type = str(orm_columns[idx].type).lower()
            else:
                # Fallback: guess type from column name
                col_type = "float" if col in (float_cols or []) else "int" if col in (int_cols or []) else "date" if col in (date_cols or []) else ""
            if col in df.columns:
                if "float" in col_type or "numeric" in col_type or "double" in col_type:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif "integer" in col_type or "bigint" in col_type:
                    df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
                elif "date" in col_type:
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # Fallback for explicit lists
    if float_cols:
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    if int_cols:
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # Always convert specified columns to string, float, or date
    if always_string_cols:
        for col in always_string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
    if always_float_cols:
        for col in always_float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if always_date_cols:
        dummy_date = date(1900, 1, 1)
        for col in always_date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
                df[col] = df[col].apply(lambda x: dummy_date if pd.isna(x) or str(x) == "NaT" else x)

    # Add symbol column if needed
    if symbol is not None:
        df["symbol"] = symbol

    # Align columns with ORM schema
    if orm_columns:
        orm_col_names = [col.name if hasattr(col, "name") else col for col in orm_columns]
        for col in orm_col_names:
            if col not in df.columns:
                df[col] = None
        df = df[orm_col_names]

    # Replace remaining NaN with None for DB compatibility
    df = df.where(pd.notnull(df), None)

    # If DataFrame is empty, insert dummy row
    if df.empty and dummy_row:
        df = pd.DataFrame([dummy_row])

    return df

def create_dummy_row_with_dates(orm_columns, symbol):
    """
    Create a dummy row with proper date defaults for date columns.
    
    This function generates a dictionary representing a dummy database record
    with appropriate default values for each column type. Date columns receive
    a default date of 1900-01-01, symbol columns receive the provided symbol,
    and all other columns receive None.
    
    Args:
        orm_columns (list): List of SQLAlchemy column objects from ORM model
        symbol (str): Stock symbol to use for the symbol column
    
    Returns:
        dict: Dictionary with column names as keys and default values
        
    Examples:
        >>> dummy_row = create_dummy_row_with_dates(DailyTimeSeries.__table__.columns, "AAPL")
        >>> # Returns: {"date": date(1900, 1, 1), "symbol": "AAPL", "open": None, ...}
    """
    dummy_row = {}
    dummy_date = date(1900, 1, 1)  # Default date for all date columns
    
    for col in orm_columns:
        if col.name == 'symbol':
            dummy_row[col.name] = symbol
        elif str(col.type).lower().startswith('date'):
            dummy_row[col.name] = dummy_date
        else:
            dummy_row[col.name] = None
    
    return dummy_row

def preprocess_company_fundamentals(df, symbol):
    """
    Preprocess company fundamentals data from Alpha Vantage API.

    Transforms raw company fundamentals data into a format compatible with
    the CompanyFundamentalsTable ORM model. Handles column mapping, data
    type conversion, and missing value management.
    """
    if should_skip_symbol(df, symbol, "company fundamentals"):
        return None
    logger.info(f"Before transforming company_fundamentals for {symbol}:\n{df.to_string()}")

    orm_columns = [col for col in CompanyFundamentals.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)

    # --- PATCH: Robust missing data and type conversion ---
    # Replace all common placeholders with np.nan
    placeholders = ["None", "none", "NaN", "nan", "", "-", "null", "NULL"]
    df.replace(to_replace=placeholders, value=np.nan, inplace=True)

    # Rename columns
    df = df.rename(columns=COMPANY_FUNDAMENTALS_MAP)

    # Type conversion based on ORM schema
    for col in orm_columns:
        col_name = col.name
        col_type = str(col.type).lower()
        if col_name in df.columns:
            if "float" in col_type:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            elif "integer" in col_type or "bigint" in col_type:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").round().astype("Int64")
            elif "date" in col_type:
                df[col_name] = pd.to_datetime(df[col_name], errors="coerce").dt.date

    # --- PATCH: Replace NaT/nan in date columns with dummy date ---
    dummy_date = date(1900, 1, 1)
    for col in orm_columns:
        col_name = col.name
        col_type = str(col.type).lower()
        if "date" in col_type and col_name in df.columns:
            df[col_name] = df[col_name].apply(lambda x: dummy_date if pd.isna(x) or str(x) == "NaT" else x)

    # Add symbol column if needed
    if symbol is not None:
        df["symbol"] = symbol

    # Align columns with ORM schema
    orm_col_names = [col.name for col in orm_columns]
    for col in orm_col_names:
        if col not in df.columns:
            df[col] = None
    df = df[orm_col_names]

    # Replace remaining NaN with None for DB compatibility
    df = df.where(pd.notnull(df), None)

    # If DataFrame is empty, insert dummy row
    if df.empty:
        df = pd.DataFrame([dummy_row])

    logger.info(f"Transformed company_fundamentals for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_daily_timeseries(df, symbol):
    """
    Preprocess daily time series data from Alpha Vantage API.
    
    Transforms raw daily OHLCV (Open, High, Low, Close, Volume) data into a format
    compatible with the DailyTimeSeries ORM model. Handles date index conversion,
    column mapping from Alpha Vantage format, and data standardization.
    
    Args:
        df (pd.DataFrame): Raw daily time series DataFrame with date index from Alpha Vantage
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame with date column and standardized OHLCV data
        
    Examples:
        >>> processed_df = preprocess_daily_timeseries(raw_df, "AAPL")
    """
    if should_skip_symbol(df, symbol, "daily timeseries"):
        return None
    
    # Reset index to convert date index to a column
    df = df.reset_index()
    df = df.rename(columns={'index': 'date'})  # Rename the index column to 'date'
        
    orm_columns = [col for col in DailyTimeSeries.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    
    df = standardize_and_clean(
        df,
        column_map=DAILY_TIMESERIES_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    logger.info(f"Transformed daily_timeseries for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_annual_balance_sheet(df, symbol):
    """
    Preprocess annual balance sheet data from Alpha Vantage API.
    """
    if should_skip_symbol(df, symbol, "annual balance sheet"):
        return None
    logger.info(f"Before transforming annual_balance_sheet for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in AnnualBalanceSheet.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=ANNUAL_BALANCE_SHEET_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )

    # --- PATCH: Robust integer conversion for common_stock_shares_outstanding ---
    if 'common_stock_shares_outstanding' in df.columns:
        df['common_stock_shares_outstanding'] = (
            pd.to_numeric(df['common_stock_shares_outstanding'], errors='coerce')
            .round()
            .astype('Int64')
        )

    # --- PATCH: Assert no raw API keys remain ---
    raw_keys = [
        "intangibleAssetsExcludingGoodwill",
        "currentLongTermDebt",
        "longTermDebtNoncurrent",
    ]
    leftover = [col for col in raw_keys if col in df.columns]
    if leftover:
        raise ValueError(f"Unmapped API keys remain in annual balance sheet DataFrame: {leftover}")

    logger.info(f"Transformed annual_balance_sheet for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_quarterly_balance_sheet(df, symbol):
    """
    Preprocess quarterly balance sheet data from Alpha Vantage API.
    """
    if should_skip_symbol(df, symbol, "quarterly balance sheet"):
        return None
    logger.info(f"Before transforming quarterly_balance_sheet for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in QuarterlyBalanceSheet.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=QUARTERLY_BALANCE_SHEET_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )

    # --- PATCH: Robust integer conversion for common_stock_shares_outstanding ---
    if 'common_stock_shares_outstanding' in df.columns:
        df['common_stock_shares_outstanding'] = (
            pd.to_numeric(df['common_stock_shares_outstanding'], errors='coerce')
            .round()
            .astype('Int64')
        )

    # --- PATCH: Assert no raw API keys remain ---
    raw_keys = [
        "intangibleAssetsExcludingGoodwill",
        "currentLongTermDebt",
        "longTermDebtNoncurrent",
    ]
    leftover = [col for col in raw_keys if col in df.columns]
    if leftover:
        raise ValueError(f"Unmapped API keys remain in quarterly balance sheet DataFrame: {leftover}")

    logger.info(f"Transformed quarterly_balance_sheet for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_annual_cash_flow(df, symbol):
    """
    Preprocess annual cash flow statement data from Alpha Vantage API.
    
    Transforms raw annual cash flow data into a format compatible with
    the AnnualCashFlowTable ORM model. Handles cash flow statement
    data formatting and standardization.
    
    Args:
        df (pd.DataFrame): Raw annual cash flow DataFrame from Alpha Vantage
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for database insertion
    """
    if should_skip_symbol(df, symbol, "annual cash flow"):
        return None
    # logger.info(f"Before transforming annual_cash_flow for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in AnnualCashFlow.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=ANNUAL_CASH_FLOW_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    logger.info(f"Transformed annual_cash_flow for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_quarterly_cash_flow(df, symbol):
    """
    Preprocess quarterly cash flow statement data from Alpha Vantage API.
    
    Transforms raw quarterly cash flow data into a format compatible with
    the QuarterlyCashFlowTable ORM model. Handles quarterly cash flow
    statement data formatting and standardization.
    
    Args:
        df (pd.DataFrame): Raw quarterly cash flow DataFrame from Alpha Vantage
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for database insertion
    """
    if should_skip_symbol(df, symbol, "quarterly cash flow"):
        return None
    # logger.info(f"Before transforming quarterly_cash_flow for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in QuarterlyCashFlow.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=QUARTERLY_CASH_FLOW_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    logger.info(f"Transformed quarterly_cash_flow for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_annual_earnings(df, symbol):
    """
    Preprocess annual earnings data from Alpha Vantage API.
    
    Transforms raw annual earnings data into a format compatible with
    the AnnualEarningsTable ORM model. Handles earnings data
    formatting and standardization.
    
    Args:
        df (pd.DataFrame): Raw annual earnings DataFrame from Alpha Vantage
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for database insertion
    """
    if should_skip_symbol(df, symbol, "annual earnings"):
        return None
    # logger.info(f"Before transforming annual_earnings for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in AnnualEarnings.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=ANNUAL_EARNINGS_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    logger.info(f"Transformed annual_earnings for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_quarterly_earnings(df, symbol):
    """
    Preprocess quarterly earnings data from Alpha Vantage API.
    
    Transforms raw quarterly earnings data into a format compatible with
    the QuarterlyEarningsTable ORM model. Handles quarterly earnings
    data formatting and standardization.
    
    Args:
        df (pd.DataFrame): Raw quarterly earnings DataFrame from Alpha Vantage  
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for database insertion
    """
    if should_skip_symbol(df, symbol, "quarterly earnings"):
        return None
    # logger.info(f"Before transforming quarterly_earnings for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in QuarterlyEarnings.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=QUARTERLY_EARNINGS_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    logger.info(f"Transformed quarterly_earnings for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_annual_income_statement(df, symbol):
    """
    Preprocess annual income statement data from Alpha Vantage API.
    
    Transforms raw annual income statement data into a format compatible with
    the AnnualIncomeStatement ORM model. Handles income statement data
    formatting and standardization.
    
    Args:
        df (pd.DataFrame): Raw annual income statement DataFrame from Alpha Vantage
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for database insertion
    """
    if should_skip_symbol(df, symbol, "annual income statement"):
        return None
    # logger.info(f"Before transforming annual_income_statement for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in AnnualIncomeStatement.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=ANNUAL_INCOME_STATEMENT_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    logger.info(f"Transformed annual_income_statement for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_quarterly_income_statement(df, symbol):
    """
    Preprocess quarterly income statement data from Alpha Vantage API.
    
    Transforms raw quarterly income statement data into a format compatible with
    the QuarterlyIncomeStatement ORM model. Handles quarterly income statement
    data formatting and standardization.
    
    Args:
        df (pd.DataFrame): Raw quarterly income statement DataFrame from Alpha Vantage
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for database insertion
    """
    if should_skip_symbol(df, symbol, "quarterly income statement"):
        return None
    # logger.info(f"Before transforming quarterly_income_statement for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in QuarterlyIncomeStatement.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=QUARTERLY_INCOME_STATEMENT_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    logger.info(f"Transformed quarterly_income_statement for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_insider_transactions(df, symbol):
    """
    Preprocess insider transaction data from Alpha Vantage API.
    
    Transforms raw insider transaction data into a format compatible with
    the InsiderTransactions ORM model. Handles duplicate transaction aggregation,
    weighted average price calculation, and transaction data standardization.
    
    Special handling includes:
    - Aggregation of duplicate transactions by primary key
    - Weighted average calculation for share prices when aggregating
    - Sum of shares for transactions with same key attributes
    
    Args:
        df (pd.DataFrame): Raw insider transactions DataFrame from Alpha Vantage
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame with aggregated transactions ready for database insertion
        
    Examples:
        >>> processed_df = preprocess_insider_transactions(raw_df, "AAPL")
    """
    if should_skip_symbol(df, symbol, "insider transactions"):
        return None
    # logger.info(f"Before transforming insider_transactions for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in InsiderTransactions.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=INSIDER_TRANSACTION_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )

    # Ensure transaction_date is a valid date
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce').dt.date
        df = df[df['transaction_date'].notna()]

    # Ensure numeric columns are properly converted
    if 'shares' in df.columns:
        df['shares'] = pd.to_numeric(df['shares'], errors='coerce')
    if 'share_price' in df.columns:
        df['share_price'] = pd.to_numeric(df['share_price'], errors='coerce')

    # Remove rows missing required fields
    required_fields = [
        'transaction_date', 'symbol', 'executive', 'security_type', 'acquisition_or_disposal'
    ]
    for col in required_fields:
        df = df[df[col].notna()]

    # Security types that allow zero price
    zero_price_allowed = [
        "Non-Qualified Stock Option (right to buy)",
        "Incentive Stock Option (right to buy)",
        "Restricted Stock Unit"
    ]

    # Remove rows with shares <= 0 or NaN
    df = df[(df['shares'].notna()) & (df['shares'] > 0)]

    # Remove rows with share_price <= 0 or NaN for types that require price
    df = df[
        (df['security_type'].isin(zero_price_allowed)) |
        ((df['share_price'].notna()) & (df['share_price'] > 0))
    ]

    df['share_price'] = df['share_price'].fillna(0.0)  # Fill NaN prices with 0.0
    # Group by primary key columns and aggregate to handle duplicates
    if not df.empty and len(df) > 1:
        pk_columns = ['transaction_date', 'symbol', 'executive', 'executive_title', 'security_type', 'acquisition_or_disposal']
        
        # Check if we have duplicates based on primary key
        duplicates = df.duplicated(subset=pk_columns, keep=False)
        if duplicates.any():
            logger.info(f"Found {duplicates.sum()} duplicate rows for {symbol}, aggregating...")
            
            def safe_weighted_average(group):
                """Calculate weighted average price by shares, handling edge cases."""
                shares = pd.to_numeric(group['shares'], errors='coerce')
                prices = pd.to_numeric(group['share_price'], errors='coerce')
                
                # Filter out NaN values
                valid_mask = pd.notna(shares) & pd.notna(prices) & (shares > 0)
                
                if valid_mask.any():
                    valid_shares = shares[valid_mask]
                    valid_prices = prices[valid_mask]
                    return (valid_shares * valid_prices).sum() / valid_shares.sum()
                else:
                    # Return first non-null price, or NaN if all are null
                    return prices.dropna().iloc[0] if not prices.dropna().empty else None
            
            # Define aggregation functions
            agg_dict = {
                'shares': lambda x: pd.to_numeric(x, errors='coerce').sum(),
                'share_price': lambda x: safe_weighted_average(pd.DataFrame({'shares': df.loc[x.index, 'shares'], 'share_price': x}))
            }
            
            df = df.groupby(pk_columns, as_index=False).agg(agg_dict)
            logger.info(f"After groupby aggregation for {symbol}: {len(df)} rows")
    
    logger.info(f"Transformed insider_transactions for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_stock_splits(df, symbol):
    """
    Preprocess stock split data from Alpha Vantage API.
    
    Transforms raw stock split data into a format compatible with
    the StockSplit ORM model. Handles stock split event data
    formatting and standardization.
    
    Args:
        df (pd.DataFrame): Raw stock splits DataFrame from Alpha Vantage
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for database insertion
    """
    if should_skip_symbol(df, symbol, "stock splits"):
        return None
    logger.info(f"Before transforming stock_splits for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in StockSplit.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=STOCK_SPLIT_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    logger.info(f"Transformed stock_splits for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_dividends(df, symbol):
    """
    Preprocess dividend data from Alpha Vantage API.

    Transforms raw dividend data into a format compatible with
    the DividendsTable ORM model. Handles dividend payment data
    formatting and standardization, including filtering of records
    without valid ex-dividend dates.
    
    Args:
        df (pd.DataFrame): Raw dividends DataFrame from Alpha Vantage
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame with valid dividend records ready for database insertion
        
    Note:
        Records without valid ex_dividend_date values are dropped as they represent
        incomplete dividend information.
    """
    if should_skip_symbol(df, symbol, "dividends"):
        return None
    logger.info(f"Before transforming dividends for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in Dividends.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=DIVIDENDS_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    df.dropna(subset=["ex_dividend_date"], inplace=True)
    df = df.where(pd.notnull(df), None)
    logger.info(f"After dropping rows without ex_dividend_date for {symbol}, remaining rows: {df.to_string()}")

    # Fill every non-valid date with the dummy date
    dummy_date = date(1900, 1, 1)
    date_cols = ["declaration_date", "record_date", "payment_date"]
    df[date_cols] = df[date_cols].fillna(dummy_date)
    logger.info(f"After filling date columns with dummy date for {symbol}:\n{df[date_cols].to_string()}")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # Group by symbol and ex_dividend_date, sum amount, take first non-null for dates
    if not df.empty:
        df = (
            df.groupby(["symbol", "ex_dividend_date"], as_index=False)
            .agg({
                "amount": "sum",
                "declaration_date": lambda x: next((d for d in x if d is not None and not (isinstance(d, float) and np.isnan(d))), dummy_date),
                "record_date": lambda x: next((d for d in x if d is not None and not (isinstance(d, float) and np.isnan(d))), dummy_date),
                "payment_date": lambda x: next((d for d in x if d is not None and not (isinstance(d, float) and np.isnan(d))), dummy_date),
            })
        )
    logger.info(f"Transformed dividends for {symbol}:\n{df.to_string()}")
    return df

def should_skip_symbol(df, symbol, data_type):
    """
    Utility to check if the DataFrame is empty and log a skip message.
    Returns True if the symbol should be skipped.
    """
    if df is None or df.empty:
        logger.warning(f"Skipping {symbol}: No {data_type} data returned by API.")
        return True
    return False
