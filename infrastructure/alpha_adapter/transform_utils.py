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
    CompanyFundamentalsTable, DailyTimeSeries, AnnualBalanceSheetTable, QuarterlyBalanceSheetTable,
    AnnualCashFlowTable, QuarterlyCashFlowTable, AnnualEarningsTable, QuarterlyEarningsTable,
    AnnualIncomeStatement, QuarterlyIncomeStatement, InsiderTransactions, StockSplit, DividendsTable
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
    
    This function performs comprehensive data cleaning and standardization including:
    - Column renaming based on provided mapping
    - Data type conversion and validation
    - Missing value handling
    - Column alignment with ORM schema
    - Insertion of dummy rows for missing data
    
    Args:
        df (pd.DataFrame): Input DataFrame to be cleaned and standardized
        column_map (dict, optional): Mapping of original column names to standardized names
        date_cols (list, optional): Columns to convert to datetime format
        float_cols (list, optional): Columns to convert to float format
        int_cols (list, optional): Columns to convert to integer format
        symbol (str, optional): Stock symbol to add as a column
        dropna_col (str, optional): Column name to drop rows where this column is NaN
        always_string_cols (list, optional): Columns to always convert to string type
        always_float_cols (list, optional): Columns to always convert to float with 0.0 default
        always_date_cols (list, optional): Columns to always convert to date with 1900-01-01 default
        orm_columns (list, optional): List of expected ORM column names for alignment
        dummy_row (dict, optional): Dictionary representing a dummy row for missing data
    
    Returns:
        pd.DataFrame: Cleaned and standardized DataFrame ready for database ingestion
        
    Examples:
        >>> df = standardize_and_clean(
        ...     df, 
        ...     column_map={"old_name": "new_name"}, 
        ...     symbol="AAPL",
        ...     orm_columns=["date", "symbol", "open", "high", "low", "close"]
        ... )
    """
    if df is None or df.empty:
        logger.warning(f"Missing or empty data for symbol {symbol}. Inserting dummy row.")
        df = pd.DataFrame([dummy_row])
        return df

    logger.info(f"Raw columns: {list(df.columns)}")
    if column_map:
        logger.info(f"Column map: {column_map}")
        df = df.rename(columns=column_map)
        logger.info(f"Columns after renaming: {list(df.columns)}")
    df.replace(to_replace=["None", "none", "NaN", "nan", ""], value=np.nan, inplace=True)
    df = df.infer_objects(copy=False)

    if date_cols:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    if float_cols:
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    if int_cols:
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
    if always_string_cols:
        for col in always_string_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
    if always_float_cols:
        for col in always_float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")
    if always_date_cols:
        for col in always_date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date.fillna(datetime.date(1900, 1, 1))
    if symbol is not None:
        df["symbol"] = symbol
    
    df = df.where(pd.notnull(df), None)
    
    if orm_columns:
        logger.info(f"ORM columns: {orm_columns}")
        missing = [col for col in orm_columns if col not in df.columns]
        logger.info(f"Missing ORM columns: {missing}")
        for col in orm_columns:
            if col not in df.columns:
                df[col] = None
        df = df[orm_columns]

    if dropna_col:
        logger.info(f"Non-NA count for dropna_col '{dropna_col}': {df[dropna_col].notna().sum() if dropna_col in df.columns else 'Column missing'}")
        df.dropna(subset=[dropna_col], inplace=True)

    df = df.where(pd.notnull(df), None)

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
    
    Args:
        df (pd.DataFrame): Raw company fundamentals DataFrame from Alpha Vantage
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for database insertion
        
    Examples:
        >>> processed_df = preprocess_company_fundamentals(raw_df, "AAPL")
    """
    logger.info(f"Before transforming company_fundamentals for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in CompanyFundamentalsTable.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=COMPANY_FUNDAMENTALS_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
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
    logger.info(f"Before transforming daily_timeseries for {symbol}:\n{df.to_string()} original columns: {list(df.columns)}")
    
    # Reset index to convert date index to a column
    df = df.reset_index()
    df = df.rename(columns={'index': 'date'})  # Rename the index column to 'date'
    
    logger.info(f"After reset_index for {symbol}, columns: {list(df.columns)}")
    
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
    
    Transforms raw annual balance sheet data into a format compatible with
    the AnnualBalanceSheetTable ORM model. Handles financial statement
    data formatting and standardization.
    
    Args:
        df (pd.DataFrame): Raw annual balance sheet DataFrame from Alpha Vantage
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for database insertion
    """
    logger.info(f"Before transforming annual_balance_sheet for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in AnnualBalanceSheetTable.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=ANNUAL_BALANCE_SHEET_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    logger.info(f"Transformed annual_balance_sheet for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_quarterly_balance_sheet(df, symbol):
    """
    Preprocess quarterly balance sheet data from Alpha Vantage API.
    
    Transforms raw quarterly balance sheet data into a format compatible with
    the QuarterlyBalanceSheetTable ORM model. Handles quarterly financial
    statement data formatting and standardization.
    
    Args:
        df (pd.DataFrame): Raw quarterly balance sheet DataFrame from Alpha Vantage
        symbol (str): Stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
        pd.DataFrame: Processed DataFrame ready for database insertion
    """
    logger.info(f"Before transforming quarterly_balance_sheet for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in QuarterlyBalanceSheetTable.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=QUARTERLY_BALANCE_SHEET_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
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
    logger.info(f"Before transforming annual_cash_flow for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in AnnualCashFlowTable.__table__.columns]
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
    logger.info(f"Before transforming quarterly_cash_flow for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in QuarterlyCashFlowTable.__table__.columns]
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
    logger.info(f"Before transforming annual_earnings for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in AnnualEarningsTable.__table__.columns]
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
    logger.info(f"Before transforming quarterly_earnings for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in QuarterlyEarningsTable.__table__.columns]
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
    logger.info(f"Before transforming annual_income_statement for {symbol}:\n{df.to_string()}")
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
    logger.info(f"Before transforming quarterly_income_statement for {symbol}:\n{df.to_string()}")
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
    logger.info(f"Before transforming insider_transactions for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in InsiderTransactions.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=INSIDER_TRANSACTION_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    
    # Ensure numeric columns are properly converted
    if 'shares' in df.columns:
        df['shares'] = pd.to_numeric(df['shares'], errors='coerce')
    if 'share_price' in df.columns:
        df['share_price'] = pd.to_numeric(df['share_price'], errors='coerce')
    
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
    logger.info(f"Before transforming dividends for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in DividendsTable.__table__.columns]
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
    logger.info(f"Transformed dividends for {symbol}:\n{df.head().to_string()}")
    return df
