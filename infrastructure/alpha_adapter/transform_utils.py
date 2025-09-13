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

import logging
from datetime import date
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from .column_maps import (
    COMPANY_FUNDAMENTALS_MAP, DIVIDENDS_MAP, STOCK_SPLIT_MAP, INSIDER_TRANSACTION_MAP,
    ANNUAL_BALANCE_SHEET_MAP, QUARTERLY_BALANCE_SHEET_MAP,
    ANNUAL_CASH_FLOW_MAP, QUARTERLY_CASH_FLOW_MAP,
    ANNUAL_EARNINGS_MAP, QUARTERLY_EARNINGS_MAP,
    ANNUAL_INCOME_STATEMENT_MAP, QUARTERLY_INCOME_STATEMENT_MAP, DAILY_TIMESERIES_MAP
)

from infrastructure.databases.company.postgre_manager.company_table_objects import (
    CompanyFundamentals, DailyTimeSeries, AnnualBalanceSheet, QuarterlyBalanceSheet,
    AnnualCashFlow, QuarterlyCashFlow, AnnualEarnings, QuarterlyEarnings,
    AnnualIncomeStatement, QuarterlyIncomeStatement, InsiderTransactions, StockSplit, Dividends
)

def standardize_and_clean(
    df,
    column_map=None,
    date_cols=None,
    float_cols=None,
    int_cols=None,
    symbol=None,
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

    if orm_columns:
        orm_col_objs = orm_columns if hasattr(orm_columns[0], "type") else None
        orm_col_names = [col.name if orm_col_objs else col for col in orm_columns]
        for idx, col in enumerate(orm_col_names):
            if orm_col_objs:
                col_type = str(orm_columns[idx].type).lower()
            else:
                col_type = "float" if col in (float_cols or []) else "int" if col in (int_cols or []) else "date" if col in (date_cols or []) else ""
            if col in df.columns:
                if "float" in col_type or "numeric" in col_type or "double" in col_type:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif "integer" in col_type or "bigint" in col_type:
                    df[col] = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
                elif "date" in col_type:
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

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

    if always_string_cols:
        for col in always_string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
    if always_float_cols:
        for col in always_float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if always_date_cols:
        for col in always_date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    if symbol is not None:
        df["symbol"] = symbol

    if orm_columns:
        orm_col_names = [col.name if hasattr(col, "name") else col for col in orm_columns]
        for col in orm_col_names:
            if col not in df.columns:
                df[col] = None
        df = df[orm_col_names]

    df = df.where(pd.notnull(df), None)

    if df.empty and dummy_row:
        df = pd.DataFrame([dummy_row])

    return df

def create_dummy_row_with_dates(orm_columns, symbol):
    """
    Create a dummy row with proper date defaults for date columns.
    PK date columns get dummy_date, non-PK date columns get None.
    """
    dummy_row = {}
    dummy_date = date(1900, 1, 1)
    for col in orm_columns:
        if col.name == 'symbol':
            dummy_row[col.name] = symbol
        elif "date" in str(col.type).lower():
            if getattr(col, "primary_key", False):
                dummy_row[col.name] = dummy_date
            else:
                dummy_row[col.name] = None
        else:
            dummy_row[col.name] = None
    return dummy_row

def preprocess_company_fundamentals(df, symbol):
    if should_skip_symbol(df, symbol, "company fundamentals"):
        return None
    logger.info(f"Before transforming company_fundamentals for {symbol}:\n{df.to_string()}")
    orm_columns = [col for col in CompanyFundamentals.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=COMPANY_FUNDAMENTALS_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    df['symbol'] = symbol
    df.dropna(subset=['symbol'], inplace=True)

    for col in ['dividend_date', 'ex_dividend_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
            df[col] = df[col].where(pd.notnull(df[col]), None)

    logger.info(f"Transformed company_fundamentals for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_daily_timeseries(df, symbol):
    if should_skip_symbol(df, symbol, "daily timeseries"):
        return None
    df = df.reset_index()
    df = df.rename(columns={'index': 'date'})
    orm_columns = [col for col in DailyTimeSeries.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=DAILY_TIMESERIES_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    # Drop rows missing PKs
    df.dropna(subset=["date", "symbol"], inplace=True)
    logger.info(f"Transformed daily_timeseries for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_annual_balance_sheet(df, symbol):
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
    df.dropna(subset=["symbol", "fiscal_date_ending"], inplace=True)
    if 'common_stock_shares_outstanding' in df.columns:
        df['common_stock_shares_outstanding'] = (
            pd.to_numeric(df['common_stock_shares_outstanding'], errors='coerce')
            .round()
            .astype('Int64')
        )
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
    df.dropna(subset=["symbol", "fiscal_date_ending"], inplace=True)
    if 'common_stock_shares_outstanding' in df.columns:
        df['common_stock_shares_outstanding'] = (
            pd.to_numeric(df['common_stock_shares_outstanding'], errors='coerce')
            .round()
            .astype('Int64')
        )
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
    if should_skip_symbol(df, symbol, "annual cash flow"):
        return None
    orm_columns = [col for col in AnnualCashFlow.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=ANNUAL_CASH_FLOW_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    df.dropna(subset=["symbol", "fiscal_date_ending"], inplace=True)
    logger.info(f"Transformed annual_cash_flow for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_quarterly_cash_flow(df, symbol):
    if should_skip_symbol(df, symbol, "quarterly cash flow"):
        return None
    orm_columns = [col for col in QuarterlyCashFlow.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=QUARTERLY_CASH_FLOW_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    df.dropna(subset=["symbol", "fiscal_date_ending"], inplace=True)
    logger.info(f"Transformed quarterly_cash_flow for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_annual_earnings(df, symbol):
    if should_skip_symbol(df, symbol, "annual earnings"):
        return None
    orm_columns = [col for col in AnnualEarnings.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=ANNUAL_EARNINGS_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    df.dropna(subset=["symbol", "fiscal_date_ending"], inplace=True)
    logger.info(f"Transformed annual_earnings for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_quarterly_earnings(df, symbol):
    if should_skip_symbol(df, symbol, "quarterly earnings"):
        return None
    orm_columns = [col for col in QuarterlyEarnings.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=QUARTERLY_EARNINGS_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    df.dropna(subset=["symbol", "fiscal_date_ending"], inplace=True)
    logger.info(f"Transformed quarterly_earnings for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_annual_income_statement(df, symbol):
    if should_skip_symbol(df, symbol, "annual income statement"):
        return None
    orm_columns = [col for col in AnnualIncomeStatement.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=ANNUAL_INCOME_STATEMENT_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    df.dropna(subset=["symbol", "fiscal_date_ending"], inplace=True)
    logger.info(f"Transformed annual_income_statement for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_quarterly_income_statement(df, symbol):
    if should_skip_symbol(df, symbol, "quarterly income statement"):
        return None
    orm_columns = [col for col in QuarterlyIncomeStatement.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=QUARTERLY_INCOME_STATEMENT_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    df.dropna(subset=["symbol", "fiscal_date_ending"], inplace=True)
    logger.info(f"Transformed quarterly_income_statement for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_insider_transactions(df, symbol):
    if should_skip_symbol(df, symbol, "insider transactions"):
        return None
    orm_columns = [col for col in InsiderTransactions.__table__.columns]
    dummy_row = create_dummy_row_with_dates(orm_columns, symbol)
    df = standardize_and_clean(
        df,
        column_map=INSIDER_TRANSACTION_MAP,
        symbol=symbol,
        orm_columns=[col.name for col in orm_columns],
        dummy_row=dummy_row
    )
    # Drop rows missing PKs
    required_fields = [
        'transaction_date', 'symbol', 'executive', 'security_type', 'acquisition_or_disposal'
    ]
    for col in required_fields:
        df = df[df[col].notna()]
    if 'shares' in df.columns:
        df['shares'] = pd.to_numeric(df['shares'], errors='coerce')
    if 'share_price' in df.columns:
        df['share_price'] = pd.to_numeric(df['share_price'], errors='coerce')
    df = df[(df['shares'].notna()) & (df['shares'] > 0)]
    zero_price_allowed = [
        "Non-Qualified Stock Option (right to buy)",
        "Incentive Stock Option (right to buy)",
        "Restricted Stock Unit"
    ]
    df = df[
        (df['security_type'].isin(zero_price_allowed)) |
        ((df['share_price'].notna()) & (df['share_price'] > 0))
    ]
    df['share_price'] = df['share_price'].fillna(0.0)
    # Group by PK columns and aggregate to handle duplicates
    if not df.empty and len(df) > 1:
        pk_columns = ['transaction_date', 'symbol', 'executive', 'executive_title', 'security_type', 'acquisition_or_disposal']
        duplicates = df.duplicated(subset=pk_columns, keep=False)
        if duplicates.any():
            logger.info(f"Found {duplicates.sum()} duplicate rows for {symbol}, aggregating...")
            def safe_weighted_average(group):
                shares = pd.to_numeric(group['shares'], errors='coerce')
                prices = pd.to_numeric(group['share_price'], errors='coerce')
                valid_mask = pd.notna(shares) & pd.notna(prices) & (shares > 0)
                if valid_mask.any():
                    valid_shares = shares[valid_mask]
                    valid_prices = prices[valid_mask]
                    return (valid_shares * valid_prices).sum() / valid_shares.sum()
                else:
                    return prices.dropna().iloc[0] if not prices.dropna().empty else None
            agg_dict = {
                'shares': lambda x: pd.to_numeric(x, errors='coerce').sum(),
                'share_price': lambda x: safe_weighted_average(pd.DataFrame({'shares': df.loc[x.index, 'shares'], 'share_price': x}))
            }
            df = df.groupby(pk_columns, as_index=False).agg(agg_dict)
            logger.info(f"After groupby aggregation for {symbol}: {len(df)} rows")
    logger.info(f"Transformed insider_transactions for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_stock_splits(df, symbol):
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
    df.dropna(subset=["symbol", "effective_date"], inplace=True)
    logger.info(f"Transformed stock_splits for {symbol}:\n{df.head().to_string()}")
    return df

def preprocess_dividends(df, symbol):
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
    # Drop rows missing PKs
    df.dropna(subset=["symbol", "ex_dividend_date"], inplace=True)
    df = df.where(pd.notnull(df), None)
    logger.info(f"After dropping rows without ex_dividend_date for {symbol}, remaining rows: {df.to_string()}")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # Group by PK, sum amount, take first non-null for optional dates
    if not df.empty:
        df = (
            df.groupby(["symbol", "ex_dividend_date"], as_index=False)
            .agg({
                "amount": "sum",
                "declaration_date": lambda x: next((d for d in x if d is not None and not (isinstance(d, float) and np.isnan(d))), None),
                "record_date": lambda x: next((d for d in x if d is not None and not (isinstance(d, float) and np.isnan(d))), None),
                "payment_date": lambda x: next((d for d in x if d is not None and not (isinstance(d, float) and np.isnan(d))), None),
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
