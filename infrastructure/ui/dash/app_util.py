import pandas as pd
import dash_mantine_components as dmc

# Shared alias for Dash store payloads
Records = list[dict] | None


# Date parsing hints derived from ORM table definitions (company_table_objects)
_DATE_COLUMNS_BY_TABLE: dict[str, list[str]] = {
    "daily_timeseries": ["date"],
    "dividends": ["date", "payment_date", "record_date", "declaration_date", "ex_dividend_date"],
    "company_fundamentals": ["latest_quarter", "dividend_date", "ex_dividend_date"],
    "balance_sheet_quarterly": ["fiscal_date_ending"],
    "annual_balance_sheet": ["fiscal_date_ending"],
    "earnings": ["fiscal_date_ending", "reported_date"],
    "income_statement_quarterly": ["fiscal_date_ending"],
    "cashflow_statement_quarterly": ["fiscal_date_ending"],
    "insider_transactions": ["transaction_date"],
}


DEFAULT_DATE_RANGE_YEARS = 2


def get_default_date_range(
    df: pd.DataFrame,
    date_col: str,
    *,
    years: int = DEFAULT_DATE_RANGE_YEARS,
) -> tuple[str | None, str | None]:
    """
    Compute a default date window from a DataFrame using a date column.

    The function assumes the input frame contains ``date_col``. It finds the latest
    available date, subtracts ``years`` (default 2 years), and returns both endpoints
    formatted as ``YYYY-MM-DD`` strings for use in Dash pickers.

    Args:
        df (pd.DataFrame): DataFrame containing the date column.
        date_col (str): Column name to use for date range calculations.
        years (int): Number of years to look back for the default range.

    Returns:
        tuple[str | None, str | None]: ``(start_date, end_date)`` as ISO date strings.
    """
    if df is None or df.empty or date_col not in df.columns:
        return None, None
    date_series = pd.to_datetime(df[date_col])
    end_date_val = date_series.max()
    if pd.isna(end_date_val):
        return None, None
    start_date_val = end_date_val - pd.DateOffset(years=years)
    start_date = start_date_val.strftime("%Y-%m-%d")
    end_date = end_date_val.strftime("%Y-%m-%d")
    return start_date, end_date


def get_last_2_years_range(price_df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Convenience wrapper for daily price data (uses the ``date`` column)."""
    return get_default_date_range(price_df, "date", years=DEFAULT_DATE_RANGE_YEARS)


def df_to_records(df):
    """
    Convert a DataFrame to a list-of-dicts suitable for storing in Dash dcc.Store.

    - Returns ``None`` for ``None`` or empty DataFrames to keep store payloads small.
    - Uses ``DataFrame.to_dict('records')`` for JSON-serializable output.

    Args:
        df (pd.DataFrame | None): Source DataFrame.

    Returns:
        list[dict] | None: List of row dictionaries, or ``None`` if input is missing/empty.
    """
    if df is None:
        return None
    if hasattr(df, 'empty') and df.empty:
        return None
    return df.to_dict("records")


def empty_load(status: str) -> tuple:
    """
    Standardized 12-output empty payload for load callbacks.

    Keeps the callback return footprint consistent when data is missing
    (e.g., no symbol provided or no daily_timeseries found). The status
    message is placed in the third position to align with the status-div title.

    Args:
        status (str): Message to display in the status div.

    Returns:
        tuple: (start_date, end_date, status, price, dividends, company_base,
                q_balance, a_balance, earnings, q_income, cashflow, insider)
                where all data entries are None.
    """

    return (None, None, status, *([None] * 9))


def guard_store(data, message: str, *, color: str = "red", icon: str = "alert-circle"):
    """
    One-liner guard: returns None when data is present; otherwise a small empty-state tuple.

    Returns a tuple (content, False) to align with callback output shapes.
    """
    if data is not None and (not hasattr(data, "__len__") or len(data) > 0):
        return None
    return (
        dmc.Group(
            [
                dmc.ThemeIcon(icon, color=color, variant="light", size=32),
                dmc.Text(message, c=color),
            ],
            gap="sm",
            align="center",
        ),
        False,
    )


def filter_by_date(
    df: pd.DataFrame,
    start_date,
    end_date,
    date_col: str | None,
) -> pd.DataFrame:
    """Safely filter a DataFrame by an inclusive date range.

    - No-ops when ``df`` is empty, dates are missing, or the column is absent.
    - Parses the date column to datetime if needed (non-destructive to the caller's frame).

    Args:
        df (pd.DataFrame): Source DataFrame.
        start_date: Start date (string/datetime) or falsy to skip filtering.
        end_date: End date (string/datetime) or falsy to skip filtering.
        date_col (str | None): Column to use for filtering.

    Returns:
        pd.DataFrame: Filtered frame (or original if filtering is skipped).
    """

    if df is None or df.empty:
        return df
    if not date_col or date_col not in df.columns:
        return df
    if not start_date or not end_date:
        return df

    filtered = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(filtered[date_col]):
        filtered[date_col] = pd.to_datetime(filtered[date_col])

    mask = (filtered[date_col] >= pd.to_datetime(start_date)) & (filtered[date_col] <= pd.to_datetime(end_date))
    return filtered.loc[mask]


def records_to_df(
    records,
    *,
    table: str | None = None,
    parse_dates: list[str] | None = None,
    sort_by: str | None = None,
) -> pd.DataFrame:
    """Convert Dash store records (list of dicts) to a DataFrame with optional date parsing and sorting.

    Args:
        records: The list of dictionaries (or None) coming from a Dash store.
        table: Optional table name to infer date columns from ORM definitions.
        parse_dates: Optional list of column names to parse as datetimes (overrides inferred list if provided).
        sort_by: Optional column name to sort by after parsing (falls back to first date column if omitted).

    Returns:
        A pandas DataFrame (empty if input is None or empty).
    """

    if records is None or len(records) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    inferred_dates = parse_dates if parse_dates is not None else _DATE_COLUMNS_BY_TABLE.get(table, [])
    for col in inferred_dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    sort_col = sort_by or (inferred_dates[0] if inferred_dates else None)
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col)

    return df