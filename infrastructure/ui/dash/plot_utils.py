import os
import json
import pandas as pd

DEFAULT_PLOTLY_WIDTH = 1800
DEFAULT_PLOTLY_HEIGHT = 700


def _load_column_description_json():
    """
    Loads the alpha_vantage_column_description_hun.json config as a Python dict.
    Returns:
        dict: The loaded JSON content.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "configs",
        "alpha_vantage_column_description_hun.json"
    )
    with open(config_path, "r") as f:
        return json.load(f)

def _get_column_descriptions(table_name: str = None):
    """
    Loads column descriptions for a given table from the Hungarian JSON config.
    Args:
        table_name (str): The table name section in the JSON config.
    Returns:
        dict: Mapping of column names to Hungarian descriptions.
    """
    config = _load_column_description_json()
    desc = {}
    for entry in config.get(table_name, []):
        desc.update(entry)
    return desc

def _find_row_by_date(balance_df, date):
    # Accepts date as string or datetime, returns the row for the closest matching date
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)
    # Find the row with the exact date, or the closest previous date if not found
    idx = balance_df["fiscal_date_ending"].sub(date).abs().idxmin()
    return balance_df.loc[idx]


def _flatten_group(group):
    """Recursively flatten a nested group dict into a list of column names."""
    if isinstance(group, dict):
        cols = []
        for v in group.values():
            cols.extend(_flatten_group(v))
        return cols
    elif isinstance(group, list):
        cols = []
        for v in group:
            cols.extend(_flatten_group(v))
        return cols
    else:
        return [group]
    

def _auto_load_table_descriptions(df, table_name=None):
    """
    Generic loader for column descriptions for any ORM table.
    If table_name is None, tries to infer from DataFrame columns.
    Returns a dict mapping column names to descriptions.
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "configs",
        "alpha_vantage_column_description_hun.json"
    )
    with open(config_path, "r") as f:
        config = json.load(f)

    # Try to infer table_name if not given
    if table_name is None:
        # Heuristic for balance sheet
        if "fiscal_date_ending" in df.columns:
            unique_dates = pd.to_datetime(df["fiscal_date_ending"].dropna().unique())
            if len(unique_dates) > 12:
                table_name = "balance_sheet_quarterly"
            else:
                table_name = "balance_sheet_annual"
        elif "latest_quarter" in df.columns and "name" in df.columns:
            table_name = "company_fundamentals"
        elif "date" in df.columns and "open" in df.columns and "close" in df.columns:
            table_name = "daily_timeseries"
        elif "ex_dividend_date" in df.columns and "amount" in df.columns:
            table_name = "dividends"
        # Add more heuristics as needed
        else:
            table_name = None

    # Fallback: try to use the first matching section
    section = table_name if table_name in config else None
    if not section:
        # Try to find a section that matches most columns
        max_overlap = 0
        for sec, entries in config.items():
            if not isinstance(entries, list):
                continue
            cols = set()
            for entry in entries:
                cols.update(entry.keys())
            overlap = len(set(df.columns) & cols)
            if overlap > max_overlap:
                max_overlap = overlap
                section = sec

    descriptions = {}
    if section and section in config:
        for entry in config.get(section, []):
            for k, v in entry.items():
                descriptions[k] = v
    return descriptions
