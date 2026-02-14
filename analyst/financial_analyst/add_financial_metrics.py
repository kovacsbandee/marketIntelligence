"""
add_financial_metrics.py
------------------------

Pure calculation functions that enrich quarterly financial statement
DataFrames with derived metrics (margins, ratios, growth rates, etc.).

Each function follows the same contract as the functions in
``analyst/quantitative_analyst/add_indicators_to_price_data.py``:

* Accept a ``pd.DataFrame`` (and optional parameters).
* Return a **new** DataFrame with the original columns *plus* newly
  computed columns.
* Never mutate the input DataFrame.

These functions are called by ``Symbol.add_all_financial_metrics()``
at init time so that the derived columns are available to
``FinancialAnalyst`` and to the Dash UI without recomputation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division; returns NaN where *denominator* is 0 or NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce *columns* to numeric in-place (on the copy), skipping missing cols."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Income-statement metrics  (source: income_statement_quarterly)
# ---------------------------------------------------------------------------

def calculate_profitability_margins(data: pd.DataFrame) -> pd.DataFrame:
    """Add gross, operating and net profit margin columns.

    New columns
    -----------
    * ``gross_margin``      – gross_profit / total_revenue
    * ``operating_margin``  – operating_income / total_revenue
    * ``net_margin``        – net_income / total_revenue
    * ``ebitda_margin``     – ebitda / total_revenue

    Parameters
    ----------
    data : pd.DataFrame
        Income-statement quarterly DataFrame.  Must contain at least
        ``total_revenue``; other columns are used when present.

    Returns
    -------
    pd.DataFrame
        Copy of *data* with the new columns appended.
    """
    df = data.copy()
    df = _ensure_numeric(df, [
        "gross_profit", "total_revenue", "operating_income",
        "net_income", "ebitda",
    ])

    if "total_revenue" not in df.columns:
        return df

    revenue = df["total_revenue"]

    if "gross_profit" in df.columns:
        df["gross_margin"] = _safe_divide(df["gross_profit"], revenue)

    if "operating_income" in df.columns:
        df["operating_margin"] = _safe_divide(df["operating_income"], revenue)

    if "net_income" in df.columns:
        df["net_margin"] = _safe_divide(df["net_income"], revenue)

    if "ebitda" in df.columns:
        df["ebitda_margin"] = _safe_divide(df["ebitda"], revenue)

    return df


def calculate_revenue_growth(data: pd.DataFrame) -> pd.DataFrame:
    """Add quarter-over-quarter and year-over-year revenue growth columns.

    New columns
    -----------
    * ``revenue_qoq_growth`` – pct change vs previous quarter
    * ``revenue_yoy_growth`` – pct change vs same quarter last year (4 periods)

    Parameters
    ----------
    data : pd.DataFrame
        Income-statement quarterly DataFrame sorted by date ascending.

    Returns
    -------
    pd.DataFrame
    """
    df = data.copy()
    df = _ensure_numeric(df, ["total_revenue"])

    if "total_revenue" not in df.columns:
        return df

    df["revenue_qoq_growth"] = df["total_revenue"].pct_change()

    if len(df) >= 4:
        df["revenue_yoy_growth"] = df["total_revenue"].pct_change(periods=4)

    return df


# ---------------------------------------------------------------------------
# Balance-sheet metrics  (source: balance_sheet_quarterly)
# ---------------------------------------------------------------------------

def calculate_liquidity_ratios(data: pd.DataFrame) -> pd.DataFrame:
    """Add current ratio, quick ratio and cash ratio columns.

    New columns
    -----------
    * ``current_ratio`` – total_current_assets / total_current_liabilities
    * ``quick_ratio``   – (total_current_assets − inventory) / total_current_liabilities
    * ``cash_ratio``    – cash_and_cash_equivalents / total_current_liabilities

    Parameters
    ----------
    data : pd.DataFrame
        Balance-sheet quarterly DataFrame.

    Returns
    -------
    pd.DataFrame
    """
    df = data.copy()
    df = _ensure_numeric(df, [
        "total_current_assets", "total_current_liabilities",
        "cash_and_cash_equivalents", "inventory",
    ])

    denom = df.get("total_current_liabilities")
    if denom is None:
        return df

    if "total_current_assets" in df.columns:
        df["current_ratio"] = _safe_divide(df["total_current_assets"], denom)

    if "total_current_assets" in df.columns and "inventory" in df.columns:
        quick_assets = df["total_current_assets"] - df["inventory"].fillna(0)
        df["quick_ratio"] = _safe_divide(quick_assets, denom)

    if "cash_and_cash_equivalents" in df.columns:
        df["cash_ratio"] = _safe_divide(df["cash_and_cash_equivalents"], denom)

    return df


def calculate_leverage_ratios(data: pd.DataFrame) -> pd.DataFrame:
    """Add debt-to-equity and debt-to-assets ratio columns.

    New columns
    -----------
    * ``debt_to_equity`` – total_liabilities / total_shareholder_equity
    * ``debt_to_assets`` – total_liabilities / total_assets
    * ``equity_ratio``   – total_shareholder_equity / total_assets

    Parameters
    ----------
    data : pd.DataFrame
        Balance-sheet quarterly DataFrame.

    Returns
    -------
    pd.DataFrame
    """
    df = data.copy()
    df = _ensure_numeric(df, [
        "total_liabilities", "total_assets", "total_shareholder_equity",
    ])

    if "total_liabilities" in df.columns and "total_shareholder_equity" in df.columns:
        df["debt_to_equity"] = _safe_divide(
            df["total_liabilities"], df["total_shareholder_equity"]
        )

    if "total_liabilities" in df.columns and "total_assets" in df.columns:
        df["debt_to_assets"] = _safe_divide(
            df["total_liabilities"], df["total_assets"]
        )

    if "total_shareholder_equity" in df.columns and "total_assets" in df.columns:
        df["equity_ratio"] = _safe_divide(
            df["total_shareholder_equity"], df["total_assets"]
        )

    return df


# ---------------------------------------------------------------------------
# Cash-flow metrics  (source: cash_flow_quarterly)
# ---------------------------------------------------------------------------

def calculate_cashflow_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Add free-cash-flow and cash-flow quality columns.

    New columns
    -----------
    * ``free_cash_flow``    – operating_cashflow + capital_expenditures
                              (CapEx is typically negative, so this is OpCF − |CapEx|)
    * ``cf_to_net_income``  – operating_cashflow / net_income
    * ``capex_to_revenue``  – capital_expenditures / net_income  (intensity)

    Parameters
    ----------
    data : pd.DataFrame
        Cash-flow quarterly DataFrame.

    Returns
    -------
    pd.DataFrame
    """
    df = data.copy()
    df = _ensure_numeric(df, [
        "operating_cashflow", "capital_expenditures", "net_income",
    ])

    if "operating_cashflow" in df.columns and "capital_expenditures" in df.columns:
        df["free_cash_flow"] = df["operating_cashflow"] + df["capital_expenditures"]

    if "operating_cashflow" in df.columns and "net_income" in df.columns:
        df["cf_to_net_income"] = _safe_divide(
            df["operating_cashflow"], df["net_income"]
        )

    if "capital_expenditures" in df.columns and "net_income" in df.columns:
        df["capex_to_net_income"] = _safe_divide(
            df["capital_expenditures"].abs(), df["net_income"]
        )

    return df


# ---------------------------------------------------------------------------
# Earnings metrics  (source: earnings_quarterly)
# ---------------------------------------------------------------------------

def calculate_earnings_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Add derived earnings columns.

    New columns
    -----------
    * ``eps_qoq_growth``     – quarter-over-quarter growth of reported_eps
    * ``earnings_beat``      – 1 if reported_eps > estimated_eps, 0 otherwise
    * ``surprise_abs``       – absolute value of the surprise column

    Parameters
    ----------
    data : pd.DataFrame
        Earnings quarterly DataFrame sorted by date ascending.

    Returns
    -------
    pd.DataFrame
    """
    df = data.copy()
    df = _ensure_numeric(df, [
        "reported_eps", "estimated_eps", "surprise", "surprise_percentage",
    ])

    if "reported_eps" in df.columns:
        df["eps_qoq_growth"] = df["reported_eps"].pct_change()

    if "reported_eps" in df.columns and "estimated_eps" in df.columns:
        df["earnings_beat"] = (df["reported_eps"] > df["estimated_eps"]).astype(int)

    if "surprise" in df.columns:
        df["surprise_abs"] = df["surprise"].abs()

    return df
