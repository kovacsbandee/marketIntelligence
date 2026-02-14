"""Centralized ID and tab constants for Dash components.

Keep all string IDs in one place to avoid magic strings scattered across
layout and callbacks.
"""

class Ids:
    # Inputs / controls
    SYMBOL_INPUT = "symbol-input"
    LOAD_BUTTON = "load-btn"
    START_DATE_PICKER = "start-date-picker"
    END_DATE_PICKER = "end-date-picker"
    RANGE_PRESET = "range-preset"
    STATUS_DIV = "status-div"
    MAIN_TABS = "main-tabs"
    PRICE_OVERLAY_TOGGLE = "price-overlay-toggle"
    PRICE_CHART = "price-chart"

    # Indicator charts
    RSI_CHART = "rsi-chart"
    MACD_CHART = "macd-chart"
    STOCHASTIC_CHART = "stochastic-chart"
    OBV_CHART = "obv-chart"
    ADX_CHART = "adx-chart"

    # Quantitative analysis
    ANALYZE_BUTTON = "analyze-btn"
    ANALYSIS_CLICKED_DATE = "analysis-clicked-date"
    ANALYSIS_INTERVAL = "analysis-interval"
    ANALYSIS_RUN_ID = "analysis-run-id"
    AGGREGATE_SCORE_CARD = "aggregate-score-card"
    ANALYSIS_SCORES_CONTAINER = "analysis-scores-container"
    ANALYSIS_DATE_DISPLAY = "analysis-date-display"
    # Per-indicator score card containers (keyed by indicator name)
    SCORE_CARD_PREFIX = "score-card-"  # e.g. "score-card-rsi"

    # All chart IDs that should get a vertical marker line
    ALL_CHART_IDS = ["price-chart", "rsi-chart", "macd-chart", "stochastic-chart", "obv-chart", "adx-chart"]

    # Stores
    PRICE_STORE = "price-store"
    DIVIDENDS_STORE = "dividends-store"
    COMPANY_BASE_STORE = "company-base-store"
    Q_BALANCE_STORE = "q_balance-store"
    A_BALANCE_STORE = "a_balance-store"
    EARNINGS_STORE = "earnings-store"
    Q_INCOME_STORE = "q_income-store"
    CASHFLOW_STORE = "cashflow-store"
    INSIDER_STORE = "insider-transactions-store"
    ANALYSIS_STORE = "analysis-store"

    # Content + loading overlays
    COMPANY_BASE_CONTENT = "company-base-content"
    COMPANY_BASE_LOADING = "company-base-loading"

    PRICE_INDICATOR_CONTENT = "price-indicator-content"
    PRICE_INDICATOR_LOADING = "price-indicator-loading"

    EARNINGS_CONTENT = "earnings-content"
    EARNINGS_LOADING = "earnings-loading"

    INCOME_STATEMENT_CONTENT = "income-statement-content"
    INCOME_STATEMENT_LOADING = "income-statement-loading"

    BALANCE_SHEET_CONTENT = "balance-sheet-content"
    BALANCE_SHEET_LOADING = "balance-sheet-loading"

    CASH_FLOW_CONTENT = "cash-flow-content"
    CASH_FLOW_LOADING = "cash-flow-loading"

    INSIDER_CONTENT = "insider-transactions-content"
    INSIDER_LOADING = "insider-transactions-loading"


class Tabs:
    COMPANY_BASE = "company-base"
    PRICE_INDICATOR = "price-indicator"
    EARNINGS = "earnings"
    INCOME_STATEMENT = "income-statement"
    BALANCE_SHEET = "balance-sheet"
    CASH_FLOW = "cash-flow"
    INSIDER_TRANSACTIONS = "insider-transactions"

    ITEMS = [
        ("Company Base", COMPANY_BASE),
        ("Price Indicator", PRICE_INDICATOR),
        ("Earnings", EARNINGS),
        ("Income Statement", INCOME_STATEMENT),
        ("Balance Sheet", BALANCE_SHEET),
        ("Cash Flow", CASH_FLOW),
        ("Insider Transactions", INSIDER_TRANSACTIONS),
    ]
