import math
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from infrastructure.ui.dash.plot_utils import DEFAULT_PLOTLY_WIDTH, DEFAULT_PLOTLY_HEIGHT, _get_column_descriptions 


def plot_quarterly_revenue_net_income_vs_stock_price(symbol: str, income_df: pd.DataFrame, price_df: pd.DataFrame) -> go.Figure:
    """
    Plots quarterly total revenue, net income, and stock price for a given stock symbol.

    This function generates a Plotly figure with two y-axes, visualizing the relationship between a company's quarterly
    financial performance (total revenue and net income) and its stock price over time. It takes as input the stock symbol,
    a DataFrame containing quarterly income statement data, and a DataFrame containing historical stock price data.

    The function performs the following steps:
    - Validates the input DataFrames for required columns and non-emptiness.
    - Filters both DataFrames for the specified stock symbol (case-insensitive).
    - Sorts the income statement data by fiscal quarter end date.
    - Converts relevant columns to appropriate data types (datetime for dates, numeric for financials).
    - For each fiscal quarter, finds the corresponding or next available stock closing price.
    - Plots three traces:
        1. Total revenue (primary y-axis, blue line with circle markers)
        2. Net income (primary y-axis, green line with diamond markers)
        3. Stock closing price (secondary y-axis, orange dashed line with square markers)
    - Customizes axis titles, legend, layout, and styling for clarity and aesthetics.

    Parameters
    ----------
    symbol : str
        The stock ticker symbol (e.g., "AAPL") for which to plot the data.
    income_df : pd.DataFrame
        DataFrame containing quarterly income statement data. Must include columns:
        - "fiscal_date_ending": End date of the fiscal quarter (string or datetime)
        - "total_revenue": Total revenue for the quarter (numeric)
        - "net_income": Net income for the quarter (numeric)
        - "symbol": (optional) Stock symbol for filtering
    price_df : pd.DataFrame
        DataFrame containing historical stock price data. Must include columns:
        - "date": Date of the stock price (string or datetime)
        - "close": Closing price of the stock (numeric)
        - "symbol": (optional) Stock symbol for filtering

    Returns
    -------
    go.Figure
        A Plotly Figure object containing the multi-axis line plot. If input data is invalid or missing required columns,
        returns an empty Figure.

    Notes
    -----
    - The function expects that the income statement and price data are at least quarterly and daily frequency, respectively.
    - If multiple symbols are present in the DataFrames, only data matching the provided symbol will be used.
    - If no matching data is found after filtering, or required columns are missing, an empty figure is returned.
    - The function uses a helper `_get_column_descriptions` to provide human-readable axis and legend labels.

    Examples
    --------
    >>> fig = plot_quarterly_revenue_net_income_vs_stock_price(
    ...     "AAPL", income_df=income_data, price_df=price_data
    ... )
    >>> fig.show()
    """
    if income_df is None or price_df is None or len(income_df) == 0 or len(price_df) == 0:
        print("[DEBUG] Empty income_df or price_df.")
        return go.Figure()
    df_income = income_df.copy()
    df_price = price_df.copy()
    if "symbol" in df_income.columns:
        df_income = df_income[df_income["symbol"].str.upper() == symbol.upper()]
    if "symbol" in df_price.columns:
        df_price = df_price[df_price["symbol"].str.upper() == symbol.upper()]
    if df_income.empty or df_price.empty:
        return go.Figure()
    if "fiscal_date_ending" not in df_income.columns or "total_revenue" not in df_income.columns or "net_income" not in df_income.columns:
        return go.Figure()
    if "date" not in df_price.columns or "close" not in df_price.columns:
        return go.Figure()
    df_income = df_income.sort_values("fiscal_date_ending")
    x = pd.to_datetime(df_income["fiscal_date_ending"])
    revenue = pd.to_numeric(df_income["total_revenue"], errors="coerce")
    net_income = pd.to_numeric(df_income["net_income"], errors="coerce")
    df_price["date"] = pd.to_datetime(df_price["date"])
    close_prices = []
    for dt in x:
        price_row = df_price[df_price["date"] >= dt]
        if not price_row.empty:
            close_prices.append(price_row.iloc[0]["close"])
        else:
            close_prices.append(df_price.iloc[-1]["close"])
    descriptions = _get_column_descriptions("income_statement_quarterly")
    label_overrides = {
        "total_revenue": "Total revenue (sales)",
        "net_income": "Net income (profit after taxes)",
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=revenue,
        mode="lines+markers",
        name=label_overrides.get("total_revenue", descriptions.get("total_revenue", "Total Revenue")),
        line=dict(color="#228be6", width=3),
        marker=dict(symbol="circle", size=8, color="#228be6"),
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=net_income,
        mode="lines+markers",
        name=label_overrides.get("net_income", descriptions.get("net_income", "Net Income")),
        line=dict(color="#40c057", width=3),
        marker=dict(symbol="diamond", size=8, color="#40c057"),
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=close_prices,
        mode="lines+markers",
        name="Stock price (close after quarter end)",
        line=dict(color="#fab005", width=3, dash="dash"),
        marker=dict(symbol="square", size=8, color="#fab005"),
        yaxis="y2"
    ))
    fig.update_layout(
        title=f"{symbol.upper()} Revenue, Net Income, and Stock Price by Quarter",
        xaxis_title="Fiscal quarter end date",
        yaxis=dict(
            title="Revenue and net income (USD)",
            showgrid=True,
            zeroline=True
        ),
        yaxis2=dict(
            title="Stock price (USD)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend_title="Series",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14)
    )
    return fig


def plot_quarterly_profit_margins(symbol: str, income_df: pd.DataFrame) -> go.Figure:
    """
    Plots gross, operating and net profit margins over time to assess profitability trends.

    The function filters `income_df` for the chosen `symbol` and computes three margins for each quarter:

    * **Gross profit margin** = `gross_profit` divided by `total_revenue`.
    * **Operating profit margin** = `operating_income` divided by `total_revenue`.
    * **Net profit margin** = `net_income` divided by `total_revenue`.

    It then builds a line chart with `fiscal_date_ending` on the x‑axis and the calculated margins (expressed as
    percentages) on the y‑axis.  Investors and analysts watch these margins closely because they indicate how
    efficiently a company turns sales into profits.  A higher gross profit margin suggests efficient operations and
    provides a basis for comparison with peers:contentReference[oaicite:2]{index=2}; operating and net profit margins offer
    insight into how much profit is generated after operating expenses, taxes and interest:contentReference[oaicite:3]{index=3}.
    By examining margin trends quarter over quarter, investors can identify improvements or deterioration in
    profitability and evaluate whether the company’s fundamentals justify changes in its stock price:contentReference[oaicite:4]{index=4}.
    """
    # Validate input DataFrame
    if income_df is None or len(income_df) == 0:
        return go.Figure()
    df = income_df.copy()
    # Filter by symbol if column exists
    if "symbol" in df.columns:
        df = df[df["symbol"].str.upper() == symbol.upper()]
    if df.empty:
        return go.Figure()
    # Required columns
    required_cols = ["fiscal_date_ending", "total_revenue", "gross_profit", "operating_income", "net_income"]
    for col in required_cols:
        if col not in df.columns:
            return go.Figure()
    # Sort by fiscal_date_ending
    df = df.sort_values("fiscal_date_ending")
    # Convert columns to numeric
    df["total_revenue"] = pd.to_numeric(df["total_revenue"], errors="coerce")
    df["gross_profit"] = pd.to_numeric(df["gross_profit"], errors="coerce")
    df["operating_income"] = pd.to_numeric(df["operating_income"], errors="coerce")
    df["net_income"] = pd.to_numeric(df["net_income"], errors="coerce")
    df = df.dropna(subset=["fiscal_date_ending", "total_revenue"])
    # Avoid division by zero
    df = df[df["total_revenue"] != 0]
    if df.empty:
        return go.Figure()
    # Calculate margins
    gross_margin = df["gross_profit"] / df["total_revenue"] * 100
    operating_margin = df["operating_income"] / df["total_revenue"] * 100
    net_margin = df["net_income"] / df["total_revenue"] * 100
    x = pd.to_datetime(df["fiscal_date_ending"])
    # Get column descriptions if available
    descriptions = _get_column_descriptions("income_statement_quarterly") if "_get_column_descriptions" in globals() else {}
    label_overrides = {
        "gross_profit": "Gross margin (gross profit / revenue)",
        "operating_income": "Operating margin (operating income / revenue)",
        "net_income": "Net margin (net income / revenue)",
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=gross_margin,
        mode="lines+markers",
        name=label_overrides.get("gross_profit", descriptions.get("gross_profit", "Gross Profit Margin")),
        line=dict(color="#228be6", width=3),
        marker=dict(symbol="circle", size=8, color="#228be6"),
        hovertemplate="%{y:.2f}%<br>%{x|%Y-%m-%d}<extra>Gross Margin</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=operating_margin,
        mode="lines+markers",
        name=label_overrides.get("operating_income", descriptions.get("operating_income", "Operating Profit Margin")),
        line=dict(color="#40c057", width=3),
        marker=dict(symbol="diamond", size=8, color="#40c057"),
        hovertemplate="%{y:.2f}%<br>%{x|%Y-%m-%d}<extra>Operating Margin</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=net_margin,
        mode="lines+markers",
        name=label_overrides.get("net_income", descriptions.get("net_income", "Net Profit Margin")),
        line=dict(color="#fa5252", width=3),
        marker=dict(symbol="square", size=8, color="#fa5252"),
        hovertemplate="%{y:.2f}%<br>%{x|%Y-%m-%d}<extra>Net Margin</extra>"
    ))
    fig.update_layout(
        title=f"{symbol.upper()} Profit Margins by Quarter",
        xaxis_title="Fiscal quarter end date",
        yaxis_title="Margin (% of revenue)",
        legend_title="Margin type",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH if 'DEFAULT_PLOTLY_WIDTH' in globals() else 900,
        height=DEFAULT_PLOTLY_HEIGHT if 'DEFAULT_PLOTLY_HEIGHT' in globals() else 500,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=14)
    )
    fig.update_yaxes(ticksuffix="%", zeroline=True)
    return fig

def plot_expense_breakdown_vs_revenue(symbol: str, income_df: pd.DataFrame) -> go.Figure:
    """
    Creates small multiple bar charts to visualize how major operating expenses compare with total revenue each quarter.
    """
    # Validate input DataFrame
    if income_df is None or len(income_df) == 0:
        return go.Figure()
    df = income_df.copy()
    # Filter by symbol if column exists
    if "symbol" in df.columns:
        df = df[df["symbol"].str.upper() == symbol.upper()]
    if df.empty:
        return go.Figure()
    # Required columns
    base_cols = ["fiscal_date_ending", "total_revenue"]
    expense_cols = [
        "cost_of_revenue",
        "cost_of_goods_and_services_sold",
        "selling_general_and_administrative",
        "research_and_development",
        "operating_expenses"
    ]
    for col in base_cols:
        if col not in df.columns:
            return go.Figure()
    # Only keep expense columns that exist
    expense_cols = [col for col in expense_cols if col in df.columns]
    if not expense_cols:
        return go.Figure()
    # Sort by fiscal_date_ending
    df = df.sort_values("fiscal_date_ending")
    x = pd.to_datetime(df["fiscal_date_ending"])
    total_revenue = pd.to_numeric(df["total_revenue"], errors="coerce")
    # Prepare subplots: one for each expense + one for total revenue
    n = len(expense_cols)
    ncols = 2
    nrows = math.ceil(n / ncols)
    expense_titles = {
        "cost_of_revenue": "Cost of revenue (% of revenue)",
        "cost_of_goods_and_services_sold": "Cost of goods & services sold (% of revenue)",
        "selling_general_and_administrative": "Selling, general & admin (% of revenue)",
        "research_and_development": "Research & development (% of revenue)",
        "operating_expenses": "Operating expenses (% of revenue)",
    }
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[expense_titles.get(col, col.replace('_', ' ').title()) for col in expense_cols],
        shared_xaxes=False,
        vertical_spacing=0.13,
        horizontal_spacing=0.13
    )
    # Get column descriptions if available
    descriptions = _get_column_descriptions("income_statement_quarterly") if "_get_column_descriptions" in globals() else {}
    for i, col in enumerate(expense_cols):
        row = i // ncols + 1
        colnum = i % ncols + 1
        expense = pd.to_numeric(df[col], errors="coerce")
        pct = (expense / total_revenue * 100).replace([float('inf'), -float('inf')], float('nan'))
        # Bar for expense as % of revenue
        fig.add_trace(
            go.Bar(
                x=x,
                y=pct,
                name=expense_titles.get(col, descriptions.get(col, col.replace('_', ' ').title())),
                marker_color="#fa5252",
                hovertemplate="%{y:.2f}%<br>%{x|%Y-%m-%d}<extra>Expense % of Revenue</extra>"
            ),
            row=row, col=colnum
        )
        # Line for total revenue (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=total_revenue,
                mode="lines+markers",
                name="Total revenue (USD)",
                line=dict(color="#228be6", width=2, dash="dash"),
                marker=dict(symbol="circle", size=6, color="#228be6"),
                yaxis=f"y{i+1}2",
                hovertemplate="Revenue: %{y:,.0f}<br>%{x|%Y-%m-%d}<extra>Total Revenue</extra>"
            ),
            row=row, col=colnum
        )
        # Add secondary y-axis for revenue
        fig.update_yaxes(
            title_text="Expense as % of revenue",
            row=row, col=colnum,
            ticksuffix="%"
        )
        fig.update_yaxes(
            title_text="Total revenue (USD)",
            row=row, col=colnum,
            secondary_y=True
        )
    fig.update_layout(
        title=f"{symbol.upper()} Expense Share of Revenue by Quarter",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH if 'DEFAULT_PLOTLY_WIDTH' in globals() else 1200,
        height=400 * nrows,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=14),
        showlegend=False
    )
    fig.update_annotations(font_size=16)
    return fig

def plot_income_statement_waterfall(symbol: str, income_df: pd.DataFrame) -> go.Figure:
    """
    Builds a waterfall chart to illustrate the progression from total revenue to net income in a single quarter.
    """
    # Validate input DataFrame
    if income_df is None or len(income_df) == 0:
        return go.Figure()
    df = income_df.copy()
    # Filter by symbol if column exists
    if "symbol" in df.columns:
        df = df[df["symbol"].str.upper() == symbol.upper()]
    if df.empty:
        return go.Figure()
    # Required columns
    base_cols = ["fiscal_date_ending", "total_revenue", "net_income"]
    step_cols = [
        "cost_of_revenue",
        "operating_expenses",
        "investment_income_net",
        "net_interest_income",
        "other_non_operating_income",
        "income_tax_expense",
        "interest_and_debt_expense"
    ]
    for col in base_cols:
        if col not in df.columns:
            return go.Figure()
    # Use the most recent quarter
    df = df.sort_values("fiscal_date_ending")
    row = df.iloc[-1]
    # Prepare waterfall steps
    steps = []
    labels = []
    values = []
    descriptions = _get_column_descriptions("income_statement_quarterly") if "_get_column_descriptions" in globals() else {}
    # Start with total revenue
    label_overrides = {
        "total_revenue": "Total revenue (sales)",
        "cost_of_revenue": "Cost of revenue (expense)",
        "operating_expenses": "Operating expenses (expense)",
        "investment_income_net": "Investment income (adds to profit)",
        "net_interest_income": "Net interest income (adds to profit)",
        "other_non_operating_income": "Other non‑operating income (adds to profit)",
        "income_tax_expense": "Income tax (reduces profit)",
        "interest_and_debt_expense": "Interest & debt expense (reduces profit)",
        "net_income": "Net income (profit after taxes)",
    }
    labels.append(label_overrides.get("total_revenue", descriptions.get("total_revenue", "Total Revenue")))
    values.append(row["total_revenue"])
    steps.append("absolute")
    # Add/subtract each step
    for col in step_cols:
        if col in row and not pd.isnull(row[col]):
            val = row[col]
            # For income items, add; for expenses, subtract
            if col in ["investment_income_net", "net_interest_income", "other_non_operating_income"]:
                labels.append(label_overrides.get(col, descriptions.get(col, col.replace('_', ' ').title())))
                values.append(val)
                steps.append("relative")
            else:
                labels.append(label_overrides.get(col, descriptions.get(col, col.replace('_', ' ').title())))
                values.append(-val)
                steps.append("relative")
    # End with net income
    labels.append(label_overrides.get("net_income", descriptions.get("net_income", "Net Income")))
    values.append(row["net_income"])
    steps.append("total")
    # Build waterfall chart
    fig = go.Figure(go.Waterfall(
        name = "Income Statement",
        orientation = "v",
        measure = steps,
        x = labels,
        text = [f"{v:,.0f}" if not pd.isnull(v) else "N/A" for v in values],
        y = values,
        connector = {"line": {"color": "rgb(63, 63, 63)"}},
        decreasing = {"marker": {"color": "#fa5252"}},
        increasing = {"marker": {"color": "#40c057"}},
        totals = {"marker": {"color": "#228be6"}},
        hovertemplate = "%{label}: %{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        title=f"{symbol.upper()} How Revenue Becomes Net Income (Latest Quarter)",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH if 'DEFAULT_PLOTLY_WIDTH' in globals() else 1200,
        height=600,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=14)
    )
    return fig

def plot_operating_profit_ebit_ebitda_trends(symbol: str, income_df: pd.DataFrame) -> go.Figure:
    """
    Generates a line chart comparing `operating_income`, `ebit` and `ebitda` across fiscal quarters.
    """
    # Validate input DataFrame
    if income_df is None or len(income_df) == 0:
        return go.Figure()
    df = income_df.copy()
    # Filter by symbol if column exists
    if "symbol" in df.columns:
        df = df[df["symbol"].str.upper() == symbol.upper()]
    if df.empty:
        return go.Figure()
    # Required columns
    required_cols = ["fiscal_date_ending", "operating_income", "ebit", "ebitda"]
    for col in required_cols:
        if col not in df.columns:
            return go.Figure()
    # Sort by fiscal_date_ending
    df = df.sort_values("fiscal_date_ending")
    x = pd.to_datetime(df["fiscal_date_ending"])
    operating_income = pd.to_numeric(df["operating_income"], errors="coerce")
    ebit = pd.to_numeric(df["ebit"], errors="coerce")
    ebitda = pd.to_numeric(df["ebitda"], errors="coerce")
    # Get column descriptions if available
    descriptions = _get_column_descriptions("income_statement_quarterly") if "_get_column_descriptions" in globals() else {}
    label_overrides = {
        "operating_income": "Operating income (profit from core operations)",
        "ebit": "EBIT (earnings before interest & taxes)",
        "ebitda": "EBITDA (before interest, taxes, depreciation & amortization)",
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=operating_income,
        mode="lines+markers",
        name=label_overrides.get("operating_income", descriptions.get("operating_income", "Operating Income")),
        line=dict(color="#228be6", width=3),
        marker=dict(symbol="circle", size=8, color="#228be6"),
        hovertemplate="%{y:,.0f}<br>%{x|%Y-%m-%d}<extra>Operating Income</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=ebit,
        mode="lines+markers",
        name=label_overrides.get("ebit", descriptions.get("ebit", "EBIT")),
        line=dict(color="#fab005", width=3),
        marker=dict(symbol="diamond", size=8, color="#fab005"),
        hovertemplate="%{y:,.0f}<br>%{x|%Y-%m-%d}<extra>EBIT</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=ebitda,
        mode="lines+markers",
        name=label_overrides.get("ebitda", descriptions.get("ebitda", "EBITDA")),
        line=dict(color="#40c057", width=3),
        marker=dict(symbol="square", size=8, color="#40c057"),
        hovertemplate="%{y:,.0f}<br>%{x|%Y-%m-%d}<extra>EBITDA</extra>"
    ))
    fig.update_layout(
        title=f"{symbol.upper()} Operating Income, EBIT, and EBITDA by Quarter",
        xaxis_title="Fiscal quarter end date",
        yaxis_title="Amount (USD)",
        legend_title="Profit measure",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH if 'DEFAULT_PLOTLY_WIDTH' in globals() else 900,
        height=DEFAULT_PLOTLY_HEIGHT if 'DEFAULT_PLOTLY_HEIGHT' in globals() else 500,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=14)
    )
    return fig

def plot_expense_growth_scatter(symbol: str, income_df: pd.DataFrame) -> go.Figure:
    """
    Constructs a bubble scatter plot to analyze quarter‑over‑quarter changes in expense categories relative to revenue.

    For each pair of consecutive quarters in `income_df` (filtered by `symbol`), the function calculates the
    percentage change in major expense categories — such as `selling_general_and_administrative`,
    `research_and_development`, `operating_expenses`, and `cost_of_goods_and_services_sold` — as well as the
    percentage change in `total_revenue`.  Each expense item is represented as a point whose x‑coordinate is its
    relative growth rate, y‑coordinate is its absolute growth (difference in dollar amount) and bubble size
    corresponds to its proportion of total expenses in the prior quarter.  A reference line represents the revenue
    growth rate for comparison.  Bubble charts excel at showing which items drive changes from period to period
    and whether expenses are growing faster than revenue:contentReference[oaicite:12]{index=12}.  This visualization helps
    investors identify cost categories that may erode profitability or signal strategic investment and to evaluate
    whether expense growth is sustainable relative to revenue:contentReference[oaicite:13]{index=13}.
    """
    # Validate input DataFrame
    if income_df is None or len(income_df) == 0:
        return go.Figure()
    df = income_df.copy()
    # Filter by symbol if column exists
    if "symbol" in df.columns:
        df = df[df["symbol"].str.upper() == symbol.upper()]
    if df.empty:
        return go.Figure()
    # Required columns
    base_cols = ["fiscal_date_ending", "total_revenue"]
    expense_cols = [
        "selling_general_and_administrative",
        "research_and_development",
        "operating_expenses",
        "cost_of_goods_and_services_sold"
    ]
    for col in base_cols:
        if col not in df.columns:
            return go.Figure()
    # Only keep expense columns that exist
    expense_cols = [col for col in expense_cols if col in df.columns]
    if not expense_cols:
        return go.Figure()
    # Sort by fiscal_date_ending
    df = df.sort_values("fiscal_date_ending")
    df = df.reset_index(drop=True)
    # Convert columns to numeric
    df["total_revenue"] = pd.to_numeric(df["total_revenue"], errors="coerce")
    for col in expense_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop rows with missing fiscal_date_ending or total_revenue
    df = df.dropna(subset=["fiscal_date_ending", "total_revenue"])
    if len(df) < 2:
        return go.Figure()
    # Calculate total expenses for bubble size
    df["total_expenses"] = df[expense_cols].sum(axis=1)
    # Prepare data for scatter plot
    x_growth = []  # relative growth rate (pct)
    y_growth = []  # absolute growth (delta)
    bubble_size = []  # prior quarter's proportion of total expenses
    labels = []
    quarter_labels = []
    expense_labels = []
    revenue_growth = []  # for reference line
    descriptions = _get_column_descriptions("income_statement_quarterly") if "_get_column_descriptions" in globals() else {}
    expense_labels_map = {
        "selling_general_and_administrative": "Selling, general & admin",
        "research_and_development": "Research & development",
        "operating_expenses": "Operating expenses",
        "cost_of_goods_and_services_sold": "Cost of goods & services sold",
    }
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        prev_total_expenses = prev["total_expenses"] if prev["total_expenses"] != 0 else np.nan
        # Revenue growth for this period
        rev_growth = (curr["total_revenue"] - prev["total_revenue"]) / prev["total_revenue"] if prev["total_revenue"] != 0 else np.nan
        revenue_growth.append(rev_growth * 100)
        for col in expense_cols:
            prev_val = prev[col]
            curr_val = curr[col]
            # Relative growth rate (pct)
            rel_growth = (curr_val - prev_val) / prev_val * 100 if prev_val != 0 else np.nan
            # Absolute growth (delta)
            abs_growth = curr_val - prev_val
            # Bubble size: prior quarter's proportion of total expenses
            size = prev_val / prev_total_expenses * 100 if prev_total_expenses and not np.isnan(prev_total_expenses) else np.nan
            x_growth.append(rel_growth)
            y_growth.append(abs_growth)
            bubble_size.append(size)
            labels.append(expense_labels_map.get(col, descriptions.get(col, col.replace('_', ' ').title())))
            quarter_labels.append(str(curr["fiscal_date_ending"]))
            expense_labels.append(col)
    # Build scatter plot
    fig = go.Figure()
    # Add bubbles for each expense item
    fig.add_trace(go.Scatter(
        x=x_growth,
        y=y_growth,
        mode="markers",
        marker=dict(
            size=[max(8, s if not np.isnan(s) else 8) for s in bubble_size],
            sizemode="area",
            sizeref=2.*max([s for s in bubble_size if not np.isnan(s)] + [8])/60.0 if bubble_size else 1,
            sizemin=8,
            color="#fa5252",
            opacity=0.7,
            line=dict(width=1, color="#333")
        ),
        text=[f"{l} ({q})" for l, q in zip(labels, quarter_labels)],
        customdata=np.stack([labels, quarter_labels, bubble_size], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>Quarter: %{customdata[1]}<br>Rel. Growth: %{x:.2f}%<br>Abs. Growth: %{y:,.0f}<br>Prior % of Total Expenses: %{customdata[2]:.1f}%<extra></extra>",
        name="Expense Items"
    ))
    # Add reference line for revenue growth (x = revenue growth)
    for i in range(len(revenue_growth)):
        fig.add_shape(
            type="line",
            x0=revenue_growth[i], x1=revenue_growth[i],
            y0=min(y_growth) if y_growth else 0, y1=max(y_growth) if y_growth else 1,
            line=dict(color="#228be6", width=2, dash="dash"),
            opacity=0.5,
            name="Revenue Growth"
        )
    # Add annotation for revenue growth
    for i, rev_g in enumerate(revenue_growth):
        fig.add_annotation(
            x=rev_g, y=max(y_growth) if y_growth else 0,
            text=f"Revenue growth reference ({quarter_labels[i]})",
            showarrow=True,
            arrowhead=2,
            ax=40, ay=-40,
            font=dict(color="#228be6", size=12),
            bgcolor="#e7f5ff",
            opacity=0.7
        )
    fig.update_layout(
        title=f"{symbol.upper()} Expense Growth vs Revenue Growth (Quarterly)",
        xaxis_title="Expense growth rate vs prior quarter (%)",
        yaxis_title="Expense change vs prior quarter (USD)",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH if 'DEFAULT_PLOTLY_WIDTH' in globals() else 900,
        height=DEFAULT_PLOTLY_HEIGHT if 'DEFAULT_PLOTLY_HEIGHT' in globals() else 500,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=14),
        legend_title="Expense category"
    )
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#228be6")
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#228be6")
    return fig

def plot_tax_and_interest_effects(symbol: str, income_df: pd.DataFrame) -> go.Figure:
    """
    Visualizes how interest and tax expenses impact income before tax and net income on a quarterly basis.

    After filtering `income_df` for the selected `symbol`, the function creates a stacked bar chart showing
    `income_before_tax` broken down into `interest_and_debt_expense` and `income_tax_expense`, with `net_income`
    overlaid as a line.  This combination highlights how financing costs and tax liabilities reduce pre‑tax earnings
    to arrive at net income.  By comparing the sizes of `interest_and_debt_expense` and `income_tax_expense` across
    quarters, investors can see whether changes in capital structure or tax rates affect profitability.  The
    effective tax rate (computed as `income_tax_expense` divided by `income_before_tax`) and interest burden
    provide context for evaluating how much of the company’s earnings are consumed by obligations rather than
    operations.  Monitoring these components helps investors anticipate how future income statement items may
    influence earnings and, ultimately, stock price:contentReference[oaicite:14]{index=14}.
    """
    # Validate input DataFrame
    if income_df is None or len(income_df) == 0:
        return go.Figure()
    df = income_df.copy()
    # Filter by symbol if column exists
    if "symbol" in df.columns:
        df = df[df["symbol"].str.upper() == symbol.upper()]
    if df.empty:
        return go.Figure()
    # Required columns
    base_cols = ["fiscal_date_ending", "income_before_tax", "interest_and_debt_expense", "income_tax_expense", "net_income"]
    for col in base_cols:
        if col not in df.columns:
            return go.Figure()
    # Sort by fiscal_date_ending
    df = df.sort_values("fiscal_date_ending")
    x = pd.to_datetime(df["fiscal_date_ending"])
    income_before_tax = pd.to_numeric(df["income_before_tax"], errors="coerce")
    interest_exp = pd.to_numeric(df["interest_and_debt_expense"], errors="coerce")
    tax_exp = pd.to_numeric(df["income_tax_expense"], errors="coerce")
    net_income = pd.to_numeric(df["net_income"], errors="coerce")
    # Calculate stacked bar components
    interest_exp = interest_exp.fillna(0)
    tax_exp = tax_exp.fillna(0)
    # The sum of interest and tax should not exceed income_before_tax; clip if needed
    total_stack = interest_exp + tax_exp
    other = income_before_tax - total_stack
    other = other.clip(lower=0)
    # Get column descriptions if available
    descriptions = _get_column_descriptions("income_statement_quarterly") if "_get_column_descriptions" in globals() else {}
    label_overrides = {
        "interest_and_debt_expense": "Interest & debt expense",
        "income_tax_expense": "Income tax expense",
        "net_income": "Net income",
    }
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Stacked bars: Interest, Tax, Other
    fig.add_trace(go.Bar(
        x=x,
        y=interest_exp,
    name=label_overrides.get("interest_and_debt_expense", descriptions.get("interest_and_debt_expense", "Interest & Debt Expense")),
        marker_color="#fa5252",
        hovertemplate="Interest: %{y:,.0f}<br>%{x|%Y-%m-%d}<extra></extra>"
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=x,
        y=tax_exp,
    name=label_overrides.get("income_tax_expense", descriptions.get("income_tax_expense", "Income Tax Expense")),
        marker_color="#fab005",
        hovertemplate="Tax: %{y:,.0f}<br>%{x|%Y-%m-%d}<extra></extra>"
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=x,
        y=other,
    name="Other pre‑tax income",
        marker_color="#40c057",
        hovertemplate="Other: %{y:,.0f}<br>%{x|%Y-%m-%d}<extra></extra>"
    ), secondary_y=False)
    # Overlay net income as a line
    fig.add_trace(go.Scatter(
        x=x,
        y=net_income,
        mode="lines+markers",
    name=label_overrides.get("net_income", descriptions.get("net_income", "Net Income")),
        line=dict(color="#228be6", width=3),
        marker=dict(symbol="circle", size=8, color="#228be6"),
        hovertemplate="Net Income: %{y:,.0f}<br>%{x|%Y-%m-%d}<extra></extra>",
        showlegend=True
    ), secondary_y=True)
    # Effective tax rate annotation (optional)
    effective_tax_rate = (tax_exp / income_before_tax * 100).replace([np.inf, -np.inf], np.nan)
    for i, (dt, etr) in enumerate(zip(x, effective_tax_rate)):
        if not np.isnan(etr):
            fig.add_annotation(
                x=dt, y=tax_exp.iloc[i],
                text=f"Tax Rate: {etr:.1f}%",
                showarrow=False,
                yshift=18,
                font=dict(size=11, color="#fab005"),
                bgcolor="#fffbe6",
                opacity=0.7
            )
    fig.update_layout(
        barmode="stack",
    title=f"{symbol.upper()} How Interest and Taxes Reduce Pre‑Tax Income",
    xaxis_title="Fiscal quarter end date",
    yaxis_title="Pre‑tax income components (USD)",
    yaxis2_title="Net income (USD)",
    legend_title="Component",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH if 'DEFAULT_PLOTLY_WIDTH' in globals() else 900,
        height=DEFAULT_PLOTLY_HEIGHT if 'DEFAULT_PLOTLY_HEIGHT' in globals() else 500,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=14)
    )
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#228be6", secondary_y=False)
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#228be6", secondary_y=True)
    return fig

def plot_metric_vs_future_stock_return(symbol: str, income_df: pd.DataFrame, price_df: pd.DataFrame, metric: str) -> go.Figure:
    """
    Creates a scatter plot showing the relationship between a chosen income statement metric and the stock’s
    subsequent quarterly return.

    The `metric` argument should be the name of a column in `income_df` (e.g., `total_revenue`, `net_income`,
    `operating_income`, `gross_profit`, `ebitda`, or any other income statement field).  For each fiscal quarter,
    the function computes the value of the selected metric and the percentage change in stock price from the
    earnings announcement date to the end of the next quarter using `price_df`.  It then plots the metric on the
    x‑axis and the subsequent return on the y‑axis.  By fitting a trend line or calculating the correlation,
    investors can assess whether higher values of the chosen fundamental measure lead to positive future stock
    performance.  This analysis operationalizes the principle that earnings and profitability drive stock prices
    over the long term:contentReference[oaicite:15]{index=15}:contentReference[oaicite:16]{index=16} and allows retail investors to test which
    income statement variables have the strongest predictive power for returns.  The function is flexible and
    encourages experimentation across all columns in the Alpha Vantage income statement file, enabling
    comprehensive exploration of how fundamentals influence future stock movements.
    """
    # Validate input DataFrames
    if income_df is None or price_df is None or len(income_df) == 0 or len(price_df) == 0:
        return go.Figure()
    df_income = income_df.copy()
    df_price = price_df.copy()
    # Filter by symbol if column exists
    if "symbol" in df_income.columns:
        df_income = df_income[df_income["symbol"].str.upper() == symbol.upper()]
    if "symbol" in df_price.columns:
        df_price = df_price[df_price["symbol"].str.upper() == symbol.upper()]
    if df_income.empty or df_price.empty:
        return go.Figure()
    # Required columns
    if "fiscal_date_ending" not in df_income.columns or metric not in df_income.columns:
        return go.Figure()
    if "date" not in df_price.columns or "close" not in df_price.columns:
        return go.Figure()
    # Sort and convert types
    df_income = df_income.sort_values("fiscal_date_ending")
    df_price["date"] = pd.to_datetime(df_price["date"])
    df_income["fiscal_date_ending"] = pd.to_datetime(df_income["fiscal_date_ending"])
    # For each quarter, get metric and future return
    x_metric = []
    y_return = []
    quarter_labels = []
    for i in range(len(df_income) - 1):
        row = df_income.iloc[i]
        next_row = df_income.iloc[i+1]
        metric_val = pd.to_numeric(row[metric], errors="coerce")
        # Find price at fiscal_date_ending (or closest after)
        start_date = row["fiscal_date_ending"]
        end_date = next_row["fiscal_date_ending"]
        price_start = df_price[df_price["date"] >= start_date]["close"]
        price_end = df_price[df_price["date"] >= end_date]["close"]
        if not price_start.empty and not price_end.empty:
            price_start_val = price_start.iloc[0]
            price_end_val = price_end.iloc[0]
            future_return = (price_end_val - price_start_val) / price_start_val * 100 if price_start_val != 0 else np.nan
            x_metric.append(metric_val)
            y_return.append(future_return)
            quarter_labels.append(str(start_date.date()))
    # Get column descriptions if available
    descriptions = _get_column_descriptions("income_statement_quarterly") if "_get_column_descriptions" in globals() else {}
    metric_label = descriptions.get(metric, metric.replace('_', ' ').title())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_metric,
        y=y_return,
        mode="markers",
        marker=dict(size=12, color="#228be6", line=dict(width=1, color="#333"), opacity=0.8),
        text=[f"Quarter: {q}" for q in quarter_labels],
        hovertemplate=f"{metric_label}: %{{x:,.2f}}<br>Future Return: %{{y:.2f}}%<br>%{{text}}<extra></extra>",
        name="Quarterly Points"
    ))
    # Add trendline if enough points
    if len(x_metric) > 1 and all([not np.isnan(x) and not np.isnan(y) for x, y in zip(x_metric, y_return)]):
        try:
            m, b = np.polyfit(x_metric, y_return, 1)
            x_fit = np.linspace(min(x_metric), max(x_metric), 100)
            y_fit = m * x_fit + b
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                line=dict(color="#fa5252", width=2, dash="dash"),
                name="Trendline",
                hoverinfo="skip"
            ))
            corr = np.corrcoef(x_metric, y_return)[0, 1]
            fig.add_annotation(
                xref="paper", yref="paper", x=0.99, y=0.01, showarrow=False,
                text=f"Corr: {corr:.2f}", font=dict(size=13, color="#fa5252"), bgcolor="#fff0f0", opacity=0.8
            )
        except Exception:
            pass
    fig.update_layout(
        title=f"{symbol.upper()} {metric_label} vs Next‑Quarter Stock Return",
        xaxis_title=f"{metric_label} (quarterly value)",
        yaxis_title="Next‑quarter stock return (%)",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH if 'DEFAULT_PLOTLY_WIDTH' in globals() else 900,
        height=DEFAULT_PLOTLY_HEIGHT if 'DEFAULT_PLOTLY_HEIGHT' in globals() else 500,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=14),
        legend_title="Quarter"
    )
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor="#228be6")
    return fig

def plot_key_metrics_dashboard(symbol: str, income_df: pd.DataFrame, price_df: pd.DataFrame) -> go.Figure:
    """
    Produces a summary dashboard displaying headline income statement metrics and their recent changes alongside stock performance.

    This function aggregates the most recent quarter’s values for key metrics — including `total_revenue`,
    `gross_profit`, `operating_income`, `net_income`, `ebit`, and `ebitda` — and calculates their quarter‑over‑quarter
    percentage changes.  It also computes gross, operating and net profit margins and displays the current stock
    price and its change since the previous quarter using `price_df`.  The dashboard arranges these values in a
    compact layout (e.g., with cards or tiles) so that investors can quickly grasp how the company’s financial
    position has evolved.  A mini line chart or sparkline for each metric can show recent trends.  This type of
    key‑metrics dashboard is ideal for summarizing earnings updates: it provides a quick overview of the most
    important P&L figures and how they changed versus the previous period:contentReference[oaicite:17]{index=17}, catering to
    users who already understand the company’s structure and just want to see the latest numbers:contentReference[oaicite:18]{index=18}.
    By juxtaposing these fundamentals with the stock price, the dashboard helps investors evaluate whether market
    reactions align with changes in the underlying business.
    """
    # Validate input DataFrames
    if income_df is None or price_df is None or len(income_df) == 0 or len(price_df) == 0:
        return go.Figure()
    df_income = income_df.copy()
    df_price = price_df.copy()
    # Filter by symbol if column exists
    if "symbol" in df_income.columns:
        df_income = df_income[df_income["symbol"].str.upper() == symbol.upper()]
    if "symbol" in df_price.columns:
        df_price = df_price[df_price["symbol"].str.upper() == symbol.upper()]
    if df_income.empty or df_price.empty:
        return go.Figure()
    # Metrics to show
    metrics = ["total_revenue", "gross_profit", "operating_income", "net_income", "ebit", "ebitda"]
    # Only keep metrics that exist
    metrics = [m for m in metrics if m in df_income.columns]
    if not metrics:
        return go.Figure()
    # Sort by fiscal_date_ending
    df_income = df_income.sort_values("fiscal_date_ending")
    # Get last and previous quarter
    if len(df_income) < 2:
        return go.Figure()
    last_row = df_income.iloc[-1]
    prev_row = df_income.iloc[-2]
    # Get column descriptions if available
    descriptions = _get_column_descriptions("income_statement_quarterly") if "_get_column_descriptions" in globals() else {}
    label_overrides = {
        "total_revenue": "Total revenue (sales)",
        "gross_profit": "Gross profit (revenue minus cost of revenue)",
        "operating_income": "Operating income (profit from core business)",
        "net_income": "Net income (profit after taxes)",
        "ebit": "EBIT (earnings before interest & taxes)",
        "ebitda": "EBITDA (earnings before interest, taxes, depreciation & amortization)",
    }
    # Prepare cards for each metric
    cards = []
    for m in metrics:
        val = pd.to_numeric(last_row[m], errors="coerce")
        prev_val = pd.to_numeric(prev_row[m], errors="coerce")
        delta = val - prev_val if not np.isnan(val) and not np.isnan(prev_val) else np.nan
        pct = (delta / prev_val * 100) if prev_val and not np.isnan(delta) and prev_val != 0 else np.nan
        color = "green" if not np.isnan(pct) and pct > 0 else ("red" if not np.isnan(pct) and pct < 0 else "gray")
        cards.append(dict(
            metric=m,
            label=label_overrides.get(m, descriptions.get(m, m.replace('_', ' ').title())),
            value=f"{val:,.0f}" if not np.isnan(val) else "N/A",
            delta=f"{delta:+,.0f}" if not np.isnan(delta) else "N/A",
            pct=f"{pct:+.1f}%" if not np.isnan(pct) else "N/A",
            color=color
        ))
    # Profit margins
    total_revenue = pd.to_numeric(last_row["total_revenue"], errors="coerce") if "total_revenue" in last_row else np.nan
    gross_margin = pd.to_numeric(last_row["gross_profit"], errors="coerce") / total_revenue * 100 if "gross_profit" in last_row and total_revenue else np.nan
    operating_margin = pd.to_numeric(last_row["operating_income"], errors="coerce") / total_revenue * 100 if "operating_income" in last_row and total_revenue else np.nan
    net_margin = pd.to_numeric(last_row["net_income"], errors="coerce") / total_revenue * 100 if "net_income" in last_row and total_revenue else np.nan
    # Stock price and change
    df_price = df_price.copy()
    if not np.issubdtype(df_price["date"].dtype, np.datetime64):
        df_price["date"] = pd.to_datetime(df_price["date"], errors="coerce")
    df_price = df_price.sort_values("date")
    last_date = pd.to_datetime(last_row["fiscal_date_ending"])
    prev_date = pd.to_datetime(prev_row["fiscal_date_ending"])
    price_last = df_price[df_price["date"] >= last_date]["close"]
    price_prev = df_price[df_price["date"] >= prev_date]["close"]
    price_last_val = price_last.iloc[0] if not price_last.empty else np.nan
    price_prev_val = price_prev.iloc[0] if not price_prev.empty else np.nan
    price_delta = price_last_val - price_prev_val if not np.isnan(price_last_val) and not np.isnan(price_prev_val) else np.nan
    price_pct = (price_delta / price_prev_val * 100) if price_prev_val and not np.isnan(price_delta) and price_prev_val != 0 else np.nan
    # Build dashboard as a table
    import plotly.graph_objects as go
    header = ["Metric", "Latest quarter value", "Change vs prior quarter", "Percent change"]
    values = [
        [c["label"] for c in cards],
        [c["value"] for c in cards],
        [c["delta"] for c in cards],
        [c["pct"] for c in cards],
    ]
    # Add margins and stock price
    header += [
        "Gross margin (% of revenue)",
        "Operating margin (% of revenue)",
        "Net margin (% of revenue)",
        "Stock price (close)",
        "Stock change",
        "Stock percent change",
    ]
    values[0] += [
        "Gross margin (% of revenue)",
        "Operating margin (% of revenue)",
        "Net margin (% of revenue)",
        "Stock price (close)",
        "",
        "",
    ]
    values[1] += [f"{gross_margin:.1f}%" if not np.isnan(gross_margin) else "N/A",
                 f"{operating_margin:.1f}%" if not np.isnan(operating_margin) else "N/A",
                 f"{net_margin:.1f}%" if not np.isnan(net_margin) else "N/A",
                 f"{price_last_val:,.2f}" if not np.isnan(price_last_val) else "N/A",
                 f"{price_delta:+,.2f}" if not np.isnan(price_delta) else "N/A",
                 f"{price_pct:+.1f}%" if not np.isnan(price_pct) else "N/A"]
    values[2] += ["", "", "", "", "", ""]
    values[3] += ["", "", "", "", "", ""]
    fig = go.Figure(go.Table(
        header=dict(
            values=header,
            fill_color="#eaf1fb",
            font=dict(color="#1c7ed6", family="Inter, sans-serif", size=16),
            align="left",
            line_color="#d0d7de"
        ),
        cells=dict(
            values=values,
            fill_color=["#f8fafc", "#f1f5f9"] * ((len(values[0]) // 2) + 1),
            font=dict(color="#212529", family="Inter, sans-serif", size=14),
            align="left",
            line_color="#e9ecef",
            height=28
        )
    ))
    fig.update_layout(
        title={
            "text": f"Income Statement Key Metrics: {symbol.upper()}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 22, "family": "Inter, sans-serif", "color": "#1c7ed6"}
        },
        width=DEFAULT_PLOTLY_WIDTH if 'DEFAULT_PLOTLY_WIDTH' in globals() else 1200,
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14, family="Inter, sans-serif", color="#212529"),
        template="plotly_white"
    )
    return fig