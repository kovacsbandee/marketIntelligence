import pandas as pd
import plotly.graph_objects as go
import json
import os
from plotly.subplots import make_subplots
from plotly.colors import qualitative
from infrastructure.ui.dash.add_price_indicators import AddPriceIndicators


def add_dividends(dividend_points: pd.DataFrame,
                  filtered_dividends: pd.DataFrame,
                  dividend_date_col: str,
                  figure: go.Figure) -> go.Figure:
    figure.add_trace(
        go.Scatter(
            x=dividend_points['date'],
            y=dividend_points['close'],
            mode="markers",
            marker=dict(size=4, color="blue"),
            name="Dividends",
            hovertext=[
                f"Date: {row[dividend_date_col]}, Amount: {row['amount']}"
                for _, row in filtered_dividends.iterrows()],
            hoverinfo="text"), row=1, col=1)
    return figure


def add_macd_subplot(price_table: pd.DataFrame,
                     figure: go.Figure) -> go.Figure:

    price_table = AddPriceIndicators(table=price_table).add_macd()

    figure.add_trace(
        go.Scatter(
            x=price_table['date'],
            y=price_table['MACD_line'],
            mode='lines',
            name='MACD Line'), row=2, col=1)
    figure.add_trace(
        go.Scatter(
            x=price_table['date'],
            y=price_table['signal_line'],
            mode='lines',
            name='Signal Line'), row=2, col=1)
    figure.add_trace(
        go.Bar(
            x=price_table['date'],
            y=price_table['MACD_histogram'],
            name='MACD Histogram'), row=2, col=1)

    return figure


def add_rsi_subplot(price_table: pd.DataFrame,
                    figure: go.Figure,
                    include_macd: bool) -> go.Figure:

    price_table = AddPriceIndicators(table=price_table).add_rsi()

    figure.add_trace(
        go.Scatter(
            x=price_table['date'],
            y=price_table['RSI'],
            mode='lines',
            name='RSI'), row=3 if include_macd else 2, col=1)
    figure.add_trace(
        go.Scatter(
            x=price_table['date'],
            y=[70] * len(price_table),  # Overbought level
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Overbought (70)'), row=3 if include_macd else 2, col=1)
    figure.add_trace(
        go.Scatter(
            x=price_table['date'],
            y=[30] * len(price_table),  # Oversold level
            mode='lines',
            line=dict(dash='dash', color='green'),
            name='Oversold (30)'), row=3 if include_macd else 2, col=1)

    return figure


def plot_price_with_indicators(price_table,
                               include_macd: bool = False,
                               include_rsi: bool = False):
    """
    Plots price data with optional MACD, RSI, and dividends using Plotly's make_subplots.

    Parameters:
    price_table (pd.DataFrame): DataFrame containing price data (['date', 'open', 'high', 'low', 'close']).
    dividend_table (pd.DataFrame): DataFrame containing dividend data with a date column and amount column.
    include_macd (bool): Whether to add MACD visualization if columns exist.
    include_rsi (bool): Whether to add RSI visualization if columns exist.
    dividend_date_col (str): The column name for dividend dates in dividend_table.
    dividend_filter_range (tuple): Tuple of start and end dates (e.g., ('2023-01-01', '2023-12-31')) to filter dividends.

    Returns:
    Plotly figure object.
    """
    # Initialize subplots
    rows = 1 + int(include_macd) + int(include_rsi)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5] + [0.25] * (rows - 1),
        subplot_titles=["Price Data"] +
        (["MACD"] if include_macd else []) +
        (["RSI"] if include_rsi else [])
    )

    # Plot base candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=price_table['date'],
            open=price_table['open'],
            high=price_table['high'],
            low=price_table['low'],
            close=price_table['close'],
            name="Price",
        ),
        row=1, col=1
    )

    # Add MACD visualization if requested
    if include_macd:
        try:
            fig = add_macd_subplot(price_table=price_table, figure=fig)
        except KeyError as e:
            print(
                f"Error: MACD values not found in the price table. Missing column: {e}")

    # Add RSI visualization if requested
    if include_rsi:
        try:
            add_rsi_subplot(price_table=price_table,
                            figure=fig,
                            include_macd=include_macd)
        except KeyError as e:
            print(
                f"Error: RSI values not found in the price table. Missing column: {e}")
    fig.update_layout(
        title={
            "text": "Price Data with Indicators and Dividends",
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}  # Larger title font size
        },
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        showlegend=True,
        width=1800,
        height=1800,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14))

    return fig


def plot_insider_transactions(insider_transaction):
    """
    Plots insider transactions with share price and volume using Plotly's make_subplots.

    Parameters:
    insider_transaction (pd.DataFrame): DataFrame containing insider transaction data with columns:
        ['transaction_date', 'executive_title', 'security_type', 'acquisition_or_disposal', 'shares', 'share_price'].

    Returns:
    Plotly figure object.
    """

    # Aggregate data
    plot_df = insider_transaction.groupby(
        by=['transaction_date', 'executive_title',
            'security_type', 'acquisition_or_disposal']
    ).agg({'shares': 'mean', 'share_price': 'mean'}).reset_index()

    # Map colors for scatter plots
    scatter_colors = plot_df['acquisition_or_disposal'].map(
        {'A': 'blue', 'D': 'red'})

    # Map colors for bar plots
    unique_security_types = plot_df['security_type'].unique()
    bar_color_mapping = {security_type: qualitative.G10[i % len(
        qualitative.G10)] for i, security_type in enumerate(unique_security_types)}
    bar_colors = plot_df['security_type'].map(bar_color_mapping)

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=["Share Price", "Volume"])

    # Add scatter plots for each acquisition/disposal type
    for i, typ in enumerate(plot_df['acquisition_or_disposal'].unique()):
        filt_df = plot_df[plot_df['acquisition_or_disposal'] == typ]

        # Scatter plot for share price
        fig.add_trace(
            go.Scatter(
                x=filt_df['transaction_date'],
                y=filt_df['share_price'],
                mode='markers',
                marker=dict(color=qualitative.G10[i], size=5),
                name='Buy' if typ == 'A' else 'Sell'
            ),
            row=1, col=1
        )

        # Scatter plot for shares (volume)
        fig.add_trace(
            go.Scatter(
                x=filt_df['transaction_date'],
                y=filt_df['shares'],
                mode='markers',
                showlegend=False,
                marker=dict(color=qualitative.G10[i], size=5)
            ),
            row=2, col=1
        )

    # Update axes
    fig.update_yaxes(title="Share Price", row=1, col=1)
    fig.update_yaxes(title="Volume", row=2, col=1)
    fig.update_xaxes(title="Transaction Date", row=2, col=1)

    # Update layout
    fig.update_layout(
        title={
            "text": "Insider Transactions: Share Price and Volume",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20}
        },
        legend=dict(title="Legend"),
        template="plotly_white",
        width=1000,
        height=700,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14)
    )

    return fig


def plot_balance_sheet_time_series(
    balance_df: pd.DataFrame,
    columns: list,
    descriptions: dict = None
) -> go.Figure:
    """
    Plot a time series line chart for selected balance sheet metrics (annual or quarterly).

    Parameters:
    - balance_df (pd.DataFrame): DataFrame containing balance sheet data for a single symbol.
    - columns (list): List of column names to plot (e.g., ['total_assets', 'total_liabilities']).
    - descriptions (dict, optional): Mapping of column names to human-readable descriptions for tooltips.
      If not provided, will be auto-extracted from configs/alpha_vantage_column_description.json.

    Returns:
    - go.Figure: Plotly figure object with time series lines for each selected metric.
    """
    # Auto-load descriptions if not provided
    if descriptions is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "configs",
            "alpha_vantage_column_description.json"
        )
        with open(config_path, "r") as f:
            config = json.load(f)
        # Try to autodetect annual or quarterly
        if "fiscal_date_ending" in balance_df.columns:
            # Heuristic: if there are more than 12 unique dates, it's quarterly
            unique_dates = pd.to_datetime(balance_df["fiscal_date_ending"].dropna().unique())
            if len(unique_dates) > 12:
                section = "balance_sheet_quarterly"
            else:
                section = "balance_sheet_annual"
        else:
            section = "balance_sheet_annual"
        # Build descriptions dict
        descriptions = {}
        for entry in config.get(section, []):
            for k, v in entry.items():
                descriptions[k] = v

    # Prepare figure
    fig = go.Figure()
    # X-axis: fiscal_date_ending
    if "fiscal_date_ending" not in balance_df.columns:
        raise ValueError("DataFrame must contain 'fiscal_date_ending' column.")
    x = pd.to_datetime(balance_df["fiscal_date_ending"])

    # Plot each metric
    for col in columns:
        y = pd.to_numeric(balance_df[col], errors="coerce") if col in balance_df.columns else pd.Series([float('nan')] * len(x))
        # Plot empty line if all NaN
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=descriptions.get(col, col),
                hovertemplate=f"<b>{descriptions.get(col, col)}</b><br>Date: %{x}<br>Value: %{y}<extra></extra>"
            )
        )

    fig.update_layout(
        title="Balance Sheet Time Series",
        xaxis_title="Fiscal Date Ending",
        yaxis_title="Value",
        legend_title="Metric",
        hovermode="x unified"
    )
    return fig


def plot_balance_sheet_stacked_area(balance_df: pd.DataFrame, stack_groups: dict, descriptions: dict) -> go.Figure:
    """
    Plot stacked area chart to visualize the composition of assets or liabilities over time.

    Parameters:
    - balance_df (pd.DataFrame): DataFrame with balance sheet data.
    - stack_groups (dict): Dictionary mapping group names (e.g., 'Assets') to lists of column names to stack.
    - descriptions (dict): Mapping of column names to descriptions for hovertext.

    Returns:
    - go.Figure: Plotly figure object with stacked area plots for each group.
    """
    pass

def plot_balance_sheet_bar(balance_df: pd.DataFrame, group_columns: dict, descriptions: dict) -> go.Figure:
    """
    Plot grouped or stacked bar chart comparing current vs. non-current assets and liabilities.

    Parameters:
    - balance_df (pd.DataFrame): DataFrame with balance sheet data.
    - group_columns (dict): Dictionary with keys like 'Current Assets', 'Non-Current Assets', etc., and values as column names.
    - descriptions (dict): Mapping of column names to descriptions for hovertext.

    Returns:
    - go.Figure: Plotly figure object with grouped/stacked bars for each period.
    """
    pass

def plot_balance_sheet_pie(balance_df: pd.DataFrame, date: str, columns: list, descriptions: dict) -> go.Figure:
    """
    Plot pie or donut chart showing the breakdown of assets or liabilities for a selected date.

    Parameters:
    - balance_df (pd.DataFrame): DataFrame with balance sheet data.
    - date (str): The fiscal date to visualize (must match a row in balance_df).
    - columns (list): List of columns to include in the pie chart.
    - descriptions (dict): Mapping of column names to descriptions for labels/hovertext.

    Returns:
    - go.Figure: Plotly figure object with a pie or donut chart.
    """
    pass

def render_balance_sheet_metric_cards(balance_df: pd.DataFrame, date: str, metrics: list, descriptions: dict):
    """
    Generate Dash components (e.g., html.Div or dmc.Paper) displaying key balance sheet metrics as cards.

    Parameters:
    - balance_df (pd.DataFrame): DataFrame with balance sheet data.
    - date (str): The fiscal date to display metrics for.
    - metrics (list): List of metric column names to display.
    - descriptions (dict): Mapping of column names to descriptions for tooltips.

    Returns:
    - list: List of Dash components representing metric cards.
    """
    pass
