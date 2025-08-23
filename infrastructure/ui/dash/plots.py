import pandas as pd
import plotly.graph_objects as go
import json
import os
from plotly.subplots import make_subplots
from plotly.colors import qualitative
from infrastructure.ui.dash.add_price_indicators import AddPriceIndicators
import dash_mantine_components as dmc


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

    price_table = AddPriceIndicators(table=price_table).macd()

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

    price_table = AddPriceIndicators(table=price_table).rsi()

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

def add_vwap_subplot(price_table: pd.DataFrame,
                      figure: go.Figure) -> go.Figure:
    """
    Adds a VWAP (Volume Weighted Average Price) subplot to the figure.

    Parameters:
        price_table (pd.DataFrame): DataFrame containing price data with 'date', 'close', and 'volume' columns.
        figure (go.Figure): The Plotly figure to which the VWAP subplot will be added.

    Returns:
        go.Figure: The updated figure with the VWAP subplot added.
    """
    # Calculate VWAP using AddPriceIndicators
    price_table = AddPriceIndicators(table=price_table).vwap()

    # Add VWAP line to the subplot
    figure.add_trace(
        go.Scatter(
            x=price_table['date'],
            y=price_table['VWAP'],
            mode='lines',
            name='VWAP',
            line=dict(color='purple')
        ),
        row=4, col=1
    )

    return figure


def plot_price_with_indicators(price_table,
                               include_macd: bool = False,
                               include_rsi: bool = False,
                               include_vwap: bool = False):
    """
    Plots price data with optional MACD, RSI, VWAP, and dividends using Plotly's make_subplots.

    Parameters:
    price_table (pd.DataFrame): DataFrame containing price data (['date', 'open', 'high', 'low', 'close']).
    include_macd (bool): Whether to add MACD visualization if columns exist.
    include_rsi (bool): Whether to add RSI visualization if columns exist.
    include_vwap (bool): Whether to add VWAP visualization if columns exist.

    Returns:
    Plotly figure object.
    """
    # Initialize subplots
    rows = 1 + int(include_macd) + int(include_rsi) + int(include_vwap)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5] + [0.25] * (rows - 1),
        subplot_titles=["Price Data"] +
        (["MACD"] if include_macd else []) +
        (["RSI"] if include_rsi else []) +
        (["VWAP"] if include_vwap else [])
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

    # Add VWAP visualization if requested
    if include_vwap:
        try:
            fig = add_vwap_subplot(price_table=price_table, figure=fig)
        except KeyError as e:
            print(
                f"Error: VWAP values not found in the price table. Missing column: {e}")

    # Update layout to add date labels to each x-axis
    fig.update_layout(
        title={
            "text": "Price Data with Indicators",
            "x": 0.5,  
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20} 
        },
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        showlegend=True,
        width=1800,
        height=1800,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14)
    )

    # Add date labels to each x-axis
    for i in range(1, rows + 1):
        fig.update_xaxes(title_text="Date", row=i, col=1)

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


def _auto_load_balance_descriptions(balance_df):
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "configs",
        "alpha_vantage_column_description.json"
    )
    with open(config_path, "r") as f:
        config = json.load(f)
    # Heuristic: if there are more than 12 unique dates, it's quarterly
    if "fiscal_date_ending" in balance_df.columns:
        unique_dates = pd.to_datetime(balance_df["fiscal_date_ending"].dropna().unique())
        if len(unique_dates) > 12:
            section = "balance_sheet_quarterly"
        else:
            section = "balance_sheet_annual"
    else:
        section = "balance_sheet_annual"
    descriptions = {}
    for entry in config.get(section, []):
        for k, v in entry.items():
            descriptions[k] = v
    return descriptions


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


def _find_row_by_date(balance_df, date):
    # Accepts date as string or datetime, returns the row for the closest matching date
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)
    # Find the row with the exact date, or the closest previous date if not found
    idx = balance_df["fiscal_date_ending"].sub(date).abs().idxmin()
    return balance_df.loc[idx]


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
        hovermode="x unified",
        template="plotly_white",
        width=1800,  # Consistent wide width
        height=700,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14)
    )
    return fig


def plot_balance_sheet_stacked_area(balance_df: pd.DataFrame, stack_groups: dict, descriptions: dict = None) -> go.Figure:
    """
    Plot stacked area chart to visualize the composition of assets or liabilities over time.
    Supports nested grouping.
    """
    if descriptions is None:
        descriptions = _auto_load_balance_descriptions(balance_df)
    if "fiscal_date_ending" not in balance_df.columns:
        raise ValueError("DataFrame must contain 'fiscal_date_ending' column.")
    x = pd.to_datetime(balance_df["fiscal_date_ending"])
    fig = go.Figure()
    color_idx = 0
    color_list = qualitative.Plotly + qualitative.G10 + qualitative.Pastel
    def add_area_traces(group, parent_name=""):
        nonlocal color_idx
        if isinstance(group, dict):
            for k, v in group.items():
                add_area_traces(v, parent_name + k + " - " if parent_name else k + " - ")
        else:
            for col in group if isinstance(group, list) else [group]:
                y = pd.to_numeric(balance_df[col], errors="coerce") if col in balance_df.columns else pd.Series([float('nan')] * len(x))
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        stackgroup="one",
                        name=parent_name + descriptions.get(col, col),
                        line=dict(width=0.5, color=color_list[color_idx % len(color_list)]),
                        hovertemplate=f"<b>{descriptions.get(col, col)}</b><br>Date: %{{x}}<br>Value: %{{y:,.0f}}<extra></extra>"
                    )
                )
                color_idx += 1
    add_area_traces(stack_groups)
    fig.update_layout(
        title="Balance Sheet Stacked Area",
        xaxis_title="Fiscal Date Ending",
        yaxis_title="Value",
        legend_title="Component",
        hovermode="x unified",
        template="plotly_white",
        width=1800,  # Consistent wide width
        height=700,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14)
    )
    return fig


def plot_balance_sheet_bar(balance_df: pd.DataFrame, group_columns: dict, descriptions: dict = None) -> go.Figure:
    """
    Plot grouped or stacked bar chart comparing current vs. non-current assets and liabilities.
    Supports nested grouping.
    """
    if descriptions is None:
        descriptions = _auto_load_balance_descriptions(balance_df)
    if "fiscal_date_ending" not in balance_df.columns:
        raise ValueError("DataFrame must contain 'fiscal_date_ending' column.")
    x = pd.to_datetime(balance_df["fiscal_date_ending"])
    fig = go.Figure()
    color_idx = 0
    color_list = qualitative.Plotly + qualitative.G10 + qualitative.Pastel
    for group_name, group in group_columns.items():
        cols = _flatten_group(group)
        y = balance_df[cols].sum(axis=1, numeric_only=True) if cols else pd.Series([float('nan')] * len(x))
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                name=descriptions.get(group_name, group_name),
                marker_color=color_list[color_idx % len(color_list)],
                hovertemplate=f"<b>{descriptions.get(group_name, group_name)}</b><br>Date: %{{x}}<br>Value: %{{y:,.0f}}<extra></extra>"
            )
        )
        color_idx += 1
    fig.update_layout(
        barmode="group",
        title="Balance Sheet Bar Chart",
        xaxis_title="Fiscal Date Ending",
        yaxis_title="Value",
        legend_title="Group",
        hovermode="x unified",
        template="plotly_white",
        width=1800,  # Consistent wide width
        height=700,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14)
    )
    return fig


def plot_balance_sheet_pie(balance_df: pd.DataFrame, date: str, columns: list, descriptions: dict = None) -> go.Figure:
    """
    Generates a pie chart visualization of a balance sheet breakdown for a specific date.

    Args:
        balance_df (pd.DataFrame): A DataFrame containing balance sheet data. 
            Must include a 'fiscal_date_ending' column.
        date (str): The specific date to filter the balance sheet data. 
            Should be in a format that can be parsed by pandas.
        columns (list): A list of column names in the DataFrame to include in the pie chart.
        descriptions (dict, optional): A dictionary mapping column names to their descriptions 
            for labeling the pie chart. If None, descriptions are auto-loaded.

    Returns:
        go.Figure: A Plotly Figure object representing the pie chart.

    Raises:
        ValueError: If the DataFrame does not contain the 'fiscal_date_ending' column.

    Notes:
        - The pie chart includes a hole in the center (donut chart style).
        - The hover template displays the label and value of each slice.
        - The chart layout is styled with a white template and includes a title 
          indicating the fiscal date ending.

    Example:
        fig = plot_balance_sheet_pie(balance_df, "2023-01-01", ["assets", "liabilities"], {"assets": "Assets", "liabilities": "Liabilities"})
        fig.show()
    """
    if descriptions is None:
        descriptions = _auto_load_balance_descriptions(balance_df)
    if "fiscal_date_ending" not in balance_df.columns:
        raise ValueError("DataFrame must contain 'fiscal_date_ending' column.")
    row = _find_row_by_date(balance_df, date)
    values = [row[col] if col in row else 0 for col in columns]
    labels = [descriptions.get(col, col) for col in columns]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        title=f"Balance Sheet Breakdown ({row['fiscal_date_ending'].date()})",
        template="plotly_white",
        width=1000,  # Pie chart can be a bit smaller, but you can set to 1800 if you want
        height=700,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14)
    )
    return fig


def render_balance_sheet_metric_cards(balance_df: pd.DataFrame, date: str, metrics: list, descriptions: dict = None):
    """
    Renders a list of metric cards for a balance sheet.

    Args:
        balance_df (pd.DataFrame): A DataFrame containing balance sheet data. 
            Must include a 'fiscal_date_ending' column.
        date (str): The fiscal date for which the metrics should be displayed.
        metrics (list): A list of metric names to display on the cards.
        descriptions (dict, optional): A dictionary mapping metric names to their descriptions. 
            If None, descriptions are auto-loaded from the balance DataFrame.

    Returns:
        list: A list of Dash Mantine Components (dmc.Paper) representing the metric cards.

    Raises:
        ValueError: If the 'fiscal_date_ending' column is not present in the DataFrame.

    Notes:
        - Each card displays the metric description and its formatted value.
        - If a metric value is missing or NaN, "N/A" is displayed instead.
        - Cards are styled with a fixed width, shadow, and margin for inline display.
    """
    if descriptions is None:
        descriptions = _auto_load_balance_descriptions(balance_df)
    if "fiscal_date_ending" not in balance_df.columns:
        raise ValueError("DataFrame must contain 'fiscal_date_ending' column.")
    row = _find_row_by_date(balance_df, date)
    cards = []
    for metric in metrics:
        value = row[metric] if metric in row else None
        formatted = f"${value:,.0f}" if value is not None and pd.notnull(value) else "N/A"
        cards.append(
            dmc.Paper(
                [
                    dmc.Text(descriptions.get(metric, metric), size="sm", c="dimmed"),
                    dmc.Text(formatted, size="xl", fw=700)
                ],
                p="md",
                shadow="sm",
                radius="md",
                style={"width": 200, "display": "inline-block", "margin": "8px"}
            )
        )
    return cards


def plot_company_fundamentals_table(fundamentals_df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Returns a Plotly Table Figure displaying all available company fundamentals for the given symbol,
    styled to match the Mantine theme, with a single column and bold values.

    Args:
        fundamentals_df (pd.DataFrame): DataFrame loaded from company_fundamentals.csv.
        symbol (str): Stock symbol to display.

    Returns:
        go.Figure: Plotly Table Figure.
    """
    # Find the row for the symbol (case-insensitive)
    row = fundamentals_df[fundamentals_df['symbol'].str.upper() == symbol.upper()]
    if row.empty:
        raise ValueError(f"Symbol '{symbol}' not found in fundamentals data.")

    # Prepare data for table
    row_dict = row.iloc[0].to_dict()
    keys = list(row_dict.keys())
    values = [row_dict[k] for k in keys]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["<b>Field</b>", "<b>Value</b>"],
                    fill_color="#eaf1fb",  # Mantine light blue
                    font=dict(color="#1c7ed6", family="Inter, sans-serif", size=16),
                    align="left",
                    line_color="#d0d7de"
                ),
                cells=dict(
                    values=[
                        [f"<b>{str(k)}</b>" for k in keys],
                        [f"<b>{str(v)}</b>" for v in values]
                    ],
                    fill_color=[["#f8fafc", "#f1f5f9"] * (len(keys) // 2 + 1)] * 2,
                    font=dict(color=["#212529", "#212529"], family="Inter, sans-serif", size=14),
                    align="left",
                    line_color="#e9ecef",
                    format=["", ""],
                    height=28
                )
            )
        ]
    )
    fig.update_layout(
        title={
            "text": f"Company Fundamentals: {symbol.upper()}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 22, "family": "Inter, sans-serif", "color": "#1c7ed6"}
        },
        width=1800,
        height=700,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14, family="Inter, sans-serif", color="#212529"),
        template="plotly_white"
    )
    return fig