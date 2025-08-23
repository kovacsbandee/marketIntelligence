import pandas as pd
import plotly.graph_objects as go
import json
import os
from plotly.subplots import make_subplots
from plotly.colors import qualitative
from infrastructure.ui.dash.add_price_indicators import AddPriceIndicators
import dash_mantine_components as dmc

# --- Default Plotly Figure Size ---
DEFAULT_PLOTLY_WIDTH = 1800
DEFAULT_PLOTLY_HEIGHT = 700


class Plotting():
    """
    This will be a base class for all the plotting functionalities used in the callbacks.
    Every tab will have a child class originating from this base class, where the actual plotting will be implemented,
    as a separate Plotly figure for each metric or indicator.
    """
    def __init__(self):
        self.descriptions = self._load_column_description_json()
        pass


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
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_WIDTH,  # This plot is intentionally square (1800x1800)
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
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14)
    )

    return fig



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
        descriptions = _auto_load_table_descriptions(balance_df)

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
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
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
        descriptions = _auto_load_table_descriptions(balance_df)
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
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
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
        descriptions = _auto_load_table_descriptions(balance_df)
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
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
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
        descriptions = _auto_load_table_descriptions(balance_df)
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
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
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
        descriptions = _auto_load_table_descriptions(balance_df)
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
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14, family="Inter, sans-serif", color="#212529"),
        template="plotly_white"
    )
    return fig


def plot_eps_actual_vs_estimate(symbol, data):
    """
    Plots a time-series of the company’s **reported EPS** vs **estimated EPS** each quarter, highlighting where actual earnings beat or missed expectations. 
    This line chart uses quarter-end dates on the X-axis and EPS on the Y-axis, with one trace for actual EPS and another for consensus estimates:contentReference[oaicite:0]{index=0}. 
    Markers or annotations can indicate positive surprises (actual above estimate) and negative surprises (actual below estimate) for each quarter. 
    This visualization helps investors see trends in earnings performance over time
    – whether EPS is growing and how consistently the company exceeds or falls short of forecasts – providing insight into the company’s track record and earnings momentum.

    Plots a time-series of the company’s reported EPS vs estimated EPS each quarter,
    highlighting where actual earnings beat or missed expectations.

    Args:
        symbol (str): Stock symbol.
        data (pd.DataFrame): DataFrame with columns:
            - fiscal_date_ending (quarter end date)
            - reported_eps (actual EPS)
            - estimated_eps (consensus estimate)
            - surprise (actual - estimate)
            - surprise_percentage (optional)
            - report_time (optional, e.g. 'BTO', 'AMC')
    Returns:
        go.Figure: Plotly figure with two lines (actual, estimate) and markers for beats/misses.
    """
    descriptions = _get_column_descriptions("earnings_quarterly")
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if data.empty or "fiscal_date_ending" not in data.columns:
        return go.Figure()
    df = data.copy()
    df = df.sort_values("fiscal_date_ending")
    x = pd.to_datetime(df["fiscal_date_ending"])
    actual = pd.to_numeric(df["reported_eps"], errors="coerce")
    estimate = pd.to_numeric(df["estimated_eps"], errors="coerce")

    # Markers for beat/miss
    beat_mask = (actual > estimate) & estimate.notnull() & actual.notnull()
    miss_mask = (actual < estimate) & estimate.notnull() & actual.notnull()

    fig = go.Figure()
    # Actual EPS line
    fig.add_trace(go.Scatter(
        x=x,
        y=actual,
        mode="lines+markers",
        name=descriptions.get("reported_eps", "Actual EPS"),
        line=dict(color="#228be6", width=3),
        marker=dict(symbol="circle", size=8, color="#228be6"),
        hovertemplate=f"<b>{descriptions.get('reported_eps', 'Actual EPS')}</b><br>{descriptions.get('fiscal_date_ending', 'Date')}: "+"%{x|%Y-%m-%d}<br>"+f"{descriptions.get('reported_eps', 'EPS')}: "+"%{y:.2f}<extra></extra>"
    ))
    # Estimate EPS line
    fig.add_trace(go.Scatter(
        x=x,
        y=estimate,
        mode="lines+markers",
        name=descriptions.get("estimated_eps", "Estimate EPS"),
        line=dict(color="#fab005", width=3, dash="dash"),
        marker=dict(symbol="diamond", size=8, color="#fab005"),
        hovertemplate=f"<b>{descriptions.get('estimated_eps', 'Estimate EPS')}</b><br>{descriptions.get('fiscal_date_ending', 'Date')}: "+"%{x|%Y-%m-%d}<br>"+f"{descriptions.get('estimated_eps', 'EPS')}: "+"%{y:.2f}<extra></extra>"
    ))
    # Beat markers (green up triangle)
    fig.add_trace(go.Scatter(
        x=x[beat_mask],
        y=actual[beat_mask],
        mode="markers",
        name="Beat",
        marker=dict(symbol="triangle-up", size=14, color="green", line=dict(width=1, color="black")),
        showlegend=True,
        hovertemplate="<b>Beat</b><br>"+f"{descriptions.get('fiscal_date_ending', 'Date')}: "+"%{x|%Y-%m-%d}<br>"+f"{descriptions.get('reported_eps', 'Actual EPS')}: "+"%{y:.2f}<extra></extra>"
    ))
    # Miss markers (red down triangle)
    fig.add_trace(go.Scatter(
        x=x[miss_mask],
        y=actual[miss_mask],
        mode="markers",
        name="Miss",
        marker=dict(symbol="triangle-down", size=14, color="red", line=dict(width=1, color="black")),
        showlegend=True,
        hovertemplate="<b>Miss</b><br>"+f"{descriptions.get('fiscal_date_ending', 'Date')}: "+"%{x|%Y-%m-%d}<br>"+f"{descriptions.get('reported_eps', 'Actual EPS')}: "+"%{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"{symbol.upper()} EPS: {descriptions.get('reported_eps', 'Actual EPS')} vs {descriptions.get('estimated_eps', 'Estimate EPS')} (Quarterly)",
        xaxis_title=descriptions.get("fiscal_date_ending", "Fiscal Quarter End"),
        yaxis_title=descriptions.get("reported_eps", "EPS"),
        legend_title="Legend",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=14)
    )
    return fig

def plot_eps_surprise_percentage(symbol, data):
    """
    Creates a bar chart showing the **earnings surprise** for each quarter as a percentage. 
    Each bar represents how much the reported EPS diverged from the consensus estimate (in percentage terms):contentReference[oaicite:1]{index=1}.
      Bars extend upward (often colored green) for positive surprises (beats) and downward (colored red) for negative surprises (misses), with a zero line as the “met expectations” baseline. 
      This chart lets investors quickly gauge the magnitude of each earnings beat or miss and spot patterns (e.g. consistently beating estimates or occasional large misses). 
      Such surprise percentages are crucial since bigger positive surprises often coincide with stock price jumps, while big misses can trigger drops:contentReference[oaicite:2]{index=2}. 
      This visualization uses the `surprise_percentage` from the data to include every quarter’s surprise, helping to illustrate the volatility and reliability of earnings relative to expectations.

    Plots a bar chart showing the earnings surprise for each quarter as a percentage.
    Uses 'surprise_percentage' if available, else computes from reported/estimated EPS.

    Args:
        symbol (str): Stock symbol.
        data (pd.DataFrame): DataFrame with columns:
            - fiscal_date_ending (quarter end date)
            - reported_eps (actual EPS)
            - estimated_eps (consensus estimate)
            - surprise (actual - estimate)
            - surprise_percentage (optional)
    Returns:
        go.Figure: Plotly bar chart of surprise percentages per quarter.
    """
    descriptions = _get_column_descriptions("earnings_quarterly")
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if data.empty or "fiscal_date_ending" not in data.columns:
        return go.Figure()
    df = data.copy()
    df = df.sort_values("fiscal_date_ending")
    x = pd.to_datetime(df["fiscal_date_ending"])
    # Use surprise_percentage if available, else compute
    if "surprise_percentage" in df.columns and df["surprise_percentage"].notnull().any():
        surprise_pct = pd.to_numeric(df["surprise_percentage"], errors="coerce")
    else:
        actual = pd.to_numeric(df["reported_eps"], errors="coerce")
        estimate = pd.to_numeric(df["estimated_eps"], errors="coerce")
        surprise_pct = ((actual - estimate) / estimate * 100).where(estimate != 0)

    # Color bars: green for positive, red for negative
    colors = ["green" if v > 0 else "red" if v < 0 else "gray" for v in surprise_pct.fillna(0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=surprise_pct,
        marker_color=colors,
        name=descriptions.get("surprise_percentage", "Surprise %"),
        hovertemplate=f"<b>{descriptions.get('fiscal_date_ending', 'Date')}</b>: "+"%{x|%Y-%m-%d}<br>"+f"{descriptions.get('surprise_percentage', 'Surprise %')}: "+"%{y:.2f}%<extra></extra>"
    ))
    fig.add_shape(
        type="line",
        x0=min(x), x1=max(x),
        y0=0, y1=0,
        line=dict(color="black", width=1, dash="dash"),
        xref="x", yref="y"
    )
    fig.update_layout(
        title=f"{symbol.upper()} {descriptions.get('surprise_percentage', 'EPS Surprise Percentage')} (Quarterly)",
        xaxis_title=descriptions.get("fiscal_date_ending", "Fiscal Quarter End"),
        yaxis_title=descriptions.get("surprise_percentage", "Surprise (%)"),
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=14)
    )
    return fig


def plot_eps_actual_vs_estimate_scatter(symbol, data):
    """
    Plots actual EPS against estimated EPS for each quarterly report as a scatter plot, to visualize the **accuracy of forecasts** and the company’s bias in beating or missing. 
    Each point represents a quarter, with the X-coordinate as the estimated EPS and the Y-coordinate as the reported actual EPS. 
    A 45° reference line (where Actual = Estimate) is drawn; points above this line indicate quarters where earnings beat estimates, and points below indicate misses. 
    The distance from the line reflects the surprise magnitude – farther above means a bigger beat, farther below means a bigger miss. 
    This chart provides a clear view of the company’s overall track record: for example, a cluster of points above the line signals the company tends to outperform estimates. 
    We utilize all relevant data – reported vs estimated EPS for each quarter – and can even distinguish **announcement timing** (pre-market vs post-market) by using different marker colors or shapes for `report_time`. 
    This helps investors understand not just the trend over time, but the relationship between forecasts and actual performance in one view, reinforcing how often and by how much the company surprises the market.

    Plots actual EPS against estimated EPS for each quarterly report as a scatter plot.
    Uses ORM column names if present.

    Args:
        symbol (str): Stock symbol.
        data (pd.DataFrame): DataFrame with columns:
            - fiscal_date_ending (quarter end date)
            - reported_eps (actual EPS)
            - estimated_eps (consensus estimate)
            - report_time (optional, e.g. 'BTO', 'AMC')
    Returns:
        go.Figure: Plotly scatter plot of actual vs estimate EPS.
    """
    descriptions = _get_column_descriptions("earnings_quarterly")
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if data.empty or "reported_eps" not in data.columns or "estimated_eps" not in data.columns:
        return go.Figure()
    df = data.copy()
    df = df.dropna(subset=["reported_eps", "estimated_eps"])
    if df.empty:
        return go.Figure()
    actual = pd.to_numeric(df["reported_eps"], errors="coerce")
    estimate = pd.to_numeric(df["estimated_eps"], errors="coerce")
    hover_text = [
        f"{descriptions.get('fiscal_date_ending', 'Quarter')}: {str(row['fiscal_date_ending'])[:10]}<br>"
        f"{descriptions.get('reported_eps', 'Actual EPS')}: {row['reported_eps']:.2f}<br>"
        f"{descriptions.get('estimated_eps', 'Estimate EPS')}: {row['estimated_eps']:.2f}"
        + (f"<br>{descriptions.get('report_time', 'Report Time')}: {row['report_time']}" if "report_time" in df.columns and pd.notnull(row['report_time']) else "")
        for _, row in df.iterrows()
    ]
    # Marker style by report_time if available
    if "report_time" in df.columns:
        report_time = df["report_time"].fillna("Unknown")
        unique_times = report_time.unique()
        marker_symbols = {val: sym for val, sym in zip(unique_times, ["circle", "diamond", "square", "triangle-up", "triangle-down", "star"])}
        marker_colors = {val: col for val, col in zip(unique_times, ["#228be6", "#fab005", "#40c057", "#e8590c", "#ae3ec9", "#868e96"])}
        fig = go.Figure()
        for rt in unique_times:
            mask = report_time == rt
            fig.add_trace(go.Scatter(
                x=estimate[mask],
                y=actual[mask],
                mode="markers",
                name=f"{descriptions.get('report_time', 'Report Time')}: {rt}",
                marker=dict(
                    symbol=marker_symbols[rt],
                    size=14,
                    color=marker_colors[rt],
                    line=dict(width=1, color="black")
                ),
                hovertext=[hover_text[i] for i in range(len(df)) if mask.iloc[i]],
                hoverinfo="text"
            ))
    else:
        fig = go.Figure(go.Scatter(
            x=estimate,
            y=actual,
            mode="markers",
            marker=dict(symbol="circle", size=14, color="#228be6", line=dict(width=1, color="black")),
            name="Quarter",
            hovertext=hover_text,
            hoverinfo="text"
        ))
    # 45-degree reference line
    min_val = min(estimate.min(), actual.min())
    max_val = max(estimate.max(), actual.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        name="Actual = Estimate",
        showlegend=True,
        hoverinfo="skip"
    ))
    fig.update_layout(
        title=f"{symbol.upper()} EPS: {descriptions.get('reported_eps', 'Actual EPS')} vs {descriptions.get('estimated_eps', 'Estimate EPS')} Scatter",
        xaxis_title=descriptions.get("estimated_eps", "Estimated EPS"),
        yaxis_title=descriptions.get("reported_eps", "Actual EPS"),
        legend_title=descriptions.get("report_time", "Report Time"),
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=14)
    )
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig




def plot_quarterly_revenue_net_income_vs_stock_price(symbol: str, income_df: pd.DataFrame, price_df: pd.DataFrame) -> go.Figure:
    print(f"[DEBUG] plot_quarterly_revenue_net_income_vs_stock_price called with symbol={symbol}")
    if income_df is None or price_df is None or len(income_df) == 0 or len(price_df) == 0:
        print("[DEBUG] Empty income_df or price_df.")
        return go.Figure()
    df_income = income_df.copy()
    df_price = price_df.copy()
    print(f"[DEBUG] df_income columns: {df_income.columns.tolist()}")
    print(f"[DEBUG] df_price columns: {df_price.columns.tolist()}")
    if "symbol" in df_income.columns:
        df_income = df_income[df_income["symbol"].str.upper() == symbol.upper()]
        print(f"[DEBUG] Filtered df_income for symbol={symbol}, shape={df_income.shape}")
    if "symbol" in df_price.columns:
        df_price = df_price[df_price["symbol"].str.upper() == symbol.upper()]
        print(f"[DEBUG] Filtered df_price for symbol={symbol}, shape={df_price.shape}")
    if df_income.empty or df_price.empty:
        print("[DEBUG] df_income or df_price is empty after filtering.")
        return go.Figure()
    if "fiscal_date_ending" not in df_income.columns or "total_revenue" not in df_income.columns or "net_income" not in df_income.columns:
        print("[DEBUG] Required columns missing in df_income.")
        return go.Figure()
    if "date" not in df_price.columns or "close" not in df_price.columns:
        print("[DEBUG] Required columns missing in df_price.")
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
    print(f"[DEBUG] x (fiscal_date_ending): {x.tolist()}")
    print(f"[DEBUG] revenue: {revenue.tolist()}")
    print(f"[DEBUG] net_income: {net_income.tolist()}")
    print(f"[DEBUG] close_prices: {close_prices}")
    descriptions = _get_column_descriptions("income_statement_quarterly")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=revenue,
        mode="lines+markers",
        name=descriptions.get("total_revenue", "Total Revenue"),
        line=dict(color="#228be6", width=3),
        marker=dict(symbol="circle", size=8, color="#228be6"),
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=net_income,
        mode="lines+markers",
        name=descriptions.get("net_income", "Net Income"),
        line=dict(color="#40c057", width=3),
        marker=dict(symbol="diamond", size=8, color="#40c057"),
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=close_prices,
        mode="lines+markers",
        name="Stock Price",
        line=dict(color="#fab005", width=3, dash="dash"),
        marker=dict(symbol="square", size=8, color="#fab005"),
        yaxis="y2"
    ))
    fig.update_layout(
        title=f"{symbol.upper()} Quarterly Revenue, Net Income & Stock Price",
        xaxis_title=descriptions.get("fiscal_date_ending", "Fiscal Quarter End"),
        yaxis=dict(
            title="Revenue / Net Income",
            showgrid=True,
            zeroline=True
        ),
        yaxis2=dict(
            title="Stock Price",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend_title="Metric",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=14)
    )
    print("[DEBUG] plot_quarterly_revenue_net_income_vs_stock_price finished.")
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
    pass

def plot_expense_breakdown_vs_revenue(symbol: str, income_df: pd.DataFrame) -> go.Figure:
    """
    Creates small multiple bar charts to visualize how major operating expenses compare with total revenue each quarter.

    After filtering `income_df` by `symbol`, the function extracts the series for `total_revenue` alongside key
    expense categories such as `cost_of_revenue`, `cost_of_goods_and_services_sold`,
    `selling_general_and_administrative`, `research_and_development`, and `operating_expenses`.  For each
    expense, it calculates the amount as a percentage of `total_revenue` and produces a separate bar chart
    (small multiple) showing the expense and revenue trends across `fiscal_date_ending`.  Small multiple bar charts
    are well suited to P&L trend and breakdown analysis because they allow easy comparison of different items on
    the same scale and highlight which categories drive changes:contentReference[oaicite:5]{index=5}.  Displaying expenses
    relative to revenue helps investors gauge whether costs are scaling efficiently or eating into profitability,
    providing context for how operational decisions may influence future earnings and stock performance:contentReference[oaicite:6]{index=6}.
    """
    pass

def plot_income_statement_waterfall(symbol: str, income_df: pd.DataFrame) -> go.Figure:
    """
    Builds a waterfall chart to illustrate the progression from total revenue to net income in a single quarter.

    The waterfall visualization starts with `total_revenue` and sequentially subtracts or adds intermediate line
    items from the income statement — including `cost_of_revenue`, `operating_expenses`, `investment_income_net`,
    `net_interest_income`, `other_non_operating_income`, `income_tax_expense` and `interest_and_debt_expense` —
    ending with `net_income`.  By clearly showing how each component contributes to or detracts from profits, a
    waterfall chart provides a full income statement overview:contentReference[oaicite:7]{index=7}.  It is especially useful
    for comparing how different segments or quarters flow into net income and for highlighting which line items have
    the largest impact on profitability:contentReference[oaicite:8]{index=8}.  While waterfall charts excel at explaining
    differences and relationships between items, they are less effective for showing long‑term trends:contentReference[oaicite:9]{index=9}.
    """
    pass

def plot_operating_profit_ebit_ebitda_trends(symbol: str, income_df: pd.DataFrame) -> go.Figure:
    """
    Generates a line chart comparing `operating_income`, `ebit` and `ebitda` across fiscal quarters.

    This function filters `income_df` by `symbol` and selects `fiscal_date_ending`, `operating_income`, `ebit` and
    `ebitda`.  It plots each metric as a separate line to highlight how operating profit, earnings before
    interest and taxes (EBIT) and earnings before interest, taxes, depreciation and amortization (EBITDA) evolve
    over time.  EBITDA is widely used by analysts to track underlying profitability because it adds back
    interest, tax and non‑cash expenses such as depreciation and amortization, focusing on core operations:contentReference[oaicite:10]{index=10}.
    EBIT, which includes depreciation and amortization, provides a more conservative measure of profitability, while
    `operating_income` reflects earnings from the company’s primary business before accounting for financing and
    non‑operating items.  Comparing these measures helps investors understand how changes in capital structure,
    tax rates or asset intensity affect reported earnings and assess whether improvements in operational
    performance are driving stock price movements:contentReference[oaicite:11]{index=11}.
    """
    pass

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
    pass

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
    pass

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
    pass

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
    pass