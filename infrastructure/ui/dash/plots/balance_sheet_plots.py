import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
import dash_mantine_components as dmc


from infrastructure.ui.dash.plot_utils import _find_row_by_date, _flatten_group, _auto_load_table_descriptions, DEFAULT_PLOTLY_WIDTH, DEFAULT_PLOTLY_HEIGHT

def plot_balance_sheet_time_series(
    balance_df: pd.DataFrame,
    columns: list = None,
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
    # Set default columns if not provided
    if columns is None:
        columns = [
            "total_assets",
            "total_liabilities",
            "total_shareholder_equity",
            "total_current_assets",
            "total_current_liabilities",
            "cash_and_cash_equivalents",
            "property_plant_equipment"
        ]
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


def plot_balance_sheet_stacked_area(balance_df: pd.DataFrame, stack_groups: dict = None, descriptions: dict = None) -> go.Figure:
    """
    Plot stacked area chart to visualize the composition of assets or liabilities over time.
    Supports nested grouping.
    """
    if stack_groups is None:
        stack_groups = {
            "Assets": {
                "Current Assets": [
                    "cash_and_cash_equivalents",
                    "inventory",
                    "current_net_receivables"
                ],
                "Non-Current Assets": [
                    "property_plant_equipment",
                    "goodwill"
                ]
            },
            "Liabilities": {
                "Current Liabilities": [
                    "total_current_liabilities"
                ],
                "Non-Current Liabilities": [
                    "long_term_debt"
                ]
            }
        }
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


def plot_balance_sheet_bar(balance_df: pd.DataFrame, group_columns: dict = None, descriptions: dict = None) -> go.Figure:
    """
    Plot grouped or stacked bar chart comparing current vs. non-current assets and liabilities.
    Supports nested grouping.
    """
    if group_columns is None:
        group_columns = {
            "Current Assets": [
                "cash_and_cash_equivalents",
                "inventory",
                "current_net_receivables"
            ],
            "Non-Current Assets": [
                "property_plant_equipment",
                "goodwill"
            ],
            "Current Liabilities": [
                "total_current_liabilities"
            ],
            "Shareholder Equity": [
                "total_shareholder_equity"
            ]
        }
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


def plot_balance_sheet_pie(balance_df: pd.DataFrame, date: str, columns: list = None, descriptions: dict = None) -> go.Figure:
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
    if columns is None:
        columns = [
            "cash_and_cash_equivalents",
            "inventory",
            "current_net_receivables",
            "property_plant_equipment",
            "goodwill"
        ]
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


def render_balance_sheet_metric_cards(balance_df: pd.DataFrame, date: str, metrics: list = None, descriptions: dict = None):
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
    if metrics is None:
        metrics = [
            "total_assets",
            "total_liabilities",
            "total_shareholder_equity",
            "total_current_assets",
            "total_current_liabilities",
            "cash_and_cash_equivalents",
            "property_plant_equipment"
        ]
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
