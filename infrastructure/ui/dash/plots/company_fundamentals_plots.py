import pandas as pd
import plotly.graph_objects as go
from infrastructure.ui.dash.plot_utils import DEFAULT_PLOTLY_WIDTH, DEFAULT_PLOTLY_HEIGHT

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
