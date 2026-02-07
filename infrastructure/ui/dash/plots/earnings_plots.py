import pandas as pd
import plotly.graph_objects as go
from infrastructure.ui.dash.plot_utils import (
    _get_column_descriptions, 
    DEFAULT_PLOTLY_WIDTH, 
    DEFAULT_PLOTLY_HEIGHT
)

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
        name="Actual",
        line=dict(color="#228be6", width=3),
        marker=dict(symbol="circle", size=8, color="#228be6"),
        hovertemplate=f"<b>{descriptions.get('reported_eps', 'Actual EPS')}</b><br>{descriptions.get('fiscal_date_ending', 'Date')}: "+"%{x|%Y-%m-%d}<br>"+f"{descriptions.get('reported_eps', 'EPS')}: "+"%{y:.2f}<extra></extra>"
    ))
    # Estimate EPS line
    fig.add_trace(go.Scatter(
        x=x,
        y=estimate,
        mode="lines+markers",
        name="Estimate",
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
        title="EPS",
        xaxis_title="Date",
        yaxis_title="",
        legend_title="",
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
        name="Surprise %",
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
        title="Surprise %",
        xaxis_title="Date",
        yaxis_title="",
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
                name=str(rt),
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
        name="Parity",
        showlegend=True,
        hoverinfo="skip"
    ))
    fig.update_layout(
        title="EPS Scatter",
        xaxis_title="Estimate",
        yaxis_title="",
        legend_title="",
        template="plotly_white",
        width=DEFAULT_PLOTLY_WIDTH,
        height=DEFAULT_PLOTLY_HEIGHT,
        margin=dict(l=50, r=50, t=60, b=50),
        font=dict(size=14)
    )
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
