"""
analysis_gauge.py

Visualization helpers for the QuantitativeAnalyst indicator scores.

Provides:
- ``plot_score_gauge``: A half-circle gauge (speedometer) chart for a
  single indicator score in [-1, 1].
- ``build_indicator_score_card``: A Dash Mantine card combining a badge
  with the numeric score, a gauge chart, and an inline explanation.
- ``build_aggregate_score_card``: A prominent banner for the aggregate
  score.
- ``score_color``: Maps a [-1, 1] score to a hex colour.
"""

from __future__ import annotations

import plotly.graph_objects as go
import dash_mantine_components as dmc
from dash import dcc, html


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_COLOUR_STOPS = [
    (-1.0, "#e03131"),   # strong red   – underpriced / sell signal
    (-0.5, "#fa5252"),   # light red
    ( 0.0, "#868e96"),   # neutral grey
    ( 0.5, "#40c057"),   # light green
    ( 1.0, "#2b8a3e"),   # strong green – overpriced / buy signal
]


def score_color(score: float | None) -> str:
    """Return a hex colour for a score in [-1, 1]."""
    if score is None:
        return "#868e96"
    s = max(min(score, 1.0), -1.0)
    # Piece-wise linear interpolation
    for i in range(len(_COLOUR_STOPS) - 1):
        lo_val, lo_col = _COLOUR_STOPS[i]
        hi_val, hi_col = _COLOUR_STOPS[i + 1]
        if lo_val <= s <= hi_val:
            t = (s - lo_val) / (hi_val - lo_val) if hi_val != lo_val else 0.5
            return _lerp_hex(lo_col, hi_col, t)
    return "#868e96"


def _lerp_hex(c1: str, c2: str, t: float) -> str:
    """Linearly interpolate between two hex colours."""
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _score_label(score: float | None) -> str:
    """Human-readable label for a score."""
    if score is None:
        return "N/A"
    if score <= -0.6:
        return "Strong Under"
    if score <= -0.2:
        return "Under"
    if score < 0.2:
        return "Neutral"
    if score < 0.6:
        return "Over"
    return "Strong Over"


# ---------------------------------------------------------------------------
# Gauge chart
# ---------------------------------------------------------------------------

def plot_score_gauge(
    score: float | None,
    title: str = "",
    width: int = 220,
    height: int = 140,
) -> go.Figure:
    """Create a half-circle gauge chart for an indicator score.

    Args:
        score: Value in [-1, 1] (or None for a greyed-out gauge).
        title: Indicator name shown as the gauge title.
        width: Figure width in px.
        height: Figure height in px.

    Returns:
        Plotly Figure with a single ``Indicator`` trace.
    """
    display_value = score if score is not None else 0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=display_value,
            number={"suffix": "", "valueformat": "+.2f", "font": {"size": 18}},
            gauge={
                "axis": {"range": [-1, 1], "tickwidth": 1, "dtick": 0.5},
                "bar": {"color": score_color(score), "thickness": 0.6},
                "bgcolor": "#f1f3f5",
                "borderwidth": 1,
                "bordercolor": "#dee2e6",
                "steps": [
                    {"range": [-1, -0.5], "color": "#ffe3e3"},
                    {"range": [-0.5, 0], "color": "#fff4e6"},
                    {"range": [0, 0.5], "color": "#e6fcf5"},
                    {"range": [0.5, 1], "color": "#d3f9d8"},
                ],
                "threshold": {
                    "line": {"color": "#212529", "width": 3},
                    "thickness": 0.8,
                    "value": display_value,
                },
            },
            title={"text": title, "font": {"size": 13}},
        )
    )
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Dash component builders
# ---------------------------------------------------------------------------

def build_indicator_score_card(
    indicator_name: str,
    score: float | None,
    explanation: str = "",
    display_name: str | None = None,
) -> dmc.Paper:
    """Return a compact Mantine card with badge, gauge, and explanation.

    Args:
        indicator_name: Machine name (e.g. ``"rsi"``).
        score: Score in [-1, 1] or None.
        explanation: LLM explanation text.
        display_name: Optional human label; defaults to *indicator_name* title-cased.

    Returns:
        ``dmc.Paper`` component.
    """
    label = display_name or indicator_name.replace("_", " ").title()
    color = score_color(score)
    score_text = f"{score:+.2f}" if score is not None else "…"
    fig = plot_score_gauge(score, title="")

    badge = dmc.Badge(
        score_text,
        color=color,
        variant="filled",
        size="lg",
        radius="xl",
        style={"minWidth": 72, "fontWeight": 700},
    )
    sentiment = dmc.Text(_score_label(score), size="xs", c="dimmed", ta="center")

    gauge = dcc.Graph(
        figure=fig,
        config={"displayModeBar": False, "staticPlot": True},
        style={"height": 130, "width": 210},
    )

    explanation_block = dmc.Text(
        explanation or "Waiting for analysis…",
        size="xs",
        c="dimmed",
        style={"whiteSpace": "pre-wrap", "lineHeight": 1.4},
    )

    return dmc.Paper(
        shadow="xs",
        radius="md",
        p="sm",
        withBorder=True,
        style={"minWidth": 230, "maxWidth": 340, "flex": "1 1 280px"},
        children=[
            dmc.Stack([
                dmc.Group([
                    dmc.Text(label, fw=600, size="sm"),
                    badge,
                ], justify="space-between", align="center"),
                dmc.Center(gauge),
                sentiment,
                dmc.Divider(variant="dashed", my=4),
                explanation_block,
            ], gap=4),
        ],
    )


def build_aggregate_score_card(
    aggregate_score: float | None,
    analysis_date: str | None = None,
    symbol: str | None = None,
    n_indicators: int = 0,
) -> dmc.Paper:
    """Build a prominent aggregate-score banner.

    Args:
        aggregate_score: Mean score of all indicators.
        analysis_date: Human-readable date string.
        symbol: Ticker symbol.
        n_indicators: Number of indicators evaluated.

    Returns:
        ``dmc.Paper`` component.
    """
    color = score_color(aggregate_score)
    score_text = f"{aggregate_score:+.2f}" if aggregate_score is not None else "…"
    fig = plot_score_gauge(aggregate_score, title="Aggregate", width=260, height=160)

    subtitle_parts = []
    if symbol:
        subtitle_parts.append(symbol.upper())
    if analysis_date:
        subtitle_parts.append(f"as of {analysis_date}")
    if n_indicators:
        subtitle_parts.append(f"{n_indicators} indicators")
    subtitle = " · ".join(subtitle_parts)

    return dmc.Paper(
        shadow="sm",
        radius="lg",
        p="md",
        withBorder=True,
        style={
            "background": f"linear-gradient(135deg, {color}11,  {color}22)",
            "borderColor": color,
        },
        children=[
            dmc.Group([
                dmc.Stack([
                    dmc.Text("Quantitative Analysis", fw=700, size="lg"),
                    dmc.Text(subtitle, size="sm", c="dimmed") if subtitle else None,
                    dmc.Group([
                        dmc.Badge(
                            score_text,
                            color=color,
                            variant="filled",
                            size="xl",
                            radius="xl",
                            style={"fontSize": 22, "minWidth": 90, "fontWeight": 700},
                        ),
                        dmc.Text(_score_label(aggregate_score), size="sm", c="dimmed"),
                    ], gap=12, align="center"),
                ], gap=6),
                dcc.Graph(
                    figure=fig,
                    config={"displayModeBar": False, "staticPlot": True},
                    style={"height": 150, "width": 250},
                ),
            ], justify="space-between", align="center", wrap="wrap"),
        ],
    )


def build_placeholder_score_card(indicator_name: str, display_name: str | None = None) -> dmc.Paper:
    """Grey placeholder card shown while an indicator is being evaluated."""
    return build_indicator_score_card(
        indicator_name=indicator_name,
        score=None,
        explanation="Waiting for analysis…",
        display_name=display_name,
    )


def build_placeholder_aggregate_card() -> dmc.Paper:
    """Grey placeholder aggregate card."""
    return build_aggregate_score_card(aggregate_score=None)
