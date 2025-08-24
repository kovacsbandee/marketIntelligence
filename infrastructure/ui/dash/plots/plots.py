import pandas as pd
import plotly.graph_objects as go
import json
import os
from plotly.subplots import make_subplots
from plotly.colors import qualitative

# --- Default Plotly Figure Size ---


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
















