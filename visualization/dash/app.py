"""
Dash web application for visualizing stock price data with technical indicators.

This app allows users to input a stock symbol, loads the corresponding data from the database
using DBSymbolStorage, and displays a candlestick chart with optional MACD and RSI indicators.
A status message is shown to inform the user about the data loading process.
"""

import dash
from dash import dcc, Input, Output, State
import dash_mantine_components as dmc
from datetime import datetime, timedelta
from data_manager.src_postgre_db.db_etl_jobs.db_load_from_db_runner import DBSymbolStorage
from data_manager.src_postgre_db.db_infrastructure.postgre_adapter import PostgresAdapter
from visualization.base_plots import plot_price_with_indicators
from visualization.dash.app_util import get_last_6_months_range
import pandas as pd

app = dash.Dash(
    __name__,
    external_stylesheets=["https://unpkg.com/@mantine/ds@latest/styles.css"]
)

app.layout = dmc.MantineProvider(
    dmc.Paper([
        dmc.Text("Stock Price with Indicators", size="xl", fw=700, mb=20),
        dmc.TextInput(id="symbol-input", placeholder="Enter symbol (e.g. MSFT)", debounce=True, style={"width": 300}),
        dmc.Group([
            dmc.Stack([
                dmc.Text("Start date", size="sm", mb=2, fw=700),
                dmc.DatePicker(
                    id="start-date-picker",
                    style={"width": 140, "marginTop": 0},
                    allowDeselect=True
                ),
            ], gap=4),
            dmc.Stack([
                dmc.Text("End date", size="sm", mb=2, fw=700),
                dmc.DatePicker(
                    id="end-date-picker",
                    style={"width": 140, "marginTop": 0},
                    allowDeselect=True
                ),
            ], gap=4),
            dmc.Button("Load", id="load-btn", n_clicks=0, mt=22, style={"marginLeft": 60}),
        ], gap=140),
        dmc.Notification(id="status-div", message="", color="blue", style={"margin": "10px 0"}, withCloseButton=False, action=None),
        dmc.Paper(id="output-div", mt=20),
        dcc.Store(id="price-store"),
        dcc.Store(id="dividends-store")
    ], p=30, shadow="md")
)

@app.callback(
    Output("start-date-picker", "value"),
    Output("end-date-picker", "value"),
    Output("status-div", "title"),
    Output("price-store", "data"),
    Output("dividends-store", "data"),
    Input("load-btn", "n_clicks"),
    State("symbol-input", "value"),
    prevent_initial_call=True
)
def load_symbol(n_clicks, symbol):
    """
    Callback to load stock data for the given symbol.

    Args:
        n_clicks (int): Number of times the Load button has been clicked.
        symbol (str): The stock symbol entered by the user.

    Returns:
        Tuple[str, dict, dict]:
            - Default start date for the DatePicker.
            - Default end date for the DatePicker.
            - Status message about the data loading process.
            - Full price data as a dictionary for dcc.Store.
            - Full dividends data as a dictionary for dcc.Store, or None if no data.
    """
    if not symbol:
        return None, None, "Please enter a symbol.", None, None
    adapter = PostgresAdapter()
    storage = DBSymbolStorage(adapter, symbol.upper())
    price_df = storage.get_table("daily_timeseries")
    if price_df is None or price_df.empty:
        return None, None, storage.status_message, None, None
    dividends_df = storage.get_table("dividends")
    start_date, end_date = get_last_6_months_range(price_df)
    # Convert to dict for dcc.Store
    return start_date, end_date, storage.status_message, price_df.to_dict("records"), dividends_df.to_dict("records") if dividends_df is not None else None

@app.callback(
    Output("output-div", "children"),
    Input("start-date-picker", "value"),
    Input("end-date-picker", "value"),
    Input("price-store", "data"),
    Input("dividends-store", "data"),
    prevent_initial_call=True
)
def update_plot(start_date, end_date, price_data, dividends_data):
    """
    Callback to update the plot based on the selected date range.

    Args:
        start_date (str): Selected start date from the DatePicker.
        end_date (str): Selected end date from the DatePicker.
        price_data (list): Full price data stored in dcc.Store.
        dividends_data (list): Full dividends data stored in dcc.Store.

    Returns:
        dash.development.base_component.Component:
            - Plotly graph component with the price and indicators, or None if no data.
    """
    if not price_data:
        return None
    price_df = pd.DataFrame(price_data)
    dividends_df = pd.DataFrame(dividends_data) if dividends_data else None

    # Always default to last 6 months if not selected
    if not start_date or not end_date:
        start_date, end_date = get_last_6_months_range(price_df)
    date_range = [start_date, end_date]

    mask = (price_df["date"] >= date_range[0]) & (price_df["date"] <= date_range[1])
    selected_df = price_df.loc[mask]
    if dividends_df is not None and not dividends_df.empty and "ex_dividend_date" in dividends_df.columns:
        div_mask = (dividends_df["ex_dividend_date"] >= date_range[0]) & (dividends_df["ex_dividend_date"] <= date_range[1])
        dividends_df = dividends_df.loc[div_mask]

    try:
        fig = plot_price_with_indicators(
            price_table=selected_df,
            dividend_table=dividends_df,
            include_macd=True,
            include_rsi=True
        )

        # Set x-axis range to selected date range for zoom
        fig.update_layout(xaxis_range=[start_date, end_date])

        return dcc.Graph(figure=fig)
    except Exception as e:
        print(f"Error in update_plot: {e}")
        return dmc.Text(f"Error generating plot: {e}", color="red")

if __name__ == "__main__":
    app.run(debug=True)
