"""
Dash web application for visualizing stock price data with technical indicators.

This app allows users to input a stock symbol, loads the corresponding data from the database
using DBSymbolStorage, and displays a candlestick chart with optional MACD and RSI indicators.
A status message is shown to inform the user about the data loading process.
"""


import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_mantine_components as dmc
from datetime import datetime, timedelta
from symbol.symbol import Symbol
#from data_manager.src_postgre_db.db_etl_jobs.db_load_from_db_runner import DBSymbolStorage
from infrastructure.databases.company.postgre_manager.postgre_manager import CompanyDataManager
#from data_manager.src_postgre_db.db_infrastructure.postgre_adapter import PostgresAdapter
from infrastructure.ui.dash.plots import plot_price_with_indicators
from infrastructure.ui.dash.plots import plot_balance_sheet_time_series
#from visualization.base_plots import plot_price_with_indicators
from infrastructure.ui.dash.app_util import get_last_6_months_range
#from visualization.dash.app_util import get_last_6_months_range
import pandas as pd
import plotly.graph_objects as go

app = dash.Dash(
    __name__,
    external_stylesheets=["https://unpkg.com/@mantine/ds@latest/styles.css"]
)

# --- Mantine Default Theme ---
mantine_theme = {
    "colorScheme": "light",
    "primaryColor": "blue",
    "fontFamily": "Inter, sans-serif",
}

app.layout = dmc.MantineProvider(
    theme=mantine_theme,
    children=dmc.Paper([
        dmc.Text("Market Intelligence Dashboard", size="xl", fw=700, mb=20),
        # Global Controls
        dmc.Group([
            dmc.Stack([
                dmc.TextInput(
                    id="symbol-input",
                    placeholder="Enter symbol (e.g. MSFT)",
                    debounce=True,
                    style={"width": 300}
                ),
                dmc.Button(
                    "Load",
                    id="load-btn",
                    n_clicks=0,
                    style={"width": 100, "marginTop": 8}  # Adjust marginTop as needed for alignment
                ),
            ], gap=8),
            dmc.Stack([
                dmc.Text("Start date", size="sm", mb=2, fw=700),
                dmc.DatePicker(id="start-date-picker", style={"width": 140, "marginTop": 0}, allowDeselect=True),
            ], gap=4, style={"marginRight": 80}),
            dmc.Stack([
                dmc.Text("End date", size="sm", mb=2, fw=700),
                dmc.DatePicker(id="end-date-picker", style={"width": 140, "marginTop": 0}, allowDeselect=True),
            ], gap=4),
        ], gap=40, align="flex-start"),
        dmc.Notification(id="status-div", message="", color="blue", style={"margin": "10px 0"}, withCloseButton=False, action=None),
        # Tabs on the left
        dmc.Group([
            dmc.Paper(
                dmc.Tabs(
                    id="main-tabs",
                    value="company-base",
                    orientation="vertical",
                    children=[
                        dmc.TabsList([
                            dmc.TabsTab("Company Base", value="company-base"),
                            dmc.TabsTab("Price Indicator", value="price-indicator"),
                            dmc.TabsTab("Earnings", value="earnings"),
                            dmc.TabsTab("Income Statement", value="income-statement"),
                            dmc.TabsTab("Balance Sheet", value="balance-sheet"),
                            dmc.TabsTab("Cash Flow", value="cash-flow"),
                            dmc.TabsTab("Insider Transactions", value="insider-transactions"),
                        ]),
                        dmc.TabsPanel(value="company-base", children=[
                            dmc.Box([
                                dmc.Paper(id="company-base-content", p=10, children=[
                                    dmc.Text("Company base info will appear here.", c="dimmed")
                                ]),
                                dmc.LoadingOverlay(
                                    id="company-base-loading",
                                    visible=False,
                                ),
                            ])
                        ]),
                        dmc.TabsPanel(value="price-indicator", children=[
                            dmc.Box([
                                dmc.Paper(id="price-indicator-content", p=10),
                                dmc.LoadingOverlay(
                                    id="price-indicator-loading",
                                    visible=False,
                                ),
                            ])
                        ]),
                        dmc.TabsPanel(value="earnings", children=[
                            dmc.Box([
                                dmc.Paper(id="earnings-content", p=10, children=[
                                    dmc.Text("Earnings info will appear here.", c="dimmed")
                                ]),
                                dmc.LoadingOverlay(
                                    id="earnings-loading",
                                    visible=False,
                                ),
                            ])
                        ]),
                        dmc.TabsPanel(value="income-statement", children=[
                            dmc.Box([
                                dmc.Paper(id="income-statement-content", p=10, children=[
                                    dmc.Text("Income statement info will appear here.", c="dimmed")
                                ]),
                                dmc.LoadingOverlay(
                                    id="income-statement-loading",
                                    visible=False,
                                ),
                            ])
                        ]),
                        dmc.TabsPanel(value="balance-sheet", children=[
                            dmc.Box([
                                dmc.Paper(id="balance-sheet-content", p=10, children=[
                                    dmc.Text("Balance sheet info will appear here.", c="dimmed")
                                ]),
                                dmc.LoadingOverlay(
                                    id="balance-sheet-loading",
                                    visible=False,
                                ),
                            ])
                        ]),
                        dmc.TabsPanel(value="cash-flow", children=[
                            dmc.Box([
                                dmc.Paper(id="cash-flow-content", p=10, children=[
                                    dmc.Text("Cash flow info will appear here.", c="dimmed")
                                ]),
                                dmc.LoadingOverlay(
                                    id="cash-flow-loading",
                                    visible=False,
                                ),
                            ])
                        ]),
                        dmc.TabsPanel(value="insider-transactions", children=[
                            dmc.Box([
                                dmc.Paper(id="insider-transactions-content", p=10, children=[
                                    dmc.Text("Insider transactions info will appear here.", c="dimmed")
                                ]),
                                dmc.LoadingOverlay(
                                    id="insider-transactions-loading",
                                    visible=False,
                                ),
                            ])
                        ]),
                    ],
                    style={"minWidth": 300, "height": 600}
                ),
                shadow="sm",
                p=0,
                style={"minWidth": 350, "maxWidth": 400}
            ),
        ], style={"marginTop": 30, "justifyContent": "flex-start"}),
        dcc.Store(id="price-store"),
        dcc.Store(id="dividends-store"),
        dcc.Store(id="company-base-store"),
        dcc.Store(id="q_balance-store"),
        dcc.Store(id="a_balance-store"),
    ], p=30, shadow="md")
)

# --- Callbacks for loading data and updating tabs ---

@app.callback(
    Output("start-date-picker", "value"),
    Output("end-date-picker", "value"),
    Output("status-div", "title"),
    Output("price-store", "data"),
    Output("dividends-store", "data"),
    Output("company-base-store", "data"),
    Output("q_balance-store", "data"),
    Output("a_balance-store", "data"),
    Input("load-btn", "n_clicks"),
    State("symbol-input", "value"),
    prevent_initial_call=True
)
def load_symbol(n_clicks, symbol):
    if not symbol:
        return None, None, "Please enter a symbol.", None, None, None
    adapter = CompanyDataManager()
    storage = Symbol(adapter, symbol.upper())
    price_df = storage.get_table("daily_timeseries")
    if price_df is None or price_df.empty:
        return None, None, storage.status_message, None, None, None
    dividends_df = storage.get_table("dividends")
    company_base_df = storage.get_table("company_fundamentals")
    annual_balance_sheet_df = storage.get_table("annual_balance_sheet")
    quarterly_balance_sheet_df = storage.get_table("quarterly_balance_sheet")
    start_date, end_date = get_last_6_months_range(price_df)
    return (
        start_date,
        end_date,
        storage.status_message,
        price_df.to_dict("records"),
        dividends_df.to_dict("records") if dividends_df is not None else None,
        company_base_df.to_dict("records")[0] if company_base_df is not None and not company_base_df.empty else None,
        quarterly_balance_sheet_df.to_dict("records") if quarterly_balance_sheet_df is not None and not quarterly_balance_sheet_df.empty else None,
        annual_balance_sheet_df.to_dict("records") if annual_balance_sheet_df is not None and not annual_balance_sheet_df.empty else None,
    )

@app.callback(
    Output("company-base-content", "children"),
    Output("company-base-loading", "visible"),
    Input("main-tabs", "value"),
    Input("company-base-store", "data"),
    prevent_initial_call=True
)
def update_company_base(tab, company_base_data):
    if tab != "company-base":
        return dash.no_update, False
    if not company_base_data:
        return dmc.Text("No company info loaded.", c="red"), False
    # Placeholder: Replace with actual company info rendering
    return dmc.Text(f"Company: {company_base_data.get('name', 'N/A')}"), False

@app.callback(
    Output("price-indicator-content", "children"),
    Output("price-indicator-loading", "visible"),
    Input("main-tabs", "value"),
    Input("start-date-picker", "value"),
    Input("end-date-picker", "value"),
    Input("price-store", "data"),
    Input("dividends-store", "data"),
    prevent_initial_call=True
)
def update_price_indicator(tab, start_date, end_date, price_data, dividends_data):
    if tab != "price-indicator":
        return dash.no_update, False
    if not price_data:
        return dmc.Text("No price data loaded.", c="red"), False
    price_df = pd.DataFrame(price_data)
    dividends_df = pd.DataFrame(dividends_data) if dividends_data else None
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
            include_macd=True,
            include_rsi=True
        )
        fig.update_layout(xaxis_range=[start_date, end_date])
        return dcc.Graph(figure=fig), False
    except Exception as e:
        return dmc.Text(f"Error generating plot: {e}", c="red"), False

@app.callback(
    Output("balance-sheet-content", "children"),
    Output("balance-sheet-loading", "visible"),
    Input("main-tabs", "value"),
    Input("start-date-picker", "value"),
    Input("end-date-picker", "value"),
    Input("q_balance-store", "data"),
    prevent_initial_call=True
)
def update_balance_sheet(tab, start_date, end_date, q_balance_data):
    if tab != "balance-sheet":
        return dash.no_update, False
    if not q_balance_data:
        return dmc.Text("No balance sheet data loaded.", c="red"), False

    df = pd.DataFrame(q_balance_data)
    if df.empty or "fiscal_date_ending" not in df.columns:
        return dmc.Text("No balance sheet data loaded.", c="red"), False

    # Ensure fiscal_date_ending is datetime and sort
    df["fiscal_date_ending"] = pd.to_datetime(df["fiscal_date_ending"])
    df = df.sort_values("fiscal_date_ending")

    # Date filtering (if dates are provided)
    if start_date and end_date:
        # Convert picker values to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        mask = (df["fiscal_date_ending"] >= start_dt) & (df["fiscal_date_ending"] <= end_dt)
        selected_df = df.loc[mask]
    else:
        selected_df = df

    metrics = [
        "total_assets",
        "total_liabilities",
        "total_shareholder_equity",
        "total_current_assets",
        "total_current_liabilities",
        "cash_and_cash_equivalents",
        "property_plant_equipment"
    ]

    if selected_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No balance sheet data available for the selected date range.",
            xaxis_title="Fiscal Date Ending",
            yaxis_title="Value"
        )
        return dcc.Graph(figure=fig), False

    try:
        fig = plot_balance_sheet_time_series(
            balance_df=selected_df,
            columns=metrics
        )
        if start_date and end_date:
            fig.update_layout(xaxis_range=[start_date, end_date])
        return dcc.Graph(figure=fig), False
    except Exception as e:
        return dmc.Text(f"Error generating plot: {e}", c="red"), False

if __name__ == "__main__":
    app.run(debug=True)
