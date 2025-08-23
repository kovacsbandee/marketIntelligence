import dash
from dash import dcc, Output, Input, State, no_update
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash.development.base_component import Component

from infrastructure.ui.dash.data_service import load_symbol_data
from infrastructure.ui.dash.plots import (
    plot_price_with_indicators,
    plot_balance_sheet_time_series,
    plot_balance_sheet_stacked_area,
    plot_balance_sheet_bar,
    plot_balance_sheet_pie,
    render_balance_sheet_metric_cards,
    plot_company_fundamentals_table
)
from infrastructure.ui.dash.app_util import get_last_6_months_range


def register_callbacks(app: dash.Dash) -> None:
    """
    Register all Dash callbacks for the application.

    Args:
        app (dash.Dash): The Dash app instance.
    """

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
    def load_symbol(
        n_clicks: int,
        symbol: str
    ) -> tuple:
        """
        Callback to load all relevant data for a given symbol and update stores and UI.

        Args:
            n_clicks (int): Number of times the load button was clicked.
            symbol (str): The stock symbol entered by the user.

        Returns:
            tuple: Updated values for date pickers, status, and data stores.
        """
        if not symbol:
            return None, None, "Please enter a symbol.", None, None, None, None, None
        data = load_symbol_data(symbol)
        if data["price_df"] is None:
            return None, None, data["status_message"], None, None, None, None, None
        return (
            data["start_date"],
            data["end_date"],
            data["status_message"],
            data["price_df"].to_dict("records"),
            data["dividends_df"].to_dict("records") if data["dividends_df"] is not None else None,
            data["company_base_df"].to_dict("records")[0] if data["company_base_df"] is not None and not data["company_base_df"].empty else None,
            data["quarterly_balance_sheet_df"].to_dict("records") if data["quarterly_balance_sheet_df"] is not None and not data["quarterly_balance_sheet_df"].empty else None,
            data["annual_balance_sheet_df"].to_dict("records") if data["annual_balance_sheet_df"] is not None and not data["annual_balance_sheet_df"].empty else None,
        )

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
    def update_price_indicator(
        tab: str,
        start_date: str,
        end_date: str,
        price_data: list[dict],
        dividends_data: list[dict]
    ) -> tuple[Component, bool]:
        """
        Callback to update the price indicator panel with a plot.

        Args:
            tab (str): The currently selected tab.
            start_date (str): Selected start date.
            end_date (str): Selected end date.
            price_data (list[dict]): Price data from the store.
            dividends_data (list[dict]): Dividends data from the store.

        Returns:
            tuple: Updated content and loading state for the price indicator panel.
        """
        if tab != "price-indicator":
            return no_update, False
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
                include_rsi=True,
                include_vwap=True
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
    def update_balance_sheet(
        tab: str,
        start_date: str,
        end_date: str,
        q_balance_data: list[dict]
    ) -> tuple[Component, bool]:
        """
        Callback to update the balance sheet panel with a time series plot.

        Args:
            tab (str): The currently selected tab.
            start_date (str): Selected start date.
            end_date (str): Selected end date.
            q_balance_data (list[dict]): Quarterly balance sheet data from the store.

        Returns:
            tuple: Updated content and loading state for the balance sheet panel.
        """
        if tab != "balance-sheet":
            return no_update, False
        if not q_balance_data:
            return dmc.Text("No balance sheet data loaded.", c="red"), False

        df = pd.DataFrame(q_balance_data)
        if df.empty or "fiscal_date_ending" not in df.columns:
            return dmc.Text("No balance sheet data loaded.", c="red"), False

        df["fiscal_date_ending"] = pd.to_datetime(df["fiscal_date_ending"])
        df = df.sort_values("fiscal_date_ending")

        if start_date and end_date:
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
                        "long_term_debt"  # Only if present in your data
                    ]
                }
            }

            bar_groups = {
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

            pie_columns = [
                "cash_and_cash_equivalents",
                "inventory",
                "current_net_receivables",
                "property_plant_equipment",
                "goodwill"
            ]

            metrics = [
                "total_assets",
                "total_liabilities",
                "total_shareholder_equity",
                "total_current_assets",
                "total_current_liabilities",
                "cash_and_cash_equivalents",
                "property_plant_equipment"
            ]
            # Use the most recent date for pie/cards
            latest_date = str(selected_df["fiscal_date_ending"].max().date())

            # Compose all plots and cards
            time_series_fig = plot_balance_sheet_time_series(selected_df, columns=metrics)
            stacked_area_fig = plot_balance_sheet_stacked_area(selected_df, stack_groups)
            bar_fig = plot_balance_sheet_bar(selected_df, bar_groups)
            pie_fig = plot_balance_sheet_pie(selected_df, latest_date, pie_columns)
            cards = render_balance_sheet_metric_cards(selected_df, latest_date, metrics)

            return dmc.Stack([
                dmc.Group(cards, justify="flex-start", gap="xs"),
                dmc.Divider(label="Time Series", my=10),
                dcc.Graph(figure=time_series_fig),
                dmc.Divider(label="Stacked Area", my=10),
                dcc.Graph(
                    figure=stacked_area_fig,
                    style={"width": "100vw", "minWidth": "1200px", "height": "700px"},
                    config={"responsive": True}
                ),
                dmc.Divider(label="Bar Chart", my=10),
                dcc.Graph(figure=bar_fig),
                dmc.Divider(label="Pie Chart", my=10),
                dcc.Graph(figure=pie_fig),
            ], gap=16), False
        except Exception as e:
            return dmc.Text(f"Error generating plot: {e}", c="red"), False
        
    @app.callback(
        Output("company-base-content", "children"),
        Output("company-base-loading", "visible"),
        Input("main-tabs", "value"),
        Input("company-base-store", "data"),
        prevent_initial_call=True
    )
    def update_company_base(
        tab: str,
        company_base_data: dict
    ) -> tuple[Component, bool]:
        """
        Callback to update the company base info panel with a Plotly table.

        Args:
            tab (str): The currently selected tab.
            company_base_data (dict): The company base data from the store.

        Returns:
            tuple: Updated content and loading state for the company base panel.
        """
        if tab != "company-base":
            return no_update, False
        if not company_base_data:
            return dmc.Text("No company info loaded.", c="red"), False

        try:
            # Convert dict to DataFrame for compatibility with the plotting function
            fundamentals_df = pd.DataFrame([company_base_data])
            symbol = company_base_data.get("symbol", "")
            fig = plot_company_fundamentals_table(fundamentals_df, symbol)
            return dcc.Graph(figure=fig, config={"displayModeBar": False}), False
        except Exception as e:
            return dmc.Text(f"Error displaying company fundamentals: {e}", c="red"), False