import dash
from dash import dcc, Output, Input, State, no_update
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash.development.base_component import Component
from typing import Any

from infrastructure.ui.dash.data_service import load_symbol_data
from infrastructure.ui.dash.plots import (
    plot_price_with_indicators,
    plot_balance_sheet_time_series,
    plot_balance_sheet_stacked_area,
    plot_balance_sheet_bar,
    plot_balance_sheet_pie,
    render_balance_sheet_metric_cards,
    plot_company_fundamentals_table,
    plot_quarterly_revenue_net_income_vs_stock_price,
)
from infrastructure.ui.dash.app_util import get_last_6_months_range


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("earnings-content", "children"),
        Output("earnings-loading", "visible"),
        Input("main-tabs", "value"),
        Input("earnings-store", "data"),
        prevent_initial_call=True
    )
    def update_earnings_panel(tab: str, earnings_data: list[dict] | None):
        if tab != "earnings":
            return no_update, False
        if earnings_data is None or len(earnings_data) == 0:
            return dmc.Text("No earnings data loaded.", c="red"), False
        try:
            from infrastructure.ui.dash.plots import (
                plot_eps_actual_vs_estimate,
                plot_eps_surprise_percentage,
                plot_eps_actual_vs_estimate_scatter
            )
            earnings_df = pd.DataFrame(earnings_data)
            symbol = earnings_df["symbol"].iloc[0] if "symbol" in earnings_df.columns and len(earnings_df) > 0 else ""
            fig_eps_vs_est = plot_eps_actual_vs_estimate(symbol, earnings_df)
            fig_surprise_pct = plot_eps_surprise_percentage(symbol, earnings_df)
            fig_eps_scatter = plot_eps_actual_vs_estimate_scatter(symbol, earnings_df)
            return dmc.Stack([
                dmc.Divider(label="EPS Actual vs Estimate", my=10),
                dcc.Graph(figure=fig_eps_vs_est),
                dmc.Divider(label="EPS Surprise Percentage", my=10),
                dcc.Graph(figure=fig_surprise_pct),
                dmc.Divider(label="EPS Actual vs Estimate Scatter", my=10),
                dcc.Graph(figure=fig_eps_scatter),
            ], gap=16), False
        except Exception as e:
            return dmc.Text(f"Error displaying earnings plots: {e}", c="red"), False
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
        Output("earnings-store", "data"),
        Output("q_income-store", "data"),
        Input("load-btn", "n_clicks"),
        State("symbol-input", "value"),
        prevent_initial_call=True
    )
    def load_symbol(
        n_clicks: int,
        symbol: str
    ) -> tuple:
        """
        Callback to load all relevant data for a given symbol and update stores and UI, including earnings data.
        """
        if not symbol:
            return None, None, "Please enter a symbol.", None, None, None, None, None, None
        data = load_symbol_data(symbol)
        if data["daily_timeseries"] is None:
            return None, None, data["status_message"], None, None, None, None, None, None

        def df_to_records(df):
            if df is None:
                return None
            if hasattr(df, 'empty') and df.empty:
                return None
            return df.to_dict("records")

        earnings = data.get("earnings") if "earnings" in data else None
        return (
            data["start_date"],
            data["end_date"],
            data["status_message"],
            df_to_records(data["daily_timeseries"]),
            df_to_records(data["dividends"]),
            df_to_records(data["company_fundamentals"]),
            df_to_records(data["balance_sheet_quarterly"]),
            df_to_records(data["annual_balance_sheet"]),
            df_to_records(earnings),
            df_to_records(data.get("income_statement_quarterly") if "income_statement_quarterly" in data else None),
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
        daily_timeseries: list[dict] | None,
        dividends: list[dict] | None
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
        if daily_timeseries is None or len(daily_timeseries) == 0:
            return dmc.Text("No price data loaded.", c="red"), False
        # Convert to DataFrame if needed
        daily_timeseries_df = pd.DataFrame(daily_timeseries)
        dividends_df = pd.DataFrame(dividends) if dividends is not None else None
        if not start_date or not end_date:
            start_date, end_date = get_last_6_months_range(daily_timeseries_df)
        date_range = [start_date, end_date]
        mask = (daily_timeseries_df["date"] >= date_range[0]) & (daily_timeseries_df["date"] <= date_range[1])
        selected_timeseries = daily_timeseries_df.loc[mask]
        if dividends_df is not None and not dividends_df.empty and "ex_dividend_date" in dividends_df.columns:
            div_mask = (dividends_df["ex_dividend_date"] >= date_range[0]) & (dividends_df["ex_dividend_date"] <= date_range[1])
            dividends_df = dividends_df.loc[div_mask]
        try:
            fig = plot_price_with_indicators(
                price_table=selected_timeseries,
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
        balance_sheet_quarterly: list[dict] | None
    ) -> tuple[Component, bool]:
        """
        Callback to update the balance sheet panel with a time series plot.

        Args:
            tab (str): The currently selected tab.
            start_date (str): Selected start date.
            end_date (str): Selected end date.
            balance_sheet_quarterly (list[dict]): Quarterly balance sheet data from the store.

        Returns:
            tuple: Updated content and loading state for the balance sheet panel.
        """
        if tab != "balance-sheet":
            return no_update, False
        if balance_sheet_quarterly is None or len(balance_sheet_quarterly) == 0:
            return dmc.Text("No balance sheet data loaded.", c="red"), False

        balance_sheet_quarterly_df = pd.DataFrame(balance_sheet_quarterly)
        if balance_sheet_quarterly_df.empty or "fiscal_date_ending" not in balance_sheet_quarterly_df.columns:
            return dmc.Text("No balance sheet data loaded.", c="red"), False

        balance_sheet_quarterly_df["fiscal_date_ending"] = pd.to_datetime(balance_sheet_quarterly_df["fiscal_date_ending"])
        balance_sheet_quarterly_df = balance_sheet_quarterly_df.sort_values("fiscal_date_ending")

        if start_date and end_date:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            mask = (balance_sheet_quarterly_df["fiscal_date_ending"] >= start_dt) & (balance_sheet_quarterly_df["fiscal_date_ending"] <= end_dt)
            selected_balance_sheet = balance_sheet_quarterly_df.loc[mask]
        else:
            selected_balance_sheet = balance_sheet_quarterly_df

        metrics = [
            "total_assets",
            "total_liabilities",
            "total_shareholder_equity",
            "total_current_assets",
            "total_current_liabilities",
            "cash_and_cash_equivalents",
            "property_plant_equipment"
        ]

        if selected_balance_sheet.empty:
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
                        "long_term_debt"
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
            latest_date = str(selected_balance_sheet["fiscal_date_ending"].max().date())

            time_series_fig = plot_balance_sheet_time_series(selected_balance_sheet, columns=metrics)
            stacked_area_fig = plot_balance_sheet_stacked_area(selected_balance_sheet, stack_groups)
            bar_fig = plot_balance_sheet_bar(selected_balance_sheet, bar_groups)
            pie_fig = plot_balance_sheet_pie(selected_balance_sheet, latest_date, pie_columns)
            cards = render_balance_sheet_metric_cards(selected_balance_sheet, latest_date, metrics)

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
        company_fundamentals: list[dict] | None
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
        if company_fundamentals is None or len(company_fundamentals) == 0:
            return dmc.Text("No company info loaded.", c="red"), False

        try:
            company_fundamentals_df = pd.DataFrame(company_fundamentals)
            company_fundamentals_row = company_fundamentals_df.iloc[[0]]
            symbol = company_fundamentals_row.iloc[0].get("symbol", "") if not company_fundamentals_row.empty else ""
            fig = plot_company_fundamentals_table(company_fundamentals_row, symbol)
            return dcc.Graph(figure=fig, config={"displayModeBar": False}), False
        except Exception as e:
            return dmc.Text(f"Error displaying company fundamentals: {e}", c="red"), False

    @app.callback(
        Output("income-statement-content", "children"),
        Output("income-statement-loading", "visible"),
        Input("main-tabs", "value"),
        Input("company-base-store", "data"),
        Input("price-store", "data"),
        Input("q_income-store", "data"),
        prevent_initial_call=True
    )
    def update_income_statement_tab(
        tab: str,
        company_fundamentals: list[dict] | None,
        price_data: list[dict] | None,
        income_statement_quarterly: list[dict] | None
    ):
        """
        Callback to update the income statement tab with all relevant plots.

        Args:
            tab (str): The currently selected tab.
            company_fundamentals (list[dict] | None): Company fundamentals data.
            price_data (list[dict] | None): Daily timeseries price data.
            income_statement_quarterly (list[dict] | None): Quarterly income statement data.

        Returns:
            tuple: Updated content and loading state for the income statement panel.
        """
        if tab != "income-statement":
            print("[DEBUG] Not on income-statement tab.")
            return no_update, False
        if income_statement_quarterly is None or len(income_statement_quarterly) == 0:
            print(f"[DEBUG] No income statement data loaded. income_statement_quarterly={income_statement_quarterly}")
            return dmc.Text("No income statement data loaded.", c="red"), False
        if price_data is None or len(price_data) == 0:
            print(f"[DEBUG] No price data loaded. price_data={price_data}")
            return dmc.Text("No price data loaded.", c="red"), False
        try:
            print("[DEBUG] Creating DataFrames for income and price data.")
            income_df = pd.DataFrame(income_statement_quarterly)
            price_df = pd.DataFrame(price_data)
            print(f"[DEBUG] income_df shape: {income_df.shape}, columns: {income_df.columns.tolist()}")
            print(f"[DEBUG] price_df shape: {price_df.shape}, columns: {price_df.columns.tolist()}")
            symbol = income_df["symbol"].iloc[0] if "symbol" in income_df.columns and len(income_df) > 0 else ""
            print(f"[DEBUG] Using symbol: {symbol}")
            fig = plot_quarterly_revenue_net_income_vs_stock_price(symbol, income_df, price_df)
            print("[DEBUG] plot_quarterly_revenue_net_income_vs_stock_price returned a figure.")
            return dmc.Stack([
                dmc.Divider(label="Quarterly Revenue, Net Income & Stock Price", my=10),
                dcc.Graph(figure=fig),
            ], gap=16), False
        except Exception as e:
            print(f"[DEBUG] Exception in update_income_statement_tab: {e}")
            return dmc.Text(f"Error displaying income statement plots: {e}", c="red"), False