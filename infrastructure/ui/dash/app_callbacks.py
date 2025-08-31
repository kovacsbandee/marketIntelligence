"""
Dash app callback registrations for the Market Intelligence dashboard UI.

This module defines all Dash callback functions for updating UI components
in response to user interactions, including data loading, chart updates,
and tab content rendering. It connects the UI to the data service and
plotting utilities, and manages the state and interactivity of the dashboard.
"""

import dash
from dash import dcc, Output, Input, State, no_update
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash.development.base_component import Component

from infrastructure.ui.dash.data_service import load_symbol_data

from infrastructure.ui.dash.plots.company_fundamentals_plots import plot_company_fundamentals_table

from infrastructure.ui.dash.plots.daily_timeseries_plots import (
    plot_candlestick_chart,
    add_moving_averages_to_candlestick,
    add_bollinger_bands_to_candlestick,
    plot_candlestick_with_vwap,
    plot_rsi,
    plot_macd,
    plot_stochastic,
    plot_obv,
    plot_adx
)

from infrastructure.ui.dash.plots.balance_sheet_plots import (
    plot_balance_sheet_time_series,
    plot_balance_sheet_stacked_area,
    plot_balance_sheet_bar,
    plot_balance_sheet_pie,
    render_balance_sheet_metric_cards,
)

from infrastructure.ui.dash.plots.income_plots import (
                plot_key_metrics_dashboard,
                plot_quarterly_profit_margins,
                plot_expense_breakdown_vs_revenue,
                plot_income_statement_waterfall,
                plot_operating_profit_ebit_ebitda_trends,
                plot_expense_growth_scatter,
                plot_tax_and_interest_effects,
                plot_metric_vs_future_stock_return,
                plot_quarterly_revenue_net_income_vs_stock_price
)

from infrastructure.ui.dash.plots.earnings_plots import (
    plot_eps_actual_vs_estimate,
    plot_eps_surprise_percentage,
    plot_eps_actual_vs_estimate_scatter
)

from infrastructure.ui.dash.plots.cashflow_plots import (
    plot_cash_flow_categories,
    plot_operating_vs_net_income,
    plot_free_cash_flow
)

from infrastructure.ui.dash.plots.insider_transactions_plots import (
    prepare_insider_data,
    plot_insider_price_chart,
    plot_insider_transactions_over_time
)

from infrastructure.ui.dash.app_util import get_last_6_months_range

def register_callbacks(app: dash.Dash) -> None:

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
        Output("cashflow-store", "data"),
        Output("insider-transactions-store", "data"),
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

        return (
            data["start_date"],
            data["end_date"],
            data["status_message"],
            df_to_records(data["daily_timeseries"]),
            df_to_records(data["dividends"]),
            df_to_records(data["company_fundamentals"]),
            df_to_records(data["balance_sheet_quarterly"]),
            df_to_records(data["annual_balance_sheet"]),
            df_to_records(data.get("earnings") if "earnings" in data else None),
            df_to_records(data.get("income_statement_quarterly") if "income_statement_quarterly" in data else None),
            df_to_records(data.get("cashflow_statement_quarterly") if "cashflow_statement_quarterly" in data else None),
            df_to_records(data.get("insider_transactions") if "insider_transactions" in data else None)
        )

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
        Output("price-indicator-content", "children"),
        Output("price-indicator-loading", "visible"),
        Input("main-tabs", "value"),
        Input("start-date-picker", "value"),
        Input("end-date-picker", "value"),
        Input("price-store", "data"),
        prevent_initial_call=True
    )
    def price_charts_and_indicators(
        tab: str,
        start_date: str,
        end_date: str,
        daily_timeseries: list[dict] | None
    ) -> tuple[Component]:
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
        if not start_date or not end_date:
            start_date, end_date = get_last_6_months_range(daily_timeseries_df)
        date_range = [start_date, end_date]
        mask = (daily_timeseries_df["date"] >= date_range[0]) & (daily_timeseries_df["date"] <= date_range[1])
        selected_timeseries = daily_timeseries_df.loc[mask]
        try:
            fig = plot_candlestick_chart(selected_timeseries)
            fig.update_layout(xaxis_range=[start_date, end_date])
            fig_ma = add_moving_averages_to_candlestick(selected_timeseries)
            fig_ma.update_layout(xaxis_range=[start_date, end_date])
            fig_bb = add_bollinger_bands_to_candlestick(selected_timeseries)
            fig_bb.update_layout(xaxis_range=[start_date, end_date])
            fig_vwap = plot_candlestick_with_vwap(selected_timeseries)
            fig_vwap.update_layout(xaxis_range=[start_date, end_date])
            fig_rsi = plot_rsi(selected_timeseries)
            fig_rsi.update_layout(xaxis_range=[start_date, end_date])
            fig_macd = plot_macd(selected_timeseries)
            fig_stoch = plot_stochastic(selected_timeseries)
            fig_obv = plot_obv(selected_timeseries)
            fig_adx = plot_adx(selected_timeseries)
            return dmc.Stack([
                dmc.Divider(label="Candlestick Chart", my=10),
                dcc.Graph(figure=fig),
                dmc.Divider(label="Candlestick with Moving Averages", my=10),
                dcc.Graph(figure=fig_ma),
                dmc.Divider(label="Candlestick with Bollinger Bands", my=10),
                dcc.Graph(figure=fig_bb),
                dmc.Divider(label="Candlestick with VWAP", my=10),
                dcc.Graph(figure=fig_vwap),
                dmc.Divider(label="RSI (Relative Strength Index)", my=10),
                dcc.Graph(figure=fig_rsi),
                dmc.Divider(label="MACD (Moving Average Convergence Divergence)", my=10),
                dcc.Graph(figure=fig_macd),
                dmc.Divider(label="Stochastic Oscillator", my=10),
                dcc.Graph(figure=fig_stoch),
                dmc.Divider(label="On-Balance Volume (OBV)", my=10),
                dcc.Graph(figure=fig_obv),
                dmc.Divider(label="Average Directional Index (ADX)", my=10),
                dcc.Graph(figure=fig_adx),
            ], gap=16), False
        except Exception as e:
            return dmc.Text(f"Error generating plot: {e}", c="red"), False

    @app.callback(
        Output("earnings-content", "children"),
        Output("earnings-loading", "visible"),
        Input("main-tabs", "value"),
        Input("earnings-store", "data"),
        prevent_initial_call=True
    )
    def update_earnings_panel(tab: str, 
                              earnings_data: list[dict] | None):
        if tab != "earnings":
            return no_update, False
        if earnings_data is None or len(earnings_data) == 0:
            return dmc.Text("No earnings data loaded.", c="red"), False
        try:
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

    @app.callback(
        Output("income-statement-content", "children"),
        Output("income-statement-loading", "visible"),
        Input("main-tabs", "value"),
        Input("price-store", "data"),
        Input("q_income-store", "data"),
        prevent_initial_call=True
    )
    def update_income_statement_tab(
        tab: str,
        price_data: list[dict] | None,
        income_statement_quarterly: list[dict] | None
    ):
        """
        Callback to update the income statement tab with all relevant plots.

        Args:
            tab (str): The currently selected tab.
            price_data (list[dict] | None): Daily timeseries price data.
            income_statement_quarterly (list[dict] | None): Quarterly income statement data.

        Returns:
            tuple: Updated content and loading state for the income statement panel.
        """
        if tab != "income-statement":
            return no_update, False
        if income_statement_quarterly is None or len(income_statement_quarterly) == 0:
            return dmc.Text("No income statement data loaded.", c="red"), False
        if price_data is None or len(price_data) == 0:
            return dmc.Text("No price data loaded.", c="red"), False
        try:
            income_df = pd.DataFrame(income_statement_quarterly)
            price_df = pd.DataFrame(price_data)
            symbol = income_df["symbol"].iloc[0] if "symbol" in income_df.columns and len(income_df) > 0 else ""
            fig_dashboard = plot_key_metrics_dashboard(symbol, income_df, price_df)
            fig_margins = plot_quarterly_profit_margins(symbol, income_df)
            fig_expenses = plot_expense_breakdown_vs_revenue(symbol, income_df)
            fig_expense_growth = plot_expense_growth_scatter(symbol, income_df)
            fig_tax_interest = plot_tax_and_interest_effects(symbol, income_df)
            fig_op_trends = plot_operating_profit_ebit_ebitda_trends(symbol, income_df)
            fig_waterfall = plot_income_statement_waterfall(symbol, income_df)
            fig = plot_quarterly_revenue_net_income_vs_stock_price(symbol, income_df, price_df)
            default_metric = "net_income" if "net_income" in income_df.columns else income_df.columns[0]
            fig_metric_vs_return = plot_metric_vs_future_stock_return(symbol, income_df, price_df, default_metric)
            return dmc.Stack([
                dmc.Divider(label="Key Metrics Dashboard", my=10),
                dcc.Graph(figure=fig_dashboard),
                dmc.Divider(label="Quarterly Profit Margins", my=10),
                dcc.Graph(figure=fig_margins),
                dmc.Divider(label="Expense Breakdown vs Revenue", my=10),
                dcc.Graph(figure=fig_expenses),
                dmc.Divider(label="Expense Growth vs Revenue (Bubble Chart)", my=10),
                dcc.Graph(figure=fig_expense_growth),
                dmc.Divider(label="Tax & Interest Effects on Pre-Tax and Net Income", my=10),
                dcc.Graph(figure=fig_tax_interest),
                dmc.Divider(label="Operating Profit, EBIT & EBITDA Trends", my=10),
                dcc.Graph(figure=fig_op_trends),
                dmc.Divider(label="Income Statement Waterfall (Most Recent Quarter)", my=10),
                dcc.Graph(figure=fig_waterfall),
                dmc.Divider(label="Quarterly Revenue, Net Income & Stock Price", my=10),
                dcc.Graph(figure=fig),
                dmc.Divider(label="Metric vs Future Stock Return (Quarterly)", my=10),
                dcc.Graph(figure=fig_metric_vs_return),
            ], gap=16), False
        except Exception as e:
            print(f"[DEBUG] Exception in update_income_statement_tab: {e}")
            return dmc.Text(f"Error displaying income statement plots: {e}", c="red"), False

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

        if selected_balance_sheet.empty:
            fig = go.Figure()
            fig.update_layout(
                title="No balance sheet data available for the selected date range.",
                xaxis_title="Fiscal Date Ending",
                yaxis_title="Value"
            )
            return dcc.Graph(figure=fig), False

        try:
            latest_date = str(selected_balance_sheet["fiscal_date_ending"].max().date())

            # Time Series Plot
            time_series_fig = plot_balance_sheet_time_series(selected_balance_sheet)

            # Stacked Area Plot
            stacked_area_fig = plot_balance_sheet_stacked_area(selected_balance_sheet)

            # Bar Chart
            bar_fig = plot_balance_sheet_bar(selected_balance_sheet)

            # Pie Chart
            pie_fig = plot_balance_sheet_pie(selected_balance_sheet, latest_date)

            # Metric Cards
            cards = render_balance_sheet_metric_cards(selected_balance_sheet, latest_date)

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
        Output("cash-flow-content", "children"),
        Output("cash-flow-loading", "visible"),
        Input("main-tabs", "value"),
        Input("start-date-picker", "value"),
        Input("end-date-picker", "value"),
        Input("cashflow-store", "data"),
        prevent_initial_call=True
    )
    def update_cash_flow(
        tab: str,
        start_date: str,
        end_date: str,
        cashflow_data: list[dict] | None
    ) -> tuple[Component, bool]:
        """
        Callback to update the cash flow panel with all relevant plots.

        Args:
            tab (str): The currently selected tab.
            start_date (str): Selected start date.
            end_date (str): Selected end date.
            cashflow_data (list[dict]): Cash flow data from the store.

        Returns:
            tuple: Updated content and loading state for the cash flow panel.
        """
        if tab != "cash-flow":
            return no_update, False
        if cashflow_data is None or len(cashflow_data) == 0:
            return dmc.Text("No cash flow data loaded.", c="red"), False
        cashflow_df = pd.DataFrame(cashflow_data)
        if cashflow_df.empty:
            return dmc.Text("No cash flow data loaded.", c="red"), False
        # Filter by date if possible
        date_col = next((col for col in cashflow_df.columns if 'fiscal_date' in col.lower() or 'date' in col.lower()), None)
        if date_col and start_date and end_date:
            cashflow_df[date_col] = pd.to_datetime(cashflow_df[date_col])
            mask = (cashflow_df[date_col] >= pd.to_datetime(start_date)) & (cashflow_df[date_col] <= pd.to_datetime(end_date))
            selected_cashflow = cashflow_df.loc[mask]
        else:
            selected_cashflow = cashflow_df
        try:
            fig_categories = plot_cash_flow_categories(selected_cashflow)
            fig_ocf_vs_ni = plot_operating_vs_net_income(selected_cashflow)
            fig_fcf = plot_free_cash_flow(selected_cashflow)
            return dmc.Stack([
                dmc.Divider(label="Cash Flow Categories (Operating, Investing, Financing)", my=10),
                dcc.Graph(figure=fig_categories),
                dmc.Divider(label="Operating Cash Flow vs Net Income", my=10),
                dcc.Graph(figure=fig_ocf_vs_ni),
                dmc.Divider(label="Free Cash Flow per Period", my=10),
                dcc.Graph(figure=fig_fcf),
            ], gap=16), False
        except Exception as e:
            return dmc.Text(f"Error generating cash flow plots: {e}", c="red"), False        

    @app.callback(
        Output("insider-transactions-content", "children"),
        Output("insider-transactions-loading", "visible"),
        Input("main-tabs", "value"),
        Input("start-date-picker", "value"),
        Input("end-date-picker", "value"),
        Input("insider-transactions-store", "data"),
        Input("price-store", "data"),
        prevent_initial_call=True
    )
    def update_insider_transactions(
        tab: str,
        start_date: str,
        end_date: str,
        insider_transactions: list[dict] | None,
        price_data: list[dict] | None
    ) -> tuple[Component, bool]:
        """
        Callback to update the insider transactions tab with relevant plots.

        Args:
            tab (str): The currently selected tab.
            start_date (str): Selected start date.
            end_date (str): Selected end date.
            insider_transactions (list[dict] | None): Insider transactions data from the store.
            price_data (list[dict] | None): Daily price data from the store.

        Returns:
            tuple: Updated content and loading state for the insider transactions panel.
        """

        if tab != "insider-transactions":
            return no_update, False
        if insider_transactions is None or len(insider_transactions) == 0:
            return dmc.Text("No insider transactions data available.", c="dimmed"), False
        if price_data is None or len(price_data) == 0:
            return dmc.Text("No price data available for insider transactions plot.", c="dimmed"), False

        insider_df = pd.DataFrame(insider_transactions)
        price_df = pd.DataFrame(price_data)
        if insider_df.empty:
            return dmc.Text("No insider transactions available for this symbol.", c="dimmed"), False
        if price_df.empty:
            return dmc.Text("No price data available for this symbol.", c="dimmed"), False

        # Filter by date if possible
        if start_date and end_date:
            if "transaction_date" in insider_df.columns:
                insider_df["transaction_date"] = pd.to_datetime(insider_df["transaction_date"])
                mask = (insider_df["transaction_date"] >= pd.to_datetime(start_date)) & (insider_df["transaction_date"] <= pd.to_datetime(end_date))
                insider_df = insider_df.loc[mask]
            if "date" in price_df.columns:
                price_df["date"] = pd.to_datetime(price_df["date"])
                mask = (price_df["date"] >= pd.to_datetime(start_date)) & (price_df["date"] <= pd.to_datetime(end_date))
                price_df = price_df.loc[mask]

        if insider_df.empty:
            return dmc.Text("No insider transactions in selected date range.", c="dimmed"), False
        if price_df.empty:
            return dmc.Text("No price data in selected date range.", c="dimmed"), False

        # Prepare and plot
        try:
            insider_df = prepare_insider_data(insider_df)
            fig1 = plot_insider_price_chart(price_df, insider_df)
            fig2 = plot_insider_transactions_over_time(insider_df)
            content = dmc.Stack([
                dmc.Text("Insider Trades on Price Chart", size="md", fw=700),
                dcc.Graph(figure=fig1, style={"height": 400}),
                dmc.Text("Insider Transaction Prices Over Time", size="md", fw=700, mt=20),
                dcc.Graph(figure=fig2, style={"height": 400}),
            ], gap=24)
            return content, False
        except Exception as e:
            return dmc.Text(f"Error displaying insider transactions: {e}", c="red"), False