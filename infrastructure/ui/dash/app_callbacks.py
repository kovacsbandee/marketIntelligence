"""
Dash app callback registrations for the Market Intelligence dashboard UI.

This module defines all Dash callback functions for updating UI components
in response to user interactions, including data loading, chart updates,
tab content rendering, and quantitative analysis with progressive results.

Notes on guards & filtering:
- `guard_store` provides a single, consistent early-return for missing/empty store data to avoid noisy stack traces and show user-friendly messages.
- `filter_by_date` centralizes inclusive date-range filtering and no-ops when dates/columns are absent, keeping callbacks concise and defensive.
"""

import uuid
import threading
import logging
import dash
import pandas as pd
from typing import Any
from dash import Output, Input, State, no_update, html
import dash_mantine_components as dmc
from dash.development.base_component import Component

from infrastructure.ui.dash.data_service import load_symbol_data

from infrastructure.ui.dash.app_util import empty_load, Records, records_to_df, get_last_2_years_range, filter_by_date
from infrastructure.ui.dash.ids import Ids, Tabs
from infrastructure.ui.dash.panel_builders import (
    build_balance_sheet_panel,
    build_cash_flow_panel,
    build_company_base_panel,
    build_earnings_panel,
    build_income_statement_panel,
    build_insider_panel,
    build_price_panel,
    SCORED_INDICATORS,
    INDICATOR_DISPLAY_NAMES,
)
from infrastructure.ui.dash.plots.daily_timeseries_plots import plot_candlestick_with_overlays
from infrastructure.ui.dash.plots.analysis_gauge import (
    build_indicator_score_card,
    build_aggregate_score_card,
    build_placeholder_score_card,
    build_placeholder_aggregate_card,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server-side registry for background analysis jobs
# ---------------------------------------------------------------------------

_analysis_jobs: dict[str, dict] = {}
_analysis_jobs_lock = threading.Lock()


def _run_analysis_background(run_id: str, symbol: str, cutoff_date: str | None) -> None:
    """Run the QuantitativeAnalyst in a background thread, updating partial results progressively.

    This function instantiates a fresh Symbol and QuantitativeAnalyst, evaluates each
    indicator one-by-one, and writes partial results into ``_analysis_jobs[run_id]``
    so the Dash interval callback can pick them up.
    """
    from symbol.symbol import Symbol
    from infrastructure.databases.company.postgre_manager.company_data_manager import CompanyDataManager
    from analyst.quantitative_analyst.quantitative_analyst import QuantitativeAnalyst

    try:
        # Build Symbol with indicators
        adapter = CompanyDataManager()
        sym = Symbol(adapter, symbol, add_price_indicators=True)

        analyst = QuantitativeAnalyst(sym)
        analyst.load_articles()

        # Evaluate each indicator one-by-one for progressive updates
        scores = {}
        explanations = {}
        indicators_evaluated = []
        warnings = []
        errors = []

        for indicator in analyst.article_paths.keys():
            try:
                score, explanation = analyst.evaluate_indicator(indicator, cutoff_date=cutoff_date)
                scores[indicator] = score
                explanations[indicator] = explanation
                indicators_evaluated.append(indicator)

                # Write partial result
                with _analysis_jobs_lock:
                    job = _analysis_jobs.get(run_id)
                    if job:
                        job["completed_indicators"][indicator] = {
                            "score": score,
                            "explanation": explanation,
                        }
            except Exception as exc:
                logger.error("Error evaluating %s: %s", indicator, exc)
                errors.append(f"Error evaluating {indicator}: {exc}")
                with _analysis_jobs_lock:
                    job = _analysis_jobs.get(run_id)
                    if job:
                        job["completed_indicators"][indicator] = {
                            "score": None,
                            "explanation": f"Error: {exc}",
                        }

        # Compute aggregate
        numeric_scores = [s for s in scores.values() if isinstance(s, (int, float))]
        aggregate = None
        if numeric_scores:
            aggregate = sum(numeric_scores) / len(numeric_scores)
            aggregate = max(min(aggregate, 1.0), -1.0)

        # Get metadata from analyst
        import datetime
        try:
            df = analyst._get_last_year_data(cutoff_date=cutoff_date)
            num_data_points = len(df)
            if "date" in df.columns:
                last_date = df["date"].max()
                if hasattr(last_date, "to_pydatetime"):
                    last_date = last_date.to_pydatetime()
                last_epoch = int(last_date.timestamp())
                last_date_str = last_date.strftime("%Y-%m-%d %H:%M:%S")
            else:
                last_epoch = None
                last_date_str = None
        except Exception:
            num_data_points = 0
            last_epoch = None
            last_date_str = None

        analysis_run_dt = datetime.datetime.now()
        full_result = {
            "symbol": symbol,
            "epoch": last_epoch,
            "last_date": last_date_str,
            "num_data_points": num_data_points,
            "indicators_evaluated": indicators_evaluated,
            "analysis_run_timestamp": int(analysis_run_dt.timestamp()),
            "analysis_run_datetime": analysis_run_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "warnings": warnings,
            "errors": errors,
            "indicator_scores": scores,
            "aggregate_score": aggregate,
            "explanations": explanations,
        }

        # Store in DB (always)
        try:
            from infrastructure.databases.analysis.postgre_manager.analyst_data_manager import get_analyst_data_handler
            from infrastructure.databases.analysis.postgre_manager.analyst_table_objects import (
                AnalystQuantitativeBase, AnalystQuantitativeScore, AnalystQuantitativeExplanation,
            )
            manager = get_analyst_data_handler()
            base_row = {
                "symbol": symbol,
                "epoch": last_epoch,
                "last_date": pd.to_datetime(last_date_str) if last_date_str else None,
                "num_data_points": num_data_points,
                "obv_used": int("obv" in indicators_evaluated),
                "vwap_used": int("vwap" in indicators_evaluated),
                "macd_used": int("macd" in indicators_evaluated),
                "rsi_used": int("rsi" in indicators_evaluated),
                "stochastic_oscillator_used": int("stochastic_oscillator" in indicators_evaluated),
                "bollinger_bands_used": int("bollinger_bands" in indicators_evaluated),
                "sma_used": int("sma" in indicators_evaluated),
                "ema_used": int("ema" in indicators_evaluated),
                "adx_used": int("adx" in indicators_evaluated),
                "analysis_run_timestamp": int(analysis_run_dt.timestamp()),
                "analysis_run_datetime": analysis_run_dt,
                "warnings": ",".join(warnings) if warnings else None,
                "errors": ",".join(errors) if errors else None,
                "aggregate_score": aggregate,
            }
            score_row = {
                "symbol": symbol,
                "epoch": last_epoch,
                "aggregate_score": aggregate,
                **{ind: scores.get(ind) for ind in analyst.article_paths.keys()},
            }
            explanation_row = {
                "symbol": symbol,
                "epoch": last_epoch,
                **{ind: explanations.get(ind) for ind in analyst.article_paths.keys()},
            }
            manager.insert_new_data(table=AnalystQuantitativeBase, rows=[base_row])
            manager.insert_new_data(table=AnalystQuantitativeScore, rows=[score_row])
            manager.insert_new_data(table=AnalystQuantitativeExplanation, rows=[explanation_row])
            logger.info("Analysis stored for %s (run_id=%s)", symbol, run_id)
        except Exception as exc:
            logger.error("Failed to store analysis in DB: %s", exc, exc_info=True)

        with _analysis_jobs_lock:
            job = _analysis_jobs.get(run_id)
            if job:
                job["status"] = "done"
                job["full_result"] = full_result
                job["aggregate"] = aggregate

    except Exception as exc:
        logger.error("Background analysis failed: %s", exc, exc_info=True)
        with _analysis_jobs_lock:
            job = _analysis_jobs.get(run_id)
            if job:
                job["status"] = "error"
                job["error"] = str(exc)

def register_callbacks(app: dash.Dash) -> None:

    @app.callback(
        Output(Ids.START_DATE_PICKER, "value"),
        Output(Ids.END_DATE_PICKER, "value"),
        Output(Ids.STATUS_DIV, "title"),
        Output(Ids.PRICE_STORE, "data"),
        Output(Ids.DIVIDENDS_STORE, "data"),
        Output(Ids.COMPANY_BASE_STORE, "data"),
        Output(Ids.Q_BALANCE_STORE, "data"),
        Output(Ids.A_BALANCE_STORE, "data"),
        Output(Ids.EARNINGS_STORE, "data"),
        Output(Ids.Q_INCOME_STORE, "data"),
        Output(Ids.CASHFLOW_STORE, "data"),
        Output(Ids.INSIDER_STORE, "data"),
        Input(Ids.LOAD_BUTTON, "n_clicks"),
        Input(Ids.SYMBOL_INPUT, "n_submit"),
        State(Ids.SYMBOL_INPUT, "value"),
        prevent_initial_call=True
    )
    def load_symbol(
        n_clicks: int,
        n_submit: int,
        symbol: str
    ) -> tuple:
        """
        Callback to load all relevant data for a given symbol and update stores and UI, including earnings data.
        """
        if not symbol:
            return empty_load("Please enter a symbol.")
        data = load_symbol_data(symbol)
        if data["daily_timeseries"] is None:
            return empty_load(data["status_message"])

        return (
            data["start_date"],
            data["end_date"],
            data["status_message"],
            data["daily_timeseries"],
            data["dividends"],
            data["company_fundamentals"],
            data["balance_sheet_quarterly"],
            data["annual_balance_sheet"],
            data["earnings"],
            data["income_statement_quarterly"],
            data["cashflow_statement_quarterly"],
            data["insider_transactions"],
        )

    @app.callback(
        Output(Ids.COMPANY_BASE_CONTENT, "children"),
        Output(Ids.COMPANY_BASE_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.COMPANY_BASE_STORE, "data"),
        State(Ids.SYMBOL_INPUT, "value"),
        prevent_initial_call=True
    )
    def update_company_base(
        tab: str,
    company_fundamentals: Records,
        symbol_input: str | None
    ) -> tuple[Component, bool]:
        """
        Callback to update the company base info panel with a Plotly table.

        Args:
            tab (str): The currently selected tab.
            company_base_data (dict): The company base data from the store.

        Returns:
            tuple: Updated content and loading state for the company base panel.
        """
        if tab != Tabs.COMPANY_BASE:
            return no_update, False
        try:
            return build_company_base_panel(company_fundamentals, symbol_input)
        except Exception as e:
            return dmc.Text(f"Error displaying company fundamentals: {e}", c="red"), False

    @app.callback(
        Output(Ids.PRICE_INDICATOR_CONTENT, "children"),
        Output(Ids.PRICE_INDICATOR_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.START_DATE_PICKER, "value"),
        Input(Ids.END_DATE_PICKER, "value"),
        Input(Ids.PRICE_STORE, "data"),
        prevent_initial_call=True
    )
    def price_charts_and_indicators(
        tab: str,
        start_date: str,
        end_date: str,
    daily_timeseries: Records
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
        if tab != Tabs.PRICE_INDICATOR:
            return no_update, False
        try:
            return build_price_panel(daily_timeseries, start_date, end_date)
        except Exception as e:
            return dmc.Text(f"Error generating plot: {e}", c="red"), False

    @app.callback(
        Output(Ids.PRICE_CHART, "figure"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.START_DATE_PICKER, "value"),
        Input(Ids.END_DATE_PICKER, "value"),
        Input(Ids.PRICE_STORE, "data"),
        Input(Ids.PRICE_OVERLAY_TOGGLE, "value"),
        prevent_initial_call=True
    )
    def update_price_chart_overlays(
        tab: str,
        start_date: str,
        end_date: str,
        daily_timeseries: Records,
        overlays: list[str] | None,
    ) -> Any:
        """Toggle overlays on the primary price chart without removing other panels."""
        if tab != Tabs.PRICE_INDICATOR:
            return no_update

        overlays = overlays or []
        if not daily_timeseries:
            return {"data": [], "layout": {"title": "No price data loaded."}}

        df = records_to_df(daily_timeseries, table="daily_timeseries")
        if df.empty:
            return {"data": [], "layout": {"title": "No price data loaded."}}

        if not start_date or not end_date:
            start_date, end_date = get_last_2_years_range(df)

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        if start_dt > end_dt:
            return {"data": [], "layout": {"title": "Start date must be on or before end date."}}

        selected_timeseries = filter_by_date(df, start_dt, end_dt, "date")
        if selected_timeseries.empty:
            return {"data": [], "layout": {"title": "No price data in selected date range."}}

        fig = plot_candlestick_with_overlays(
            selected_timeseries,
            show_ma="ma" in overlays,
            show_bb="bb" in overlays,
            show_vwap="vwap" in overlays,
        )
        fig.update_layout(xaxis_range=[start_dt, end_dt])
        return fig

    @app.callback(
        Output(Ids.EARNINGS_CONTENT, "children"),
        Output(Ids.EARNINGS_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.START_DATE_PICKER, "value"),
        Input(Ids.END_DATE_PICKER, "value"),
        Input(Ids.EARNINGS_STORE, "data"),
        State(Ids.SYMBOL_INPUT, "value"),
        prevent_initial_call=True
    )
    def update_earnings_panel(
        tab: str,
        start_date: str,
        end_date: str,
        earnings_data: Records,
        symbol_input: str | None
    ):
        if tab != Tabs.EARNINGS:
            return no_update, False
        try:
            return build_earnings_panel(earnings_data, start_date, end_date, symbol_input)
        except Exception as e:
            return dmc.Text(f"Error displaying earnings plots: {e}", c="red"), False

    @app.callback(
        Output(Ids.INCOME_STATEMENT_CONTENT, "children"),
        Output(Ids.INCOME_STATEMENT_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.START_DATE_PICKER, "value"),
        Input(Ids.END_DATE_PICKER, "value"),
        Input(Ids.PRICE_STORE, "data"),
        Input(Ids.Q_INCOME_STORE, "data"),
        State(Ids.SYMBOL_INPUT, "value"),
        prevent_initial_call=True
    )
    def update_income_statement_tab(
        tab: str,
        start_date: str,
        end_date: str,
        price_data: Records,
        income_statement_quarterly: Records,
        symbol_input: str | None
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
        if tab != Tabs.INCOME_STATEMENT:
            return no_update, False
        try:
            return build_income_statement_panel(price_data, income_statement_quarterly, start_date, end_date, symbol_input)
        except Exception as e:
            print(f"[DEBUG] Exception in update_income_statement_tab: {e}")
            return dmc.Text(f"Error displaying income statement plots: {e}", c="red"), False

    @app.callback(
        Output(Ids.BALANCE_SHEET_CONTENT, "children"),
        Output(Ids.BALANCE_SHEET_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.START_DATE_PICKER, "value"),
        Input(Ids.END_DATE_PICKER, "value"),
        Input(Ids.Q_BALANCE_STORE, "data"),
        prevent_initial_call=True
    )
    def update_balance_sheet(
        tab: str,
        start_date: str,
        end_date: str,
    balance_sheet_quarterly: Records
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
        if tab != Tabs.BALANCE_SHEET:
            return no_update, False
        try:
            return build_balance_sheet_panel(balance_sheet_quarterly, start_date, end_date)
        except Exception as e:
            return dmc.Text(f"Error generating plot: {e}", c="red"), False
        
    @app.callback(
        Output(Ids.CASH_FLOW_CONTENT, "children"),
        Output(Ids.CASH_FLOW_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.START_DATE_PICKER, "value"),
        Input(Ids.END_DATE_PICKER, "value"),
        Input(Ids.CASHFLOW_STORE, "data"),
        prevent_initial_call=True
    )
    def update_cash_flow(
        tab: str,
        start_date: str,
        end_date: str,
    cashflow_data: Records
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
        if tab != Tabs.CASH_FLOW:
            return no_update, False
        try:
            return build_cash_flow_panel(cashflow_data, start_date, end_date)
        except Exception as e:
            return dmc.Text(f"Error generating cash flow plots: {e}", c="red"), False        

    @app.callback(
        Output(Ids.INSIDER_CONTENT, "children"),
        Output(Ids.INSIDER_LOADING, "visible"),
        Input(Ids.MAIN_TABS, "value"),
        Input(Ids.START_DATE_PICKER, "value"),
        Input(Ids.END_DATE_PICKER, "value"),
        Input(Ids.INSIDER_STORE, "data"),
        Input(Ids.PRICE_STORE, "data"),
        prevent_initial_call=True
    )
    def update_insider_transactions(
        tab: str,
        start_date: str,
        end_date: str,
    insider_transactions: Records,
    price_data: Records
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

        if tab != Tabs.INSIDER_TRANSACTIONS:
            return no_update, False
        try:
            return build_insider_panel(insider_transactions, price_data, start_date, end_date)
        except Exception as e:
            return dmc.Text(f"Error displaying insider transactions: {e}", c="red"), False

    @app.callback(
        Output(Ids.START_DATE_PICKER, "value", allow_duplicate=True),
        Output(Ids.END_DATE_PICKER, "value", allow_duplicate=True),
        Input(Ids.RANGE_PRESET, "value"),
        State(Ids.PRICE_STORE, "data"),
        prevent_initial_call=True,
    )
    def apply_quick_range(preset: str | None, price_data: Records):
        if not preset or not price_data:
            return no_update, no_update

        df = records_to_df(price_data, table="daily_timeseries")
        if df.empty or "date" not in df.columns:
            return no_update, no_update
        df["date"] = pd.to_datetime(df["date"])
        latest = df["date"].max()
        earliest = df["date"].min()

        if preset == "6m":
            start = (latest - pd.Timedelta(days=182)).strftime("%Y-%m-%d")
            end = latest.strftime("%Y-%m-%d")
        elif preset == "1y":
            start = (latest - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
            end = latest.strftime("%Y-%m-%d")
        elif preset == "2y":
            start = (latest - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
            end = latest.strftime("%Y-%m-%d")
        elif preset == "ytd":
            start = pd.Timestamp(year=latest.year, month=1, day=1).strftime("%Y-%m-%d")
            end = latest.strftime("%Y-%m-%d")
        elif preset == "max":
            start = earliest.strftime("%Y-%m-%d")
            end = latest.strftime("%Y-%m-%d")
        else:
            return no_update, no_update

        return start, end

    # Map button loading_state to button.loading for a spinner during requests
    app.clientside_callback(
        "function(ls){return ls && ls.is_loading;}",
        Output(Ids.LOAD_BUTTON, "loading"),
        Input(Ids.LOAD_BUTTON, "loading_state"),
    )

    # ------------------------------------------------------------------
    # Candlestick click → store the clicked date
    # ------------------------------------------------------------------
    @app.callback(
        Output(Ids.ANALYSIS_CLICKED_DATE, "data"),
        Input(Ids.PRICE_CHART, "clickData"),
        prevent_initial_call=True,
    )
    def capture_chart_click(click_data):
        """Store the date from a candlestick click for later analysis."""
        if not click_data or "points" not in click_data or not click_data["points"]:
            return no_update
        point = click_data["points"][0]
        date_str = point.get("x")
        if date_str:
            return date_str
        return no_update

    # ------------------------------------------------------------------
    # Run Analysis button → start background job, enable polling
    # ------------------------------------------------------------------
    @app.callback(
        Output(Ids.ANALYSIS_RUN_ID, "data"),
        Output(Ids.ANALYSIS_INTERVAL, "disabled"),
        Output(Ids.ANALYZE_BUTTON, "loading"),
        # Show placeholder cards immediately
        Output(Ids.AGGREGATE_SCORE_CARD, "children"),
        Output(Ids.AGGREGATE_SCORE_CARD, "style"),
        Input(Ids.ANALYZE_BUTTON, "n_clicks"),
        State(Ids.SYMBOL_INPUT, "value"),
        State(Ids.ANALYSIS_CLICKED_DATE, "data"),
        State(Ids.PRICE_STORE, "data"),
        prevent_initial_call=True,
    )
    def start_analysis(n_clicks, symbol, clicked_date, price_data):
        """Kick off the QuantitativeAnalyst in a background thread."""
        if not n_clicks or not symbol or not price_data:
            return no_update, no_update, no_update, no_update, no_update

        symbol = symbol.strip().upper()

        # Determine cutoff date
        cutoff_date = None
        if clicked_date:
            cutoff_date = clicked_date
        # else: analyst uses latest date

        run_id = str(uuid.uuid4())

        # Initialize job registry entry
        with _analysis_jobs_lock:
            _analysis_jobs[run_id] = {
                "symbol": symbol,
                "cutoff_date": cutoff_date,
                "status": "running",
                "completed_indicators": {},  # indicator -> {score, explanation}
                "aggregate": None,
                "full_result": None,
                "error": None,
            }

        # Launch background thread
        thread = threading.Thread(
            target=_run_analysis_background,
            args=(run_id, symbol, cutoff_date),
            daemon=True,
        )
        thread.start()

        # Return run_id, enable interval, show button loading, show placeholder aggregate
        return (
            run_id,
            False,  # enable interval
            True,   # button loading
            [build_placeholder_aggregate_card()],
            {"display": "block"},
        )

    # ------------------------------------------------------------------
    # Interval poller → read partial results and render cards
    # ------------------------------------------------------------------
    @app.callback(
        Output(Ids.ANALYSIS_STORE, "data"),
        Output(Ids.ANALYSIS_INTERVAL, "disabled", allow_duplicate=True),
        Output(Ids.ANALYZE_BUTTON, "loading", allow_duplicate=True),
        Output(Ids.AGGREGATE_SCORE_CARD, "children", allow_duplicate=True),
        # Per-indicator score card outputs (8 indicators)
        *[Output(f"{Ids.SCORE_CARD_PREFIX}{ind}", "children") for ind in SCORED_INDICATORS],
        *[Output(f"{Ids.SCORE_CARD_PREFIX}{ind}", "style") for ind in SCORED_INDICATORS],
        Input(Ids.ANALYSIS_INTERVAL, "n_intervals"),
        State(Ids.ANALYSIS_RUN_ID, "data"),
        prevent_initial_call=True,
    )
    def poll_analysis_results(n_intervals, run_id):
        """Poll background job for partial results and update score cards progressively."""
        n_outputs = 4 + len(SCORED_INDICATORS) * 2  # store + interval + loading + aggregate + 8 children + 8 styles

        if not run_id:
            return (no_update,) * n_outputs

        with _analysis_jobs_lock:
            job = _analysis_jobs.get(run_id)

        if not job:
            return (no_update,) * n_outputs

        # Build per-indicator card outputs
        indicator_children = []
        indicator_styles = []
        for ind in SCORED_INDICATORS:
            result = job["completed_indicators"].get(ind)
            if result is not None:
                card = build_indicator_score_card(
                    indicator_name=ind,
                    score=result["score"],
                    explanation=result["explanation"],
                    display_name=INDICATOR_DISPLAY_NAMES.get(ind),
                )
                indicator_children.append([card])
                indicator_styles.append({"display": "block"})
            else:
                indicator_children.append(no_update)
                indicator_styles.append(no_update)

        is_done = job["status"] in ("done", "error")

        # Build aggregate card
        if is_done and job["full_result"]:
            result = job["full_result"]
            agg_card = [build_aggregate_score_card(
                aggregate_score=result.get("aggregate_score"),
                analysis_date=result.get("last_date", "")[:10],
                symbol=result.get("symbol"),
                n_indicators=len(result.get("indicators_evaluated", [])),
            )]
            store_data = result
        elif is_done and job["error"]:
            agg_card = [dmc.Text(f"Analysis error: {job['error']}", c="red")]
            store_data = {"error": job["error"]}
        else:
            agg_card = no_update
            store_data = no_update

        disable_interval = is_done
        button_loading = not is_done

        return (
            store_data,
            disable_interval,
            button_loading,
            agg_card,
            *indicator_children,
            *indicator_styles,
        )

    # ------------------------------------------------------------------
    # Add vertical marker line on ALL charts when a date is clicked
    # ------------------------------------------------------------------
    @app.callback(
        Output(Ids.PRICE_CHART, "figure", allow_duplicate=True),
        Output(Ids.RSI_CHART, "figure", allow_duplicate=True),
        Output(Ids.MACD_CHART, "figure", allow_duplicate=True),
        Output(Ids.STOCHASTIC_CHART, "figure", allow_duplicate=True),
        Output(Ids.OBV_CHART, "figure", allow_duplicate=True),
        Output(Ids.ADX_CHART, "figure", allow_duplicate=True),
        Output(Ids.ANALYSIS_DATE_DISPLAY, "children"),
        Output(Ids.ANALYSIS_DATE_DISPLAY, "style"),
        Input(Ids.ANALYSIS_CLICKED_DATE, "data"),
        State(Ids.PRICE_CHART, "figure"),
        State(Ids.RSI_CHART, "figure"),
        State(Ids.MACD_CHART, "figure"),
        State(Ids.STOCHASTIC_CHART, "figure"),
        State(Ids.OBV_CHART, "figure"),
        State(Ids.ADX_CHART, "figure"),
        prevent_initial_call=True,
    )
    def add_click_marker_to_all_charts(
        clicked_date,
        price_fig, rsi_fig, macd_fig, stoch_fig, obv_fig, adx_fig,
    ):
        """Add/update a vertical dashed line on all indicator charts at the clicked date."""
        if not clicked_date:
            return (no_update,) * 8

        import plotly.graph_objects as go

        date_label = clicked_date[:10] if len(str(clicked_date)) >= 10 else clicked_date

        def _add_marker(fig_data, show_annotation: bool = False):
            """Add a vertical line marker to a figure dict; returns updated Figure."""
            if not fig_data:
                return no_update
            fig = go.Figure(fig_data)

            # Remove existing markers
            fig.layout.shapes = [
                s for s in (fig.layout.shapes or [])
                if getattr(s, "name", None) != "analysis_marker"
            ]
            fig.layout.annotations = [
                a for a in (fig.layout.annotations or [])
                if getattr(a, "name", None) != "analysis_marker"
            ]

            fig.add_shape(
                type="line",
                x0=clicked_date, x1=clicked_date,
                y0=0, y1=1,
                yref="paper",
                line=dict(color="#228be6", width=2, dash="dash"),
                name="analysis_marker",
            )
            if show_annotation:
                fig.add_annotation(
                    x=clicked_date,
                    y=1,
                    yref="paper",
                    text=f"Analysis: {date_label}",
                    showarrow=False,
                    font=dict(color="#228be6", size=11),
                    xanchor="left",
                    yanchor="bottom",
                    name="analysis_marker",
                )
            return fig

        # Date display badge for the analysis section
        date_badge = dmc.Badge(
            f"Selected date: {date_label}",
            color="blue",
            variant="light",
            size="lg",
            radius="sm",
        )

        return (
            _add_marker(price_fig, show_annotation=True),
            _add_marker(rsi_fig),
            _add_marker(macd_fig),
            _add_marker(stoch_fig),
            _add_marker(obv_fig),
            _add_marker(adx_fig),
            [date_badge],
            {"display": "block"},
        )