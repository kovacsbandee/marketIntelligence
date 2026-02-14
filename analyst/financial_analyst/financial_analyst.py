"""
financial_analyst.py
--------------------

Concrete ``Analyst`` subclass for fundamental financial analysis.

Operates on a ``Symbol`` instance whose quarterly DataFrames
(``income_statement_quarterly``, ``balance_sheet_quarterly``,
``cash_flow_quarterly``, ``earnings_quarterly``) have been enriched
with derived metric columns by ``add_financial_metrics.py`` (margins,
liquidity/leverage ratios, growth rates, etc.).

Six analysis categories are evaluated:

1. **profitability** – gross / operating / net / EBITDA margins
2. **revenue_growth** – QoQ and YoY revenue trends
3. **earnings** – EPS trends, beat/miss patterns
4. **liquidity** – current ratio, quick ratio, cash ratio
5. **leverage** – debt-to-equity, debt-to-assets, equity ratio
6. **cash_flow_health** – free cash flow, CF-to-net-income

Scores are on a continuous ``[-1, 1]`` scale (-1 = underpriced,
+1 = overpriced).  An aggregate score is the mean of all valid
category scores.

Usage::

    from symbol.symbol import Symbol
    from analyst.financial_analyst.financial_analyst import FinancialAnalyst

    storage = Symbol(adapter, symbol="AAPL")
    fa = FinancialAnalyst(storage)
    result = fa.run_analysis()
    print(result["aggregate_score"])
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from symbol.symbol import Symbol
from analyst.analyst import Analyst
from infrastructure.databases.analysis.postgre_manager.analyst_data_manager import get_analyst_data_handler
from infrastructure.databases.analysis.postgre_manager.analyst_table_objects import (
    AnalystFinancialBase,
    AnalystFinancialScore,
    AnalystFinancialExplanation,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Knowledge-base article paths
# ---------------------------------------------------------------------------
PDF_PATH = "./infrastructure/databases/rag_knowledge_base/financial_analyst/"

ARTICLE_PATHS = {
    "balance_sheet": PDF_PATH + "balancesheet_statement/Breaking Down the Balance Sheet.pdf",
    "income_statement": PDF_PATH + "income_statement/Income Statement_ How to Read and Use It.pdf",
    "cash_flow": PDF_PATH + "cashflow_statement/Cash Flow Statements_ How to Prepare and Read One.pdf",
    "earnings": PDF_PATH + "earnings_statement/Earnings_ Company Earnings Defined, With Example of Measurements.pdf",
}

# ---------------------------------------------------------------------------
# Category definitions – each maps to a source table and the columns
# (both raw and pre-computed by add_financial_metrics.py) to include in
# the LLM prompt, plus the article to attach.
# ---------------------------------------------------------------------------
FINANCIAL_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "profitability": {
        "source_table": "income_statement_quarterly",
        "columns": [
            "gross_profit",
            "total_revenue",
            "operating_income",
            "net_income",
            "ebitda",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "ebitda_margin",
        ],
        "article_key": "income_statement",
    },
    "revenue_growth": {
        "source_table": "income_statement_quarterly",
        "columns": [
            "total_revenue",
            "cost_of_revenue",
            "revenue_qoq_growth",
            "revenue_yoy_growth",
        ],
        "article_key": "income_statement",
    },
    "earnings": {
        "source_table": "earnings_quarterly",
        "columns": [
            "reported_eps",
            "estimated_eps",
            "surprise",
            "surprise_percentage",
            "eps_qoq_growth",
            "earnings_beat",
            "surprise_abs",
        ],
        "article_key": "earnings",
    },
    "liquidity": {
        "source_table": "balance_sheet_quarterly",
        "columns": [
            "total_current_assets",
            "total_current_liabilities",
            "cash_and_cash_equivalents",
            "inventory",
            "current_ratio",
            "quick_ratio",
            "cash_ratio",
        ],
        "article_key": "balance_sheet",
    },
    "leverage": {
        "source_table": "balance_sheet_quarterly",
        "columns": [
            "total_liabilities",
            "total_assets",
            "total_shareholder_equity",
            "debt_to_equity",
            "debt_to_assets",
            "equity_ratio",
        ],
        "article_key": "balance_sheet",
    },
    "cash_flow_health": {
        "source_table": "cash_flow_quarterly",
        "columns": [
            "operating_cashflow",
            "capital_expenditures",
            "net_income",
            "cashflow_from_investment",
            "cashflow_from_financing",
            "free_cash_flow",
            "cf_to_net_income",
            "capex_to_net_income",
        ],
        "article_key": "cash_flow",
    },
}

SYSTEM_PROMPT = (
    "You are a senior financial analyst responsible for "
    "evaluating company fundamentals from quarterly statements."
)


class FinancialAnalyst(Analyst):
    """Evaluate a company's fundamental health using quarterly financial data.

    Parameters
    ----------
    symbol : Symbol
        Instance containing all database tables loaded as attributes.
    openai_key : str | None
        Explicit OpenAI key; falls back to ``KEY_FOR_FINANCIAL_ANALYST``.
    rate_limit_per_minute : int
        Max OpenAI calls per 60 s.
    num_quarters : int | None
        Number of most recent quarters to include (default 8).
    logger : logging.Logger | None
    store_in_db : bool
        Whether to persist results to the analyst database.
    """

    def __init__(
        self,
        symbol: Symbol = None,
        openai_key: Optional[str] = None,
        rate_limit_per_minute: int = 60,
        num_quarters: Optional[int] = 8,
        logger: Optional[logging.Logger] = None,
        store_in_db: bool = False,
    ) -> None:
        super().__init__(
            symbol=symbol,
            article_paths=ARTICLE_PATHS,
            openai_key=openai_key,
            env_key_name="KEY_FOR_FINANCIAL_ANALYST",
            model="gpt-4o-mini",
            system_prompt=SYSTEM_PROMPT,
            rate_limit_per_minute=rate_limit_per_minute,
            logger=logger,
            store_in_db=store_in_db,
        )
        self.num_quarters = num_quarters

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------
    def _get_quarterly_df(self, table_name: str) -> Optional[pd.DataFrame]:
        """Return a quarterly DataFrame from the Symbol, sorted by date, last N quarters.

        The DataFrame is expected to already contain any derived metric
        columns added by ``Symbol.add_all_financial_metrics()``.
        """
        df = getattr(self.symbol, table_name, None)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            self.logger.warning("Table '%s' not available on Symbol", table_name)
            return None
        df = df.copy()
        # Resolve date column
        date_col = None
        for candidate in ("fiscal_date_ending", "reported_date"):
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            for c in df.columns:
                if "date" in c.lower() or "fiscal" in c.lower():
                    date_col = c
                    break
        if date_col is None:
            self.logger.warning("No date column found in %s", table_name)
            return None
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
        if self.num_quarters is not None and self.num_quarters > 0:
            df = df.tail(self.num_quarters)
        df = df.reset_index(drop=True)
        df["_date"] = df[date_col]
        return df

    # ------------------------------------------------------------------
    # Category data preparation
    # ------------------------------------------------------------------
    def _prepare_category_data(self, category: str) -> Optional[pd.DataFrame]:
        """Collect the relevant columns for a category from the Symbol's
        pre-enriched quarterly DataFrame.

        Returns ``None`` if the source table is missing.
        """
        cfg = FINANCIAL_CATEGORIES[category]
        df = self._get_quarterly_df(cfg["source_table"])
        if df is None:
            return None

        result_cols = ["_date"]
        for col_name in cfg["columns"]:
            col_map = {c.lower(): c for c in df.columns}
            actual = col_map.get(col_name.lower())
            if actual and actual in df.columns:
                df[actual] = pd.to_numeric(df[actual], errors="coerce")
                if actual != col_name:
                    df[col_name] = df[actual]
                result_cols.append(col_name)

        available = [c for c in result_cols if c in df.columns]
        if len(available) <= 1:
            return None
        return df[available]

    def _format_category_summary(self, category: str, df: pd.DataFrame) -> str:
        """Create a human-readable text summary of a category's quarterly data."""
        lines = [f"=== {category.upper().replace('_', ' ')} ==="]
        lines.append(f"Quarters: {len(df)}")
        lines.append("")

        date_col = "_date" if "_date" in df.columns else None
        for col in df.columns:
            if col == "_date":
                continue
            vals = df[col].tolist()
            if date_col:
                dates = df[date_col].dt.strftime("%Y-%m-%d").tolist()
                paired = [f"  {d}: {v}" for d, v in zip(dates, vals)]
                lines.append(f"{col}:")
                lines.extend(paired)
            else:
                lines.append(f"{col}: {vals}")
            lines.append("")

        numeric_cols = [c for c in df.columns if c != "_date"]
        if numeric_cols:
            lines.append("Summary statistics:")
            for col in numeric_cols:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series) > 1:
                    latest = series.iloc[-1]
                    trend = "improving" if series.iloc[-1] > series.iloc[-2] else "declining"
                    lines.append(
                        f"  {col}: latest={latest:.4f}, mean={series.mean():.4f}, "
                        f"std={series.std():.4f}, trend={trend}"
                    )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Category evaluation
    # ------------------------------------------------------------------
    def _build_category_prompt(self, category: str, data_summary: str, article: str) -> str:
        """Compose the user prompt for a single financial category evaluation."""
        period_desc = (
            f"the last {self.num_quarters} quarters"
            if self.num_quarters
            else "available history"
        )
        return (
            f"You are a senior financial analyst evaluating a company's "
            f"**{category.replace('_', ' ')}** based on {period_desc} of "
            f"quarterly financial data.\n\n"
            f"### Data\n{data_summary}\n\n"
            f"### Reference article\n{article}\n\n"
            "### Instructions\n"
            "1. Analyse the trends, absolute levels, and any red flags.\n"
            "2. Consider what these numbers imply about the company being "
            "underpriced or overpriced.\n"
            "3. Output a single continuous number between -1 and 1:\n"
            "   -1 = strongly underpriced (buy signal)\n"
            "    0 = fairly priced\n"
            "   +1 = strongly overpriced (sell signal)\n"
            "4. After the number, provide a 2-4 sentence explanation.\n"
            "5. Focus on the most recent trends (last 1-2 quarters) while "
            "keeping historical context.\n"
        )

    def evaluate_category(
        self, category: str
    ) -> Tuple[Optional[float], str, Optional[pd.DataFrame]]:
        """Evaluate a single financial category.

        Returns ``(score, explanation, prepared_dataframe)``.
        """
        df = self._prepare_category_data(category)
        if df is None or df.empty:
            return None, "No data available", None

        summary = self._format_category_summary(category, df)
        article_key = FINANCIAL_CATEGORIES[category]["article_key"]
        article = self._article_texts.get(article_key, "")

        prompt = self._build_category_prompt(category, summary, article)
        reply = self._call_llm(user_prompt=prompt)
        score = self._parse_score(reply) if reply else None
        return score, reply, df

    # ------------------------------------------------------------------
    # DB storage
    # ------------------------------------------------------------------
    def _store_results(self, result: dict) -> None:
        """Persist financial analysis results to the analyst DB."""
        try:
            manager = get_analyst_data_handler()
            symbol_str = result["symbol"]
            epoch = result["epoch"]
            categories_evaluated = result["categories_evaluated"]
            scores = result["category_scores"]
            explanations = result["explanations"]

            base_row = {
                "symbol": symbol_str,
                "epoch": epoch,
                "last_date": pd.to_datetime(result["last_date"]) if result["last_date"] else None,
                "num_quarters": result["num_quarters"],
                "profitability_used": int("profitability" in categories_evaluated),
                "revenue_growth_used": int("revenue_growth" in categories_evaluated),
                "earnings_used": int("earnings" in categories_evaluated),
                "liquidity_used": int("liquidity" in categories_evaluated),
                "leverage_used": int("leverage" in categories_evaluated),
                "cash_flow_health_used": int("cash_flow_health" in categories_evaluated),
                "analysis_run_timestamp": result["analysis_run_timestamp"],
                "analysis_run_datetime": pd.to_datetime(result["analysis_run_datetime"]),
                "warnings": ",".join(result["warnings"]) if result["warnings"] else None,
                "errors": ",".join(result["errors"]) if result["errors"] else None,
                "aggregate_score": result["aggregate_score"],
            }
            score_row = {
                "symbol": symbol_str,
                "epoch": epoch,
                "aggregate_score": result["aggregate_score"],
                **{cat: scores.get(cat) for cat in FINANCIAL_CATEGORIES},
            }
            explanation_row = {
                "symbol": symbol_str,
                "epoch": epoch,
                **{cat: explanations.get(cat) for cat in FINANCIAL_CATEGORIES},
            }

            manager.insert_new_data(table=AnalystFinancialBase, rows=[base_row])
            manager.insert_new_data(table=AnalystFinancialScore, rows=[score_row])
            manager.insert_new_data(table=AnalystFinancialExplanation, rows=[explanation_row])
            self.logger.info(
                "Financial analysis stored in DB for symbol %s, epoch %s",
                symbol_str, epoch,
            )
        except Exception as exc:
            self.logger.error("Failed to store financial analysis in DB: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run_analysis(self, store_in_db: bool | None = None) -> dict:
        """Run financial analysis across all categories.

        Parameters
        ----------
        store_in_db : bool | None
            Override the instance-level ``store_in_db`` flag.

        Returns
        -------
        dict
        """
        should_store = store_in_db if store_in_db is not None else self.store_in_db

        if not self._article_texts:
            self.load_articles()

        scores: Dict[str, Optional[float]] = {}
        explanations: Dict[str, str] = {}
        category_data: Dict[str, Optional[pd.DataFrame]] = {}
        warnings: List[str] = []
        errors: List[str] = []
        categories_evaluated: List[str] = []

        for category in FINANCIAL_CATEGORIES:
            try:
                score, explanation, df = self.evaluate_category(category)
                scores[category] = score
                explanations[category] = explanation
                category_data[category] = df
                categories_evaluated.append(category)
                if explanation == "No data available":
                    warnings.append(f"No data for category: {category}")
                elif score is None:
                    warnings.append(f"No score returned for category: {category}")
            except Exception as exc:
                self.logger.error("Error evaluating category '%s': %s", category, exc)
                errors.append(f"Error evaluating {category}: {exc}")

        aggregate = self._compute_aggregate(scores)
        symbol_str = self._resolve_symbol_str()

        # Determine last fiscal date across all category data
        last_date = None
        for df in category_data.values():
            if df is not None and "_date" in df.columns:
                candidate = df["_date"].max()
                if last_date is None or candidate > last_date:
                    last_date = candidate
        last_epoch = int(last_date.timestamp()) if last_date is not None else None
        last_date_str = (
            last_date.strftime("%Y-%m-%d") if last_date is not None else None
        )

        analysis_run_dt = datetime.datetime.now()
        result = {
            "symbol": symbol_str,
            "epoch": last_epoch,
            "last_date": last_date_str,
            "num_quarters": self.num_quarters,
            "categories_evaluated": categories_evaluated,
            "analysis_run_timestamp": int(analysis_run_dt.timestamp()),
            "analysis_run_datetime": analysis_run_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "warnings": warnings,
            "errors": errors,
            "category_scores": scores,
            "aggregate_score": aggregate,
            "explanations": explanations,
        }

        self.logger.info(
            "Financial analysis complete for %s - aggregate=%.3f, categories=%s",
            symbol_str,
            aggregate if aggregate is not None else float("nan"),
            categories_evaluated,
        )

        if should_store:
            self._store_results(result)

        return result
