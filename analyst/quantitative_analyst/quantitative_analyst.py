"""
quantitative_analyst.py
-----------------------

Concrete ``Analyst`` subclass for technical-indicator analysis.

Operates on a ``Symbol`` instance whose ``daily_timeseries`` DataFrame
has been enriched with indicator columns by
``add_indicators_to_price_data.py`` (SMA, EMA, RSI, MACD, Bollinger
Bands, VWAP, Stochastic, OBV, ADX).

For each configured indicator the analyst:

1. Extracts the last year of values from the ``daily_timeseries``.
2. Sends the series together with a reference Investopedia PDF article
   to the language model.
3. Receives a continuous score in ``[-1, 1]`` (-1 = underpriced,
   +1 = overpriced).

An aggregate score is the average of all valid indicator scores.

Usage::

    from symbol.symbol import Symbol
    from analyst.quantitative_analyst.quantitative_analyst import QuantitativeAnalyst

    storage = Symbol(adapter, symbol="MSFT")
    qa = QuantitativeAnalyst(storage)
    result = qa.run_analysis()
    print(result["aggregate_score"])
"""

from __future__ import annotations

import datetime
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from symbol.symbol import Symbol
from analyst.analyst import Analyst
from infrastructure.databases.analysis.postgre_manager.analyst_data_manager import get_analyst_data_handler
from infrastructure.databases.analysis.postgre_manager.analyst_table_objects import (
    AnalystQuantitativeBase,
    AnalystQuantitativeScore,
    AnalystQuantitativeExplanation,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Default article paths
# ---------------------------------------------------------------------------
PDF_PATH = "./infrastructure/databases/rag_knowledge_base/quantitative_analyst/"

ARTICLE_PATHS = {
    "obv": PDF_PATH + "On-Balance Volume (OBV)_ How It Works and How to Use It.pdf",
    "vwap": PDF_PATH + "Volume-Weighted Average Price (VWAP)_ Definition and Calculation.pdf",
    "macd": PDF_PATH + "MACD_ A Primer.pdf",
    "rsi": PDF_PATH + "Relative Strength Index (RSI)_ What It Is, How It Works, and Formula.pdf",
    "stochastic_oscillator": PDF_PATH + "What Is the Stochastic Oscillator and How Is It Used_.pdf",
    "bollinger_bands": PDF_PATH + "Understanding Bollinger Bands_ A Key Technical Analysis Tool for Investors.pdf",
    "sma": PDF_PATH + "Simple Moving Average (SMA) Explained_ Definition and Calculation Formula.pdf",
    "ema": PDF_PATH + "Exponential Moving Average (EMA)_ Definition, Formula, and Usage.pdf",
    "adx": PDF_PATH + "ADX Explained_ How to Measure and Trade Trend Strength.pdf",
}

SYSTEM_PROMPT = (
    "You are a senior financial analyst, responsible for evaluating "
    "stock price indicators."
)


class QuantitativeAnalyst(Analyst):
    """Assess whether a stock appears under- or overpriced using technical indicators.

    Parameters
    ----------
    symbol : Symbol
        Instance with ``daily_timeseries`` containing indicator columns.
    article_paths : dict[str, str]
        Mapping from indicator names to PDF file paths.
    openai_key : str | None
        Explicit OpenAI key; falls back to ``KEY_FOR_QUANT_ANALYST``.
    rate_limit_per_minute : int
        Max OpenAI calls per 60 s.
    logger : logging.Logger | None
    store_in_db : bool
        Whether to persist results to the analyst database.
    """

    def __init__(
        self,
        symbol: Symbol = None,
        article_paths: Optional[Dict[str, str]] = None,
        openai_key: Optional[str] = None,
        rate_limit_per_minute: int = 60,
        logger: Optional[logging.Logger] = None,
        store_in_db: bool = True,
    ) -> None:
        super().__init__(
            symbol=symbol,
            article_paths=article_paths or ARTICLE_PATHS,
            openai_key=openai_key,
            env_key_name="KEY_FOR_QUANT_ANALYST",
            model="gpt-5",
            system_prompt=SYSTEM_PROMPT,
            rate_limit_per_minute=rate_limit_per_minute,
            logger=logger,
            store_in_db=store_in_db,
        )
        # Embeddings cache (optional, for future use)
        self._article_embeddings: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Embedding (optional, kept for forward compatibility)
    # ------------------------------------------------------------------
    def embed_articles(self, model: str = "text-embedding-ada-002") -> None:
        """Create vector embeddings for each loaded article."""
        if not self._article_texts:
            self.load_articles()
        self._article_embeddings.clear()
        for indicator, text in self._article_texts.items():
            if not text:
                self._article_embeddings[indicator] = []
                continue
            self.rate_limiter.record_call()
            try:
                response = self._client.embeddings.create(input=[text], model=model)
                self._article_embeddings[indicator] = response.data[0].embedding
            except Exception as exc:
                self.logger.error("Embedding error for %s: %s", indicator, exc)
                self._article_embeddings[indicator] = []

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _get_last_year_data(self, cutoff_date: str | None = None) -> pd.DataFrame:
        """Return the last year of ``daily_timeseries`` data."""
        df = getattr(self.symbol, "daily_timeseries", None)
        if df is None or df.empty:
            raise ValueError("Symbol has no daily_timeseries data loaded")
        data = df.copy()
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
        else:
            data = data.reset_index().rename(columns={"index": "date"})
            data["date"] = pd.to_datetime(data["date"])
        if cutoff_date is not None:
            max_date = pd.to_datetime(cutoff_date)
            data = data[data["date"] <= max_date]
        else:
            max_date = data["date"].max()
        cutoff = max_date - pd.Timedelta(days=365)
        return data[data["date"] >= cutoff]

    def _find_indicator_column(self, df: pd.DataFrame, indicator: str) -> Optional[str]:
        """Heuristically find a column matching the indicator name."""
        key = indicator.replace("_", "").lower()
        candidates = sorted(
            col for col in df.columns if key in col.replace("_", "").lower()
        )
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # Indicator evaluation
    # ------------------------------------------------------------------
    def _build_indicator_prompt(self, indicator: str, values: List[float], article: str) -> str:
        """Compose the user prompt for a single indicator evaluation."""
        return (
            f"You are a financial analyst assessing the {indicator} values "
            f"for a company over the past year.\n"
            f"Here are the series of {indicator} values (most recent last):\n"
            f"{values}\n\n"
            f"Refer to the following article describing how to interpret this indicator:\n"
            f"{article}\n\n"
            "I want you to focus on the recent trends from the last week and month, "
            "but you can consider the whole series.  "
            "Look for patterns such as sustained increases or decreases, extreme values, "
            "or recent reversals.  "
            "Consider how these patterns relate to the article's guidance on what "
            "constitutes underpriced or overpriced signals.\n\n"
            "Based on the values and the article, output a single continuous "
            "number between -1 and 1 representing how underpriced (-1) or "
            "overpriced (+1) the company appears according to this indicator. "
            "After the number, provide a brief explanation of your reasoning. "
            "Based on your analysis I want to be able to decide if this indicator "
            "is signaling a potential buying opportunity (close to -1), "
            "a selling opportunity (close to +1), or if it's neutral (close to 0)."
        )

    def evaluate_indicator(
        self, indicator: str, cutoff_date: str | None = None
    ) -> Tuple[Optional[float], str]:
        """Evaluate a single indicator and return ``(score, explanation)``."""
        df = self._get_last_year_data(cutoff_date=cutoff_date)
        column_name = self._find_indicator_column(df, indicator)
        if not column_name:
            self.logger.warning("No column found for indicator '%s'", indicator)
            return None, "No data available"
        values_series = df[column_name].dropna()
        if values_series.empty:
            self.logger.warning("No values for indicator '%s' in last year", indicator)
            return None, "No data available"
        values_list = values_series.tolist()
        article = self._article_texts.get(indicator, "")
        prompt = self._build_indicator_prompt(indicator, values_list, article)
        reply = self._call_llm(user_prompt=prompt, temperature=None)
        score = self._parse_score(reply) if reply else None
        return score, reply

    # ------------------------------------------------------------------
    # DB storage
    # ------------------------------------------------------------------
    def _store_results(self, result: dict) -> None:
        """Persist quantitative analysis results to the analyst DB."""
        try:
            manager = get_analyst_data_handler()
            symbol_str = result["symbol"]
            epoch = result["epoch"]
            indicators_evaluated = result["indicators_evaluated"]
            scores = result["indicator_scores"]
            explanations = result["explanations"]

            base_row = {
                "symbol": symbol_str,
                "epoch": epoch,
                "last_date": pd.to_datetime(result["last_date"]) if result["last_date"] else None,
                "num_data_points": result["num_data_points"],
                "obv_used": int("obv" in indicators_evaluated),
                "vwap_used": int("vwap" in indicators_evaluated),
                "macd_used": int("macd" in indicators_evaluated),
                "rsi_used": int("rsi" in indicators_evaluated),
                "stochastic_oscillator_used": int("stochastic_oscillator" in indicators_evaluated),
                "bollinger_bands_used": int("bollinger_bands" in indicators_evaluated),
                "sma_used": int("sma" in indicators_evaluated),
                "ema_used": int("ema" in indicators_evaluated),
                "adx_used": int("adx" in indicators_evaluated),
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
                **{ind: scores.get(ind) for ind in self.article_paths},
            }
            explanation_row = {
                "symbol": symbol_str,
                "epoch": epoch,
                **{ind: explanations.get(ind) for ind in self.article_paths},
            }

            manager.insert_new_data(table=AnalystQuantitativeBase, rows=[base_row])
            manager.insert_new_data(table=AnalystQuantitativeScore, rows=[score_row])
            manager.insert_new_data(table=AnalystQuantitativeExplanation, rows=[explanation_row])
            self.logger.info(
                "Analysis stored in DB for symbol %s, epoch %s", symbol_str, epoch
            )
        except Exception as exc:
            self.logger.error("Failed to store analysis in DB: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run_analysis(
        self,
        store_in_db: bool | None = None,
        cutoff_date: str | None = None,
    ) -> dict:
        """Run quantitative analysis across all configured indicators.

        Parameters
        ----------
        store_in_db : bool | None
            Override the instance-level ``store_in_db`` flag.
        cutoff_date : str | None
            Optional end-date for the 1-year lookback window.

        Returns
        -------
        dict
        """
        should_store = store_in_db if store_in_db is not None else self.store_in_db

        if not self._article_texts:
            self.load_articles()

        scores: Dict[str, Optional[float]] = {}
        explanations: Dict[str, str] = {}
        warnings: List[str] = []
        errors: List[str] = []
        indicators_evaluated: List[str] = []

        # Get metadata from the price data
        try:
            df = self._get_last_year_data(cutoff_date=cutoff_date)
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
        except Exception as exc:
            num_data_points = 0
            last_epoch = None
            last_date_str = None
            errors.append(f"Error getting last year data: {exc}")

        # Evaluate each indicator
        for indicator in self.article_paths:
            try:
                score, explanation = self.evaluate_indicator(indicator, cutoff_date=cutoff_date)
                scores[indicator] = score
                explanations[indicator] = explanation
                indicators_evaluated.append(indicator)
                if explanation == "No data available":
                    warnings.append(f"No data for indicator: {indicator}")
                elif score is None:
                    warnings.append(f"No score for indicator: {indicator}")
            except Exception as exc:
                errors.append(f"Error evaluating {indicator}: {exc}")

        aggregate = self._compute_aggregate(scores)
        symbol_str = self._resolve_symbol_str()
        analysis_run_dt = datetime.datetime.now()

        result = {
            "symbol": symbol_str,
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

        self.logger.info(
            "Quantitative analysis complete for %s - aggregate=%.3f, indicators=%s",
            symbol_str,
            aggregate if aggregate is not None else float("nan"),
            indicators_evaluated,
        )

        if should_store:
            self._store_results(result)

        return result
