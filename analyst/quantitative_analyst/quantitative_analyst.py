"""
quantitative_analyst.py
-----------------------

This module defines a ``QuantitativeAnalyst`` class which ties together
price history, technical indicator values, supporting educational
material, and a language model to score whether a company appears
under‑ or overpriced.  The class operates on a ``Symbol`` instance from
``symbol/symbol.py`` (which holds a ``daily_timeseries`` DataFrame with
indicator columns added via ``add_indicators_to_price_data.py``).  It
fetches Investopedia articles stored as PDF files, generates vector
representations of them using OpenAI embeddings, and then invokes an
OpenAI chat model to interpret each indicator’s last‑year time series
relative to the article.  The model returns a continuous value in
``[-1, 1]`` for each indicator and an optional explanation.  An
aggregate score is computed by averaging all numeric indicator
scores.

To use this class you must provide:

* A ``Symbol`` instance already populated with price data and
  indicator columns.
* A mapping from indicator names to PDF file paths in the local
  repository.
* A valid OpenAI API key.

Example::

    from symbol.symbol import Symbol
    from analyst.quantitative_analyst.quantitative_analyst import QuantitativeAnalyst

    storage = Symbol(adapter, symbol="MSFT")
    article_paths = {
        "sma": "knowledge/investopedia/sma.pdf",
        "ema": "knowledge/investopedia/ema.pdf",
        # ... add other indicator PDFs here
    }
    qa = QuantitativeAnalyst(storage, article_paths, openai_key=my_key)
    results, agg, details = qa.run_analysis()

The returned ``results`` dictionary contains the continuous
underpriced/overpriced score per indicator, ``agg`` contains the
average score across all indicators, and ``details`` contains the
raw language‑model explanations.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from symbol.symbol import Symbol
from infrastructure.databases.analysis.postgre_manager.analyst_data_manager import get_analyst_data_handler
from infrastructure.databases.analysis.postgre_manager.analyst_table_objects import AnalystQuantitativeBase, AnalystQuantitativeScore, AnalystQuantitativeExplanation

load_dotenv()

PDF_PATH = "./infrastructure/databases/rag_knowledge_base/quantitative_analyst/"

ARTICLE_PATHS = {
    'obv': PDF_PATH + 'On-Balance Volume (OBV)_ How It Works and How to Use It.pdf',
    'vwap': PDF_PATH + 'Volume-Weighted Average Price (VWAP)_ Definition and Calculation.pdf',
    'macd': PDF_PATH + 'MACD_ A Primer.pdf',
    'rsi': PDF_PATH + 'Relative Strength Index (RSI)_ What It Is, How It Works, and Formula.pdf',
    'stochastic_oscillator': PDF_PATH + 'What Is the Stochastic Oscillator and How Is It Used_.pdf',
    'bollinger_bands': PDF_PATH + 'Understanding Bollinger Bands_ A Key Technical Analysis Tool for Investors.pdf',
    'sma': PDF_PATH + 'Simple Moving Average (SMA) Explained_ Definition and Calculation Formula.pdf',
    'ema': PDF_PATH + 'Exponential Moving Average (EMA)_ Definition, Formula, and Usage.pdf',
    'adx': PDF_PATH + 'ADX Explained_ How to Measure and Trade Trend Strength.pdf'
}

@dataclass
class RateLimiter:
    """Simple rate limiter to enforce a maximum number of API calls per minute.

    The limiter tracks the timestamps of the last calls and sleeps when
    the number of calls in the last 60 seconds would exceed the
    configured limit.  This is a best‑effort mechanism; callers should
    still abide by the terms of their API agreement.

    Attributes
    ----------
    max_calls_per_minute: int
        Maximum number of API calls allowed in a 60‑second window.
    logger: logging.Logger
        Logger used to emit informational messages about rate limiting.
    timestamps: list[float]
        Internal list of call timestamps (unix epoch seconds).
    """

    max_calls_per_minute: int = 60
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    timestamps: List[float] = field(default_factory=list)

    def record_call(self) -> None:
        """Record a call timestamp and sleep if the rate limit is exceeded."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self.timestamps = [t for t in self.timestamps if now - t < 60]
        if len(self.timestamps) >= self.max_calls_per_minute:
            # Sleep until one minute has elapsed since the oldest call
            sleep_duration = 60 - (now - self.timestamps[0])
            if sleep_duration > 0:
                self.logger.info(
                    "Rate limit exceeded: sleeping %.2f seconds to respect %d calls/minute",
                    sleep_duration,
                    self.max_calls_per_minute,
                )
                time.sleep(sleep_duration)
            # Recompute now and trim again after sleep
            now = time.time()
            self.timestamps = [t for t in self.timestamps if now - t < 60]
        # Record this call
        self.timestamps.append(time.time())


class QuantitativeAnalyst:
    """A class to assess whether a stock appears under‑ or overpriced.

    Parameters
    ----------
    symbol : Symbol
        Instance containing price history and indicator columns.  The
        analyst does not modify this object but reads from its
        ``daily_timeseries`` attribute.
    article_paths : dict[str, str]
        Mapping from indicator names (e.g. ``"sma"``) to PDF file paths
        relative to the project root.  The class reads these files and
        passes their content to the language model.
    openai_key : str
        API key for OpenAI.  If provided, it will be set on the
        ``openai`` module.  If ``None``, the existing environment
        variable ``OPENAI_API_KEY`` will be used.
    rate_limit_per_minute : int, optional
        Maximum number of OpenAI API calls per minute.  Defaults to 60.
    logger : logging.Logger, optional
        Logger for informational and error messages.  If omitted, a
        module‑level logger is used.
    """

    def __init__(
        self,
        symbol: Symbol = None,
        article_paths: Dict[str, str] = ARTICLE_PATHS,
        openai_key: Optional[str] = None,
        rate_limit_per_minute: int = 60,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.symbol = symbol
        self.article_paths = article_paths
        self.logger = logger or logging.getLogger(__name__)
        self.store_in_db = True
        self.rate_limiter = RateLimiter(max_calls_per_minute=rate_limit_per_minute, logger=self.logger)
        # Configure OpenAI API key: prefer argument, else from environment (.env)
        if openai_key:
            self._api_key = openai_key
        else:
            self._api_key = os.getenv("KEY_FOR_QUANT_ANALYST")
        self._client = openai.OpenAI(api_key=self._api_key)
        # Containers for article text and embeddings
        self._article_texts: Dict[str, str] = {}
        self._article_embeddings: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Article loading and embedding
    # ------------------------------------------------------------------
    def _read_pdf(self, path: str) -> str:
        """Read a PDF file and return its concatenated text.

        If the file cannot be read, an error is logged and an empty
        string is returned.
        """
        abs_path = os.path.abspath(path)
        try:
            reader = PdfReader(abs_path)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            return text
        except Exception as exc:
            self.logger.error("Failed to read PDF %s: %s", abs_path, exc)
            return ""

    def load_articles(self) -> None:
        """Load all Investopedia article PDFs into memory.

        This method iterates over ``article_paths`` and reads each PDF
        using :func:`_read_pdf`.  The resulting text is stored in
        ``_article_texts``.  Existing contents are overwritten.
        """
        self._article_texts.clear()
        for indicator, rel_path in self.article_paths.items():
            text = self._read_pdf(rel_path)
            if not text:
                self.logger.warning(
                    "No text extracted from %s for indicator %s", rel_path, indicator
                )
            self._article_texts[indicator] = text

    def embed_articles(self, model: str = "text-embedding-ada-002") -> None:
        """Create vector embeddings for each loaded article.

        This method will implicitly call :func:`load_articles` if the
        article texts have not yet been loaded.  Each article text is
        submitted to OpenAI's embedding endpoint one at a time.  A
        rate limiter ensures compliance with the configured calls per
        minute.  Embeddings are stored in ``_article_embeddings``.
        """
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
                embedding = response.data[0].embedding
                self._article_embeddings[indicator] = embedding
            except Exception as exc:
                self.logger.error("Embedding error for %s: %s", indicator, exc)
                self._article_embeddings[indicator] = []

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _get_last_year_data(self, cutoff_date: str | None = None) -> pd.DataFrame:
        """Return a slice of the ``daily_timeseries`` from the last year.

        The ``daily_timeseries`` DataFrame is expected to have a ``date``
        column convertible to datetime.  The slice includes all rows
        where the ``date`` is within 365 days of ``cutoff_date`` (or
        the most recent observation when *cutoff_date* is ``None``).

        Args:
            cutoff_date: Optional end-date for the 1-year window
                (ISO string or datetime).  When ``None`` the latest
                date in the data is used.
        """
        df = getattr(self.symbol, "daily_timeseries", None)
        if df is None or df.empty:
            raise ValueError("Symbol has no daily_timeseries data loaded")
        data = df.copy()
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
        else:
            # If no explicit date column exists, use the index as a date
            data = data.reset_index().rename(columns={"index": "date"})
            data["date"] = pd.to_datetime(data["date"])
        if cutoff_date is not None:
            max_date = pd.to_datetime(cutoff_date)
            # Only keep rows up to the cutoff date
            data = data[data["date"] <= max_date]
        else:
            max_date = data["date"].max()
        cutoff = max_date - pd.Timedelta(days=365)
        return data[data["date"] >= cutoff]

    def _find_indicator_column(self, df: pd.DataFrame, indicator: str) -> Optional[str]:
        """Heuristically find a column matching the indicator name.

        The search is case‑insensitive and ignores underscores.  For
        example, ``"sma"`` will match columns containing "sma" but not
        "ema".  If multiple columns match (for instance, multiple
        windows), the first match in alphabetical order is returned.
        """
        key = indicator.replace("_", "").lower()
        candidates = []
        for col in df.columns:
            col_key = col.replace("_", "").lower()
            if key in col_key:
                candidates.append(col)
        candidates.sort()
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # Indicator evaluation
    # ------------------------------------------------------------------
    def _call_language_model(
        self, indicator: str, values: List[float], article: str
    ) -> Tuple[Optional[float], str]:
        """Invoke the OpenAI chat model to score an indicator.

        The prompt instructs the model to produce a continuous value in
        ``[-1, 1]`` and to explain its reasoning.  The first numeric
        value found in the response is parsed as the score.  If the
        call fails or no number is present, ``None`` is returned for
        the score.
        """
        # Compose the user prompt
        prompt = (
            f"You are a financial analyst assessing the {indicator} values "
            f"for a company over the past year.\n"
            f"Here are the series of {indicator} values (most recent last):\n"
            f"{values}\n\n"
            f"Refer to the following article describing how to interpret this indicator:\n"
            f"{article}\n\n"
            "Based on the values and the article, output a single continuous "
            "number between -1 and 1 representing how underpriced (-1) or "
            "overpriced (+1) the company appears according to this indicator. "
            "After the number, provide a brief explanation of your reasoning."
        )
        self.rate_limiter.record_call()
        try:
            # openai>=1.0.0 API: use client object
            chat_completion = self._client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst, responsible for evaluating stock price indicators."},
                    {"role": "user", "content": prompt},
                ],
                # temperature=0.3,
                # max_tokens=200,
            )
            print(chat_completion)
            reply = chat_completion.choices[0].message.content.strip()
        except Exception as exc:
            self.logger.error("Error calling language model for %s: %s", indicator, exc)
            return None, ""
        # Extract first floating point number from the response
        match = re.search(r"-?\d*\.\d+|-?\d+", reply)
        score: Optional[float] = None
        if match:
            try:
                score = float(match.group())
                # Clip to [-1, 1] as a safety measure
                score = max(min(score, 1.0), -1.0)
            except ValueError:
                score = None
        return score, reply

    def evaluate_indicator(self, indicator: str, cutoff_date: str | None = None) -> Tuple[Optional[float], str]:
        """Evaluate a single indicator and return its score and explanation.

        Args:
            indicator: Name of the indicator to evaluate.
            cutoff_date: Optional end-date for the 1-year lookback window.
        """
        df = self._get_last_year_data(cutoff_date=cutoff_date)
        column_name = self._find_indicator_column(df, indicator)
        if not column_name:
            self.logger.warning("No column found in data for indicator '%s'", indicator)
            return None, "No data available"
        values_series = df[column_name].dropna()
        if values_series.empty:
            self.logger.warning("No values for indicator '%s' in last year", indicator)
            return None, "No data available"
        values_list = values_series.tolist()
        article = self._article_texts.get(indicator, "")
        return self._call_language_model(indicator, values_list, article)

    def run_analysis(self, store_in_db: bool = False, db_manager=None, cutoff_date: str | None = None):
        """
        Run the analysis for all configured indicators and return extended metadata.
        If store_in_db is True, store the result in the analyst database using ORM objects.
        Args:
            store_in_db (bool): If True, store the result in the database.
            db_manager: Optional AnalystDataManager instance. If None and store_in_db is True, a new one will be created.
            cutoff_date: Optional end-date for the 1-year lookback window (ISO string).
        Returns:
            dict: The analysis result as before.
        """
        import datetime
        # Ensure articles are loaded; embeddings are optional but may improve prompt quality in future iterations
        if not self._article_texts:
            self.load_articles()
        scores: Dict[str, Optional[float]] = {}
        explanations: Dict[str, str] = {}
        warnings: List[str] = []
        errors: List[str] = []
        indicators_evaluated: List[str] = []
        # Get the last year data and epoch
        try:
            df = self._get_last_year_data(cutoff_date=cutoff_date)
            num_data_points = len(df)
            if "date" in df.columns:
                last_date = df["date"].max()
                if hasattr(last_date, 'to_pydatetime'):
                    last_date = last_date.to_pydatetime()
                last_epoch = int(last_date.timestamp())
                last_date_str = last_date.strftime("%Y-%m-%d %H:%M:%S")
            else:
                last_epoch = None
                last_date_str = None
        except Exception as exc:
            df = None
            num_data_points = 0
            last_epoch = None
            last_date_str = None
            errors.append(f"Error getting last year data: {exc}")
        for indicator in self.article_paths.keys():
            try:
                score, explanation = self.evaluate_indicator(indicator, cutoff_date=cutoff_date)
                scores[indicator] = score
                explanations[indicator] = explanation
                indicators_evaluated.append(indicator)
                if explanation == "No data available":
                    warnings.append(f"No data for indicator: {indicator}")
                if score is None and explanation != "No data available":
                    warnings.append(f"No score for indicator: {indicator}")
            except Exception as exc:
                errors.append(f"Error evaluating {indicator}: {exc}")
        # Compute aggregate as mean of valid numeric scores
        numeric_scores = [s for s in scores.values() if isinstance(s, (int, float))]
        aggregate: Optional[float] = None
        if numeric_scores:
            aggregate = sum(numeric_scores) / len(numeric_scores)
            aggregate = max(min(aggregate, 1.0), -1.0)
        # Compose result
        # Try to get the symbol string from the Symbol instance
        symbol_str = getattr(self.symbol, "symbol", None)
        if symbol_str is None:
            symbol_str = getattr(self.symbol, "_symbol", None)
        if symbol_str is None and hasattr(self.symbol, "ticker"):
            symbol_str = getattr(self.symbol, "ticker", None)
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
        # Optionally store in DB
        if self.store_in_db:
            try:
                manager = db_manager or get_analyst_data_handler()
                base_row = {
                    "symbol": symbol_str,
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
                    "symbol": symbol_str,
                    "epoch": last_epoch,
                    "aggregate_score": aggregate,
                    "obv": scores.get("obv"),
                    "vwap": scores.get("vwap"),
                    "macd": scores.get("macd"),
                    "rsi": scores.get("rsi"),
                    "stochastic_oscillator": scores.get("stochastic_oscillator"),
                    "bollinger_bands": scores.get("bollinger_bands"),
                    "sma": scores.get("sma"),
                    "ema": scores.get("ema"),
                    "adx": scores.get("adx"),
                }
                explanation_row = {
                    "symbol": symbol_str,
                    "epoch": last_epoch,
                    "obv": explanations.get("obv"),
                    "vwap": explanations.get("vwap"),
                    "macd": explanations.get("macd"),
                    "rsi": explanations.get("rsi"),
                    "stochastic_oscillator": explanations.get("stochastic_oscillator"),
                    "bollinger_bands": explanations.get("bollinger_bands"),
                    "sma": explanations.get("sma"),
                    "ema": explanations.get("ema"),
                    "adx": explanations.get("adx"),
                }
                manager.insert_new_data(table=AnalystQuantitativeBase, rows=[base_row])
                manager.insert_new_data(table=AnalystQuantitativeScore, rows=[score_row])
                manager.insert_new_data(table=AnalystQuantitativeExplanation, rows=[explanation_row])
                self.logger.info("Analysis results stored in analyst database for symbol %s, epoch %s", symbol_str, last_epoch)
            except Exception as exc:
                self.logger.error("Failed to store analysis in DB: %s", exc, exc_info=True)
        return result