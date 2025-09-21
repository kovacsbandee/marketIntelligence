"""
financial_analyst.py
--------------------

This module defines a ``FinancialAnalyst`` class which parallels the
``QuantitativeAnalyst`` used for technical indicators, but operates on
fundamental company data instead of price‑based signals.  It analyzes
quarterly financial statements (income statements, balance sheets and
cash flow statements) loaded into a ``Symbol`` instance, retrieves
accompanying Investopedia articles (stored as PDFs) that explain how
each metric should be interpreted, and then asks a language model to
judge whether the company appears undervalued or overvalued relative
to each metric.  Scores are returned on a continuous scale of
``[-1, 1]`` with ``-1`` indicating strongly underpriced and ``+1``
indicating strongly overpriced.  An aggregate score is computed as
the average of all valid metric scores.

The typical usage pattern mirrors that of the ``QuantitativeAnalyst``:

```
from symbol.symbol import Symbol
from financial_analyst import FinancialAnalyst

storage = Symbol(adapter, symbol="AAPL")
article_paths = {
    "total_revenue": "knowledge/investopedia/total_revenue.pdf",
    "net_income": "knowledge/investopedia/net_income.pdf",
    # ... add more metric PDFs here
}
fa = FinancialAnalyst(storage, article_paths, openai_key=my_key, num_quarters=8)
scores, agg, details = fa.run_analysis()
```

To customise the analysis window, set ``num_quarters`` to the number
of most recent quarterly observations you wish to include.  Passing
``None`` uses the entire available history.  The class will search
across all loaded quarterly tables for each metric and will use the
first table that contains the metric.

Notes
-----
* The ``Symbol`` class loads database tables as attributes whose
  names correspond to the table names in the database.  Quarterly
  statement tables are typically named with the suffix "quarterly"; in
  some parts of the codebase the word is misspelled as "quaterly".
  This class is resilient to both conventions: it inspects all
  attributes of the ``Symbol`` instance and collects any DataFrame
  whose name contains ``income_statement``, ``balance_sheet`` or
  ``cash_flow`` and ends with either "quarterly" or "quaterly".  Missing
  tables are ignored.
* Investopedia articles must be provided via ``article_paths``; the
  class will read them from disk using ``PyPDF2``.  No web access is
  performed at runtime.
* The language model is called via OpenAI's ChatCompletion API.  A
  ``RateLimiter`` is employed to avoid exceeding a user‑specified
  number of API calls per minute.  See ``QuantitativeAnalyst`` for a
  similar implementation.
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

from symbol.symbol import Symbol

PDF_PATH = "./infrastructure/databases/rag_knowledge_base/financial_analyst/"

# Example ARTICLE_PATHS for financial metrics (update with actual PDF names as needed)

ARTICLE_PATHS = {
    'balance_sheet': PDF_PATH + 'Breaking Down the Balance Sheet.pdf',
    'income_statement': PDF_PATH + 'Income Statement_ How to Read and Use It.pdf',
    'cash_flow': PDF_PATH + 'Cash Flow Statements_ How to Prepare and Read One.pdf',
}

@dataclass
class RateLimiter:
    """Simple rate limiter to enforce a maximum number of API calls per minute.

    This is copied from ``analyst.quantitative_analyst.quantitative_analyst`` so
    that both analyst classes behave consistently.  See that module
    for additional documentation.

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
            sleep_duration = 60 - (now - self.timestamps[0])
            if sleep_duration > 0:
                self.logger.info(
                    "Rate limit exceeded: sleeping %.2f seconds to respect %d calls/minute",
                    sleep_duration,
                    self.max_calls_per_minute,
                )
                time.sleep(sleep_duration)
            # After sleeping, recompute now and trim again
            now = time.time()
            self.timestamps = [t for t in self.timestamps if now - t < 60]
        # Record the current call timestamp
        self.timestamps.append(time.time())


class FinancialAnalyst:
    """A class to assess whether a company appears under‑ or overpriced based on fundamentals.

    Parameters
    ----------
    symbol : Symbol
        Instance containing database tables loaded as attributes.  The
        quarterly statement tables (income, balance sheet, cash flow) are
        expected to be present.  The analyst reads from these tables
        but does not modify them.
    article_paths : dict[str, str]
        Mapping from metric names (e.g. ``"total_revenue"``) to PDF file
        paths relative to the project root.  The class reads these
        files and passes their content to the language model.  Each
        metric must have a corresponding entry in this mapping.
    openai_key : str, optional
        API key for OpenAI.  If provided, it will be set on the
        ``openai`` module.  If ``None``, the existing environment
        variable ``OPENAI_API_KEY`` will be used.
    rate_limit_per_minute : int, optional
        Maximum number of OpenAI API calls per minute.  Defaults to 60.
    num_quarters : int or None, optional
        Number of most recent quarters to include in each metric's
        analysis.  If ``None``, the entire available history is used.
    logger : logging.Logger, optional
        Logger for informational and error messages.  If omitted, a
        module‑level logger is used.
    """

    def __init__(
        self,
        symbol: Symbol = None,
        openai_key: Optional[str] = None,
        rate_limit_per_minute: int = 60,
        num_quarters: Optional[int] = 8,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.symbol = symbol
        self.article_paths = ARTICLE_PATHS
        self.num_quarters = num_quarters
        self.logger = logger or logging.getLogger(__name__)
        self.rate_limiter = RateLimiter(max_calls_per_minute=rate_limit_per_minute, logger=self.logger)
        # Configure OpenAI API key if provided
        if openai_key:
            openai.api_key = openai_key
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
        for metric, rel_path in self.article_paths.items():
            text = self._read_pdf(rel_path)
            if not text:
                self.logger.warning(
                    "No text extracted from %s for metric %s", rel_path, metric
                )
            self._article_texts[metric] = text

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
        for metric, text in self._article_texts.items():
            if not text:
                self._article_embeddings[metric] = []
                continue
            self.rate_limiter.record_call()
            try:
                response = openai.Embedding.create(input=[text], model=model)
                embedding = response["data"][0]["embedding"]
                self._article_embeddings[metric] = embedding
            except Exception as exc:
                self.logger.error("Embedding error for %s: %s", metric, exc)
                self._article_embeddings[metric] = []

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _find_quarterly_tables(self) -> List[pd.DataFrame]:
        """Return a list of DataFrames corresponding to quarterly statements.

        The ``Symbol`` instance exposes each database table as an
        attribute whose name matches the underlying table name.  In
        different parts of the codebase the word *quarterly* is
        sometimes misspelled as ``quaterly``.  To be resilient to both
        conventions, this method looks for any attribute on ``self.symbol``
        that contains ``income_statement``, ``balance_sheet`` or ``cash_flow``
        and ends with ``quarterly`` or ``quaterly``.  Tables that are
        present and non‑empty are returned.
        """
        tables: List[pd.DataFrame] = []
        # Possible stems for the three statement types
        stems = ["income_statement", "balance_sheet", "cash_flow"]
        for attr_name in dir(self.symbol):
            # Skip private attributes
            if attr_name.startswith("_"):
                continue
            lower_name = attr_name.lower()
            # Only consider attributes that include one of the stems and end with the quarterly suffix
            if any(stem in lower_name for stem in stems) and (
                lower_name.endswith("quarterly") or lower_name.endswith("quaterly")
            ):
                df = getattr(self.symbol, attr_name, None)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    tables.append(df)
                else:
                    self.logger.debug("Quarterly table '%s' not found or empty", attr_name)
        return tables

    def _get_quarter_series(self, metric: str) -> Optional[pd.Series]:
        """Retrieve a time series of values for a given metric from quarterly tables.

        This method searches across the available quarterly statement
        DataFrames for a column matching the provided metric name.  If
        found, it extracts the series of values and aligns it with a
        date or fiscal period column.  The return value is sorted by
        date ascending, and optionally truncated to the last
        ``num_quarters`` entries.  If the metric cannot be found in
        any quarterly table, ``None`` is returned.
        """
        for table in self._find_quarterly_tables():
            if metric not in table.columns:
                continue
            df = table.copy()
            # Determine date column name heuristically
            date_col_candidates = [
                col
                for col in df.columns
                if any(
                    key in col.lower()
                    for key in ["date", "fiscal", "quarter", "period"]
                )
            ]
            date_col: Optional[str] = None
            if date_col_candidates:
                # Choose the first candidate
                date_col_candidates.sort()
                date_col = date_col_candidates[0]
            # Create a date index
            if date_col and date_col in df.columns:
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                except Exception:
                    # Fall back to using the index as date if parsing fails
                    df.reset_index(inplace=True)
                    date_col = "index"
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            else:
                # Use the index as a date surrogate
                df.reset_index(inplace=True)
                date_col = "index"
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            # Sort by date ascending and drop rows with NaT
            df = df.dropna(subset=[date_col])
            df = df.sort_values(date_col)
            series = df[metric].copy()
            series.index = df[date_col]
            # If num_quarters is specified, take the last num_quarters observations
            if self.num_quarters is not None and self.num_quarters > 0:
                series = series.tail(self.num_quarters)
            return series
        # Metric not found in any table
        self.logger.warning("Metric '%s' not found in quarterly tables", metric)
        return None

    # ------------------------------------------------------------------
    # Metric evaluation
    # ------------------------------------------------------------------
    def _call_language_model(
        self, 
        metric: str, 
        values: List[float], 
        article: str
    ) -> Tuple[Optional[float], str]:
        """Invoke the OpenAI chat model to score a metric.

        The prompt instructs the model to produce a continuous value in
        ``[-1, 1]`` and to explain its reasoning.  The first numeric
        value found in the response is parsed as the score.  If the
        call fails or no number is present, ``None`` is returned for
        the score.
        """
        # Compose the user prompt
        if self.num_quarters is None:
            period_desc = "historical"
        else:
            period_desc = f"the last {self.num_quarters} quarters"
        prompt = (
            f"You are a financial analyst assessing the {metric} values "
            f"for a company over {period_desc}.\n"
            f"Here are the series of {metric} values (most recent last):\n"
            f"{values}\n\n"
            f"Refer to the following article describing how to interpret this metric:\n"
            f"{article}\n\n"
            "Based on the values and the article, output a single continuous "
            "number between -1 and 1 representing how underpriced (-1) or "
            "overpriced (+1) the company appears according to this metric. "
            "After the number, provide a brief explanation of your reasoning."
        )
        self.rate_limiter.record_call()
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-nano",
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            reply = response["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            self.logger.error("Error calling language model for %s: %s", metric, exc)
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

    def evaluate_metric(self, metric: str) -> Tuple[Optional[float], str]:
        """Evaluate a single metric and return its score and explanation."""
        series = self._get_quarter_series(metric)
        if series is None or series.empty:
            return None, "No data available"
        values_list = series.tolist()
        article = self._article_texts.get(metric, "")
        return self._call_language_model(metric, values_list, article)

    def run_analysis(self) -> Tuple[Dict[str, Optional[float]], Optional[float], Dict[str, str]]:
        """Run the analysis for all configured metrics.

        Returns
        -------
        metric_scores : dict[str, float or None]
            Per‑metric continuous score in ``[-1, 1]``.  If a metric
            could not be scored due to missing data or API failure, its
            value will be ``None``.
        aggregate_score : float or None
            The average of all numeric metric scores.  ``None`` if no
            metric produced a numeric score.
        explanations : dict[str, str]
            Full language‑model responses (reasoning) keyed by metric.
        """
        # Ensure articles are loaded; embeddings are optional but may
        # improve prompt quality in future iterations
        if not self._article_texts:
            self.load_articles()
        scores: Dict[str, Optional[float]] = {}
        explanations: Dict[str, str] = {}
        for metric in self.article_paths.keys():
            score, explanation = self.evaluate_metric(metric)
            scores[metric] = score
            explanations[metric] = explanation
        # Compute aggregate as mean of valid numeric scores
        numeric_scores = [s for s in scores.values() if isinstance(s, (int, float))]
        aggregate: Optional[float] = None
        if numeric_scores:
            aggregate = sum(numeric_scores) / len(numeric_scores)
            # Again clip to [-1, 1]
            aggregate = max(min(aggregate, 1.0), -1.0)
        return scores, aggregate, explanations