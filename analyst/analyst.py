"""
analyst.py
----------

Base class for all LLM-backed analyst modules.  Provides shared
infrastructure so that concrete subclasses (``QuantitativeAnalyst``,
``FinancialAnalyst``, and future ``NewsAnalyst``) only need to
implement domain-specific data preparation, prompt construction,
and DB-storage logic.

Shared functionality provided by ``Analyst``:

* OpenAI client initialisation and API-key resolution.
* ``RateLimiter`` dataclass (one instance per analyst).
* PDF article loading (``_read_pdf``, ``load_articles``).
* Score parsing from LLM replies (``_parse_score``).
* Low-level LLM chat call (``_call_llm``).
* Aggregate score computation (``_compute_aggregate``).
* Symbol-string resolution (``_resolve_symbol_str``).
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import openai
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from symbol.symbol import Symbol

load_dotenv()


# ---------------------------------------------------------------------------
# Rate limiter (shared across all analysts)
# ---------------------------------------------------------------------------
@dataclass
class RateLimiter:
    """Simple rate limiter to enforce a maximum number of API calls per minute.

    Tracks call timestamps and sleeps when the configured limit would
    be exceeded in a rolling 60-second window.

    Attributes
    ----------
    max_calls_per_minute : int
        Maximum allowed calls in a 60 s window.
    logger : logging.Logger
        Logger for informational messages about throttling.
    timestamps : list[float]
        Internal list of call timestamps (unix epoch seconds).
    """

    max_calls_per_minute: int = 60
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    timestamps: List[float] = field(default_factory=list)

    def record_call(self) -> None:
        """Record a call timestamp and sleep if the rate limit is exceeded."""
        now = time.time()
        self.timestamps = [t for t in self.timestamps if now - t < 60]
        if len(self.timestamps) >= self.max_calls_per_minute:
            sleep_duration = 60 - (now - self.timestamps[0])
            if sleep_duration > 0:
                self.logger.info(
                    "Rate limit exceeded: sleeping %.2f s to respect %d calls/min",
                    sleep_duration,
                    self.max_calls_per_minute,
                )
                time.sleep(sleep_duration)
            now = time.time()
            self.timestamps = [t for t in self.timestamps if now - t < 60]
        self.timestamps.append(time.time())


# ---------------------------------------------------------------------------
# Abstract base analyst
# ---------------------------------------------------------------------------
class Analyst(ABC):
    """Abstract base class for LLM-backed analysts.

    Parameters
    ----------
    symbol : Symbol
        A populated ``Symbol`` instance holding price/fundamental data.
    article_paths : dict[str, str]
        Mapping from logical article key to PDF file path.
    openai_key : str | None
        Explicit API key; falls back to *env_key_name* env var.
    env_key_name : str
        Name of the environment variable for the OpenAI key.
    model : str
        OpenAI model name to use for chat completions.
    system_prompt : str
        System-role message sent with every LLM call.
    rate_limit_per_minute : int
        Maximum OpenAI calls per 60 s.
    logger : logging.Logger | None
        Logger instance; defaults to module-level logger.
    store_in_db : bool
        Whether to persist results to the analyst database.
    """

    def __init__(
        self,
        symbol: Symbol = None,
        article_paths: Optional[Dict[str, str]] = None,
        openai_key: Optional[str] = None,
        env_key_name: str = "KEY_FOR_QUANT_ANALYST",
        model: str = "gpt-5",
        system_prompt: str = "You are a senior financial analyst.",
        rate_limit_per_minute: int = 60,
        logger: Optional[logging.Logger] = None,
        store_in_db: bool = False,
    ) -> None:
        self.symbol = symbol
        self.article_paths: Dict[str, str] = article_paths or {}
        self.model = model
        self.system_prompt = system_prompt
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.store_in_db = store_in_db
        self.rate_limiter = RateLimiter(
            max_calls_per_minute=rate_limit_per_minute, logger=self.logger
        )

        # Resolve API key
        api_key = openai_key or os.getenv(env_key_name) or os.getenv("KEY_FOR_QUANT_ANALYST")
        self._client = openai.OpenAI(api_key=api_key)

        # Article text cache
        self._article_texts: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Article loading
    # ------------------------------------------------------------------
    def _read_pdf(self, path: str) -> str:
        """Read a PDF file and return its concatenated text.

        If the file cannot be read, an error is logged and an empty
        string is returned.
        """
        abs_path = os.path.abspath(path)
        try:
            reader = PdfReader(abs_path)
            return "".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            self.logger.error("Failed to read PDF %s: %s", abs_path, exc)
            return ""

    def load_articles(self) -> None:
        """Load all article PDFs referenced by ``article_paths`` into
        ``_article_texts``.  Existing contents are overwritten.
        """
        self._article_texts.clear()
        for key, rel_path in self.article_paths.items():
            text = self._read_pdf(rel_path)
            if not text:
                self.logger.warning("No text from %s for key '%s'", rel_path, key)
            self._article_texts[key] = text

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _call_llm(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = 0.3,
        max_completion_tokens: int = 400,
    ) -> str:
        """Send a chat completion request and return the assistant reply.

        Uses ``self.rate_limiter`` for throttling.  Returns an empty
        string on failure.

        Note: GPT-5 only supports temperature=1 (default).  When ``temperature``
        is set to a non-default value and the model rejects it, the call
        is retried without the temperature parameter.
        """
        self.rate_limiter.record_call()
        try:
            kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt or self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max_completion_tokens,
            )
            # GPT-5 only accepts temperature=1 (the default); skip the
            # parameter entirely for that model family.
            model_lower = (self.model or "").lower()
            if temperature is not None and not model_lower.startswith("gpt-5"):
                kwargs["temperature"] = temperature
            response = self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as exc:
            self.logger.error("LLM call failed: %s", exc)
            return ""

    @staticmethod
    def _parse_score(reply: str) -> Optional[float]:
        """Extract the first numeric value from an LLM reply and clip to [-1, 1]."""
        match = re.search(r"-?\d*\.\d+|-?\d+", reply)
        if match:
            try:
                score = float(match.group())
                return max(min(score, 1.0), -1.0)
            except ValueError:
                return None
        return None

    @staticmethod
    def _compute_aggregate(scores: Dict[str, Optional[float]]) -> Optional[float]:
        """Compute the mean of all valid numeric scores, clipped to [-1, 1]."""
        numeric = [s for s in scores.values() if isinstance(s, (int, float))]
        if not numeric:
            return None
        agg = sum(numeric) / len(numeric)
        return max(min(agg, 1.0), -1.0)

    def _resolve_symbol_str(self) -> Optional[str]:
        """Extract a symbol ticker string from the ``Symbol`` instance."""
        for attr in ("_symbol", "symbol", "ticker"):
            val = getattr(self.symbol, attr, None)
            if val is not None:
                return val
        return None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def run_analysis(self, **kwargs) -> dict:
        """Run the full analysis and return a result dictionary."""
        ...

    @abstractmethod
    def _store_results(self, result: dict) -> None:
        """Persist analysis results to the analyst database."""
        ...

    def __repr__(self) -> str:
        symbol_str = self._resolve_symbol_str() or "?"
        return f"{self.__class__.__name__}(symbol={symbol_str})"

