"""Shared pytest fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_clean_ticks():
    """Factory for synthetic *cleaned* tick data (trades only, no metadata).

    Columns ['ts_event', 'price', 'size']: the input shape expected by the
    bar creators.
    """

    def _make(n: int = 5000, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        prices = 4000.0 + rng.standard_normal(n).cumsum() * 0.25
        return pd.DataFrame(
            {
                "ts_event": pd.date_range("2023-03-01", periods=n, freq="100ms"),
                "price": prices,
                "size": rng.integers(1, 50, size=n),
            }
        )

    return _make


@pytest.fixture
def make_raw_ticks():
    """Factory for synthetic *raw* tick data (with 'instrument_id' and 'action').

    This is the shape expected by ``data_cleaner_for_bars`` before cleaning.
    """

    def _make(n: int = 100, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        prices = 4000.0 + rng.standard_normal(n).cumsum() * 0.25
        return pd.DataFrame(
            {
                "instrument_id": ["ESH3"] * n,
                "action": ["T"] * n,
                "ts_event": pd.date_range("2023-03-01", periods=n, freq="100ms"),
                "price": prices,
                "size": rng.integers(1, 20, size=n),
            }
        )

    return _make
