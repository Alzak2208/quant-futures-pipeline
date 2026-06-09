"""Smoke tests for etf_trick module."""

import numpy as np
import pandas as pd
import pytest

from src.etf_trick import ETFTrick
from src.instrument_config import FUTURE_SPECS


def _make_etf_data(n: int = 100) -> pd.DataFrame:
    """Create minimal synthetic ETF input data for 2 assets."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-03-01", periods=n, freq="1h")
    es_close = 4000.0 + rng.standard_normal(n).cumsum() * 2
    nq_close = 13000.0 + rng.standard_normal(n).cumsum() * 5
    return pd.DataFrame({
        "open_ES": es_close + rng.uniform(-1, 1, n),
        "close_ES": es_close,
        "open_NQ": nq_close + rng.uniform(-2, 2, n),
        "close_NQ": nq_close,
    }, index=idx)


class TestETFTrick:
    def test_run_fixed_weights(self):
        data = _make_etf_data(100)
        etf = ETFTrick(FUTURE_SPECS)
        etf.run(data, rolling_dict={"ES": np.array([], dtype="datetime64[ns]"),
                                     "NQ": np.array([], dtype="datetime64[ns]")},
                weights=np.array([0.5, 0.5]))
        K, h = etf.get_results()
        assert len(K) == 100
        assert h.shape == (100, 2)
        assert np.isfinite(K.values).all()

    def test_get_results_before_run_raises(self):
        etf = ETFTrick(FUTURE_SPECS)
        with pytest.raises(ValueError):
            etf.get_results()

    def test_missing_open_col_raises(self):
        data = pd.DataFrame({
            "close_ES": [4000.0, 4001.0],
        }, index=pd.date_range("2023-01-01", periods=2, freq="1h"))
        etf = ETFTrick(FUTURE_SPECS)
        with pytest.raises(ValueError):
            etf.run(data, rolling_dict={}, weights=np.array([1.0]))
