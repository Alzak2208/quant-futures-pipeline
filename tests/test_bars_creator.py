"""Smoke tests for bars_creator module."""

import numpy as np
import pandas as pd
import pytest

from src.bars_creator import (
    tick_bars_creator,
    volume_bars_creator,
    dollar_bars_creator,
    apply_imbalance_bars,
    roll_indices_from_dates,
)

OHLCV_COLS = ["open", "high", "low", "close", "vwap", "volume", "tick_count"]


# ---------------------------------------------------------------------------
# Standard Bars
# ---------------------------------------------------------------------------

class TestTickBars:
    def test_basic(self, make_clean_ticks):
        df = make_clean_ticks(5000)
        bars = tick_bars_creator(df, threshold=500)
        assert list(bars.columns) == OHLCV_COLS
        assert len(bars) == 10  # 5000 // 500

    def test_dataset_smaller_than_threshold(self, make_clean_ticks):
        df = make_clean_ticks(50)
        bars = tick_bars_creator(df, threshold=1000)
        assert bars.empty

    def test_high_low_sanity(self, make_clean_ticks):
        df = make_clean_ticks(5000)
        bars = tick_bars_creator(df, threshold=500)
        assert (bars["high"] >= bars["low"]).all()
        assert (bars["high"] >= bars["open"]).all()
        assert (bars["high"] >= bars["close"]).all()


class TestVolumeBars:
    def test_basic(self, make_clean_ticks):
        df = make_clean_ticks(5000)
        bars = volume_bars_creator(df, threshold=500)
        assert list(bars.columns) == OHLCV_COLS
        assert len(bars) > 0

    def test_volume_reasonable(self, make_clean_ticks):
        df = make_clean_ticks(5000)
        bars = volume_bars_creator(df, threshold=500)
        # each bar should have volume >= 80% of threshold (last bar pruned)
        assert (bars["volume"] >= 0.8 * 500).all()


class TestDollarBars:
    def test_basic(self, make_clean_ticks):
        df = make_clean_ticks(5000)
        bars = dollar_bars_creator(df, threshold=1e6)
        assert list(bars.columns) == OHLCV_COLS
        assert len(bars) > 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestRollIndicesFromDates:
    def test_basic(self):
        ts = pd.date_range("2023-01-01", periods=100, freq="1min").to_numpy()
        roll_dates = np.array(["2023-01-01 00:30", "2023-01-01 01:00"], dtype="datetime64[ns]")
        idx = roll_indices_from_dates(ts, roll_dates)
        assert len(idx) == 2
        assert idx.dtype == np.int32

    def test_empty_rolls(self):
        ts = pd.date_range("2023-01-01", periods=100, freq="1min").to_numpy()
        idx = roll_indices_from_dates(ts, np.array([], dtype="datetime64[ns]"))
        assert len(idx) == 0


class TestApplyImbalanceBars:
    def test_empty_indices(self, make_clean_ticks):
        df = make_clean_ticks(100)
        bars = apply_imbalance_bars(df, np.array([], dtype=np.int32))
        assert bars.empty

    def test_single_bar(self, make_clean_ticks):
        df = make_clean_ticks(100)
        bars = apply_imbalance_bars(df, np.array([99], dtype=np.int32))
        assert len(bars) == 1
        assert list(bars.columns) == OHLCV_COLS
