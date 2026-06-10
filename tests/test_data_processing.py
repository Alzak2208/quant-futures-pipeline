"""Smoke tests for data_processing module."""

import numpy as np
import pandas as pd

from src.data_processing import data_cleaner_for_bars, tick_rule_creator

# ---------------------------------------------------------------------------
# tick_rule_creator
# ---------------------------------------------------------------------------


class TestTickRule:
    def test_basic_signs(self):
        prices = pd.Series([100.0, 101.0, 99.0, 99.0, 100.0])
        result = tick_rule_creator(prices)
        assert np.isnan(result[0])
        assert result[1] == 1.0  # uptick
        assert result[2] == -1.0  # downtick
        assert result[3] == -1.0  # zero-tick -> previous sign
        assert result[4] == 1.0  # uptick

    def test_single_price(self):
        result = tick_rule_creator(pd.Series([50.0]))
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_empty_series(self):
        result = tick_rule_creator(pd.Series([], dtype=float))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# data_cleaner_for_bars
# ---------------------------------------------------------------------------


class TestDataCleaner:
    def test_filters_trades_only(self, make_raw_ticks):
        df = make_raw_ticks(50)
        # inject some non-trade rows
        df.loc[0, "action"] = "Q"
        df.loc[5, "action"] = "Q"
        roll_dates, clean = data_cleaner_for_bars(df)
        assert "Q" not in clean.index  # no quotes
        assert len(clean) == 48
        assert list(clean.columns) == ["ts_event", "price", "size"]

    def test_detects_roll(self, make_raw_ticks):
        df = make_raw_ticks(50)
        df.loc[25:, "instrument_id"] = "ESM3"  # contract roll at index 25
        roll_dates, clean = data_cleaner_for_bars(df)
        assert len(roll_dates) == 1

    def test_no_roll_single_contract(self, make_raw_ticks):
        df = make_raw_ticks(50)
        roll_dates, _ = data_cleaner_for_bars(df)
        assert len(roll_dates) == 0
