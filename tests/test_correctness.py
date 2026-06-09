"""Numerical correctness tests.

Unlike the smoke tests (which check shapes and columns), these verify that the
core algorithms compute the *right numbers* on small, hand-checkable inputs.
"""

import numpy as np
import pandas as pd
import pytest

from src.bars_creator import (
    tick_bars_creator,
    dollar_bars_creator,
    apply_imbalance_bars,
    get_imbalance_bars_numba,
)
from src.etf_trick import ETFTrick
from src.instrument_config import FUTURE_SPECS
from src.allocation import pcaweights


def _ticks(prices, sizes=None):
    n = len(prices)
    if sizes is None:
        sizes = [1] * n
    return pd.DataFrame({
        "ts_event": pd.date_range("2023-03-01", periods=n, freq="1s"),
        "price": np.asarray(prices, dtype=float),
        "size": np.asarray(sizes, dtype=int),
    })


# --------------------------------------------------------------------------- #
# Bars: exact OHLCV values
# --------------------------------------------------------------------------- #

class TestBarValues:
    PRICES = [10, 11, 9, 12, 8, 7, 15, 10]

    def test_tick_bar_exact_ohlcv(self):
        bars = tick_bars_creator(_ticks(self.PRICES), threshold=4)
        assert len(bars) == 2
        assert bars["open"].tolist() == [10, 8]
        assert bars["high"].tolist() == [12, 15]
        assert bars["low"].tolist() == [9, 7]
        assert bars["close"].tolist() == [12, 10]
        assert bars["volume"].tolist() == [4, 4]
        # equal sizes -> vwap is the simple mean of the four prices
        assert bars["vwap"].tolist() == [10.5, 10.0]

    def test_apply_imbalance_bars_exact(self):
        df = _ticks(self.PRICES)
        bars = apply_imbalance_bars(df, np.array([3, 7], dtype=np.int32))
        assert bars["open"].tolist() == [10, 8]
        assert bars["close"].tolist() == [12, 10]
        assert bars["high"].tolist() == [12, 15]
        assert bars["low"].tolist() == [9, 7]
        assert bars["tick_count"].tolist() == [4, 4]

    def test_dollar_bar_vwap_within_range(self):
        rng = np.random.default_rng(0)
        prices = 4000 + rng.standard_normal(20_000).cumsum() * 0.25
        df = _ticks(prices, rng.integers(1, 10, size=20_000))
        bars = dollar_bars_creator(df, threshold=5e6)
        assert len(bars) > 1
        # VWAP is a convex combination of in-bar prices -> must sit in [low, high]
        assert (bars["vwap"] <= bars["high"] + 1e-9).all()
        assert (bars["vwap"] >= bars["low"] - 1e-9).all()


# --------------------------------------------------------------------------- #
# Imbalance engine: rolls force a bar boundary
# --------------------------------------------------------------------------- #

class TestImbalanceEngine:
    def test_roll_forces_bar_boundary(self):
        signs = np.ones(100, dtype=np.float64)
        roll_indices = np.array([50], dtype=np.int32)
        idx = get_imbalance_bars_numba(signs, roll_indices=roll_indices,
                                       initial_T=1000.0, initial_E_b=0.0)
        # the roll at tick 50 must close a bar exactly at 49 (the tick before)
        assert 49 in idx.tolist()
        assert (np.diff(idx) > 0).all()  # strictly increasing

    def test_no_roll_no_forced_boundary(self):
        signs = np.ones(100, dtype=np.float64)
        idx = get_imbalance_bars_numba(signs, roll_indices=np.array([], dtype=np.int32),
                                       initial_T=1000.0, initial_E_b=0.0)
        assert 49 not in idx.tolist()


# --------------------------------------------------------------------------- #
# ETF Trick: economic invariants
# --------------------------------------------------------------------------- #

def _etf_frame(close_by_asset: dict, n: int) -> pd.DataFrame:
    idx = pd.date_range("2023-03-01", periods=n, freq="1h")
    data = {}
    for asset, close in close_by_asset.items():
        data[f"close_{asset}"] = close
        data[f"open_{asset}"] = close  # open == close: no roll stitching needed
    return pd.DataFrame(data, index=idx)


class TestETFTrick:
    def test_flat_prices_keep_K_constant(self):
        n = 50
        data = _etf_frame({"ES": np.full(n, 4000.0), "NQ": np.full(n, 13000.0)}, n)
        empty = np.array([], dtype="datetime64[ns]")
        etf = ETFTrick(FUTURE_SPECS).run(data, {"ES": empty, "NQ": empty},
                                         weights=np.array([0.5, 0.5]))
        K, _ = etf.get_results()
        # no price moves -> no P&L -> the synthetic value must stay at its start
        assert np.allclose(K.values, 1.0)

    def test_single_asset_tracks_relative_price(self):
        n = 30
        rng = np.random.default_rng(1)
        close = 4000.0 + rng.standard_normal(n).cumsum()
        data = _etf_frame({"ES": close}, n)
        empty = np.array([], dtype="datetime64[ns]")
        etf = ETFTrick(FUTURE_SPECS).run(data, {"ES": empty}, weights=np.array([1.0]))
        K, _ = etf.get_results()
        # a single fully-invested asset: K must equal close / close[0]
        assert np.allclose(K.values, close / close[0])


# --------------------------------------------------------------------------- #
# PCA allocation: realized risk matches the target
# --------------------------------------------------------------------------- #

def _spd_cov(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    return m @ m.T + n * np.eye(n)  # symmetric positive definite


class TestPCAWeights:
    def test_portfolio_variance_equals_target_squared(self):
        cov = _spd_cov(5)
        risk_target = 2.0
        risk_distr = np.full(5, 1 / 5)
        omega = pcaweights(cov, riskDistr=risk_distr, riskTarget=risk_target)
        assert np.isclose(omega @ cov @ omega, risk_target ** 2)

    def test_default_allocation_unit_variance(self):
        cov = _spd_cov(4, seed=3)
        omega = pcaweights(cov)  # default: all risk on smallest-variance PC, target 1
        assert np.isclose(omega @ cov @ omega, 1.0)
