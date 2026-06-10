"""
Statistical comparison of bar types (AFML Chapter 2)
====================================================

Empirical study of four sampling schemes (time, tick, volume and dollar bars)
on real CME E-mini S&P 500 (ES) tick data, following *Advances in Financial
Machine Learning* (Lopez de Prado, Chapter 2). It quantifies the properties that
matter for downstream machine learning:

    1. Sampling adaptivity            (bars per day vs. information flow)
    2. Sampling-frequency stability   (Ex. 2.1.a): weekly bar counts
    3. Serial correlation of returns  (Ex. 2.1.b): lag-1 autocorrelation
    4. Variance stationarity          (Ex. 2.1.c): variance of monthly variances
    5. Return normality               (Ex. 2.1.d): Jarque-Bera, skew, kurtosis

Methodology
-----------
The tick/volume/dollar thresholds are calibrated **once over the whole period**
(global, fixed) so that each scheme produces approximately the same *total*
number of bars as the time-bar baseline. This keeps the comparison fair (same
average frequency) while preserving the key feature of interest: how the
*instantaneous* sampling rate adapts to market activity.

Months are processed one at a time, keeping only the (small) resulting bars in
memory, so the full year fits comfortably even though the raw tick captures are
hundreds of millions of rows.

Usage
-----
    python -m analysis.bar_statistics                       # ES, default months
    python -m analysis.bar_statistics --symbol ES --freq 5min

Outputs (written to ``results/``)
---------------------------------
    - bar_counts_weekly.png     (sampling adaptivity / frequency stability)
    - return_distributions.png  (normality vs. standard normal)
    - bar_statistics.csv        (summary table, also printed to stdout)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.pyplot as plt

from src.bars_creator import (
    dollar_bars_creator,
    tick_bars_creator,
    volume_bars_creator,
)
from src.data_processing import data_cleaner_for_bars, data_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# 2022 ES months light enough to process tick-by-tick (the Jan/Oct captures are
# multi-hundred-million rows and excluded for runtime).
DEFAULT_MONTHS = [
    "Feb2022",
    "Mar2022",
    "Apr2022",
    "May2022",
    "Jun2022",
    "Jul2022",
    "Aug2022",
    "Sep2022",
    "Nov2022",
    "Dec2022",
]

BAR_TYPES = ["time", "tick", "volume", "dollar"]
COLORS = {"time": "#9e9e9e", "tick": "#1f77b4", "volume": "#2ca02c", "dollar": "#d62728"}


# --------------------------------------------------------------------------- #
# Bar construction
# --------------------------------------------------------------------------- #


def time_bars_creator(ticks: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Baseline OHLCV time bars (fixed clock interval), for comparison only."""
    s = ticks.set_index("ts_event")
    ohlc = s["price"].resample(freq).ohlc()
    volume = s["size"].resample(freq).sum()
    bars = ohlc.join(volume.rename("volume"))
    return bars.dropna(subset=["close"])


def _load_clean(symbol: str, month: str) -> pd.DataFrame:
    raw = data_loader(str(DATA_DIR), symbol, month)
    _, clean = data_cleaner_for_bars(raw)
    return clean


def scan_thresholds(symbol: str, months: list[str], freq: str) -> tuple[dict, list[str]]:
    """First pass: accumulate global totals and derive fixed thresholds so that
    every scheme yields ~the same total number of bars as the time baseline."""
    n_ticks = 0
    total_volume = 0.0
    total_dollar = 0.0
    n_time = 0
    available = []
    for month in months:
        try:
            clean = _load_clean(symbol, month)
        except FileNotFoundError:
            logger.warning("[%s] file not found, skipping.", month)
            continue
        available.append(month)
        n_ticks += len(clean)
        total_volume += float(clean["size"].sum())
        total_dollar += float((clean["price"] * clean["size"]).sum())
        counts = clean.set_index("ts_event")["size"].resample(freq).sum()
        n_time += int((counts > 0).sum())
        del clean

    n_time = max(n_time, 1)
    thresholds = {
        "tick": max(int(n_ticks // n_time), 1),
        "volume": max(int(total_volume // n_time), 1),
        "dollar": total_dollar / n_time,
    }
    logger.info(
        "Global calibration: %s ticks, target ~%d bars/scheme -> tick=%d, volume=%d, dollar=%.3e",
        f"{n_ticks:,}",
        n_time,
        thresholds["tick"],
        thresholds["volume"],
        thresholds["dollar"],
    )
    return thresholds, available


def build_all(symbol: str, months: list[str], freq: str, thresholds: dict) -> dict[str, pd.DataFrame]:
    """Second pass: build the four bar series with fixed global thresholds."""
    acc: dict[str, list[pd.DataFrame]] = {b: [] for b in BAR_TYPES}
    for month in months:
        clean = _load_clean(symbol, month)
        month_bars = {
            "time": time_bars_creator(clean, freq)[["close", "volume"]],
            "tick": tick_bars_creator(clean, threshold=thresholds["tick"])[["close", "volume"]],
            "volume": volume_bars_creator(clean, threshold=thresholds["volume"])[["close", "volume"]],
            "dollar": dollar_bars_creator(clean, threshold=thresholds["dollar"])[["close", "volume"]],
        }
        for b in BAR_TYPES:
            acc[b].append(month_bars[b])
        logger.info(
            "[%s] time=%d tick=%d volume=%d dollar=%d bars",
            month,
            len(month_bars["time"]),
            len(month_bars["tick"]),
            len(month_bars["volume"]),
            len(month_bars["dollar"]),
        )
        del clean

    out: dict[str, pd.DataFrame] = {}
    for b in BAR_TYPES:
        frames = [f for f in acc[b] if not f.empty]
        df = pd.concat(frames).sort_index()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        out[b] = df[~df.index.duplicated(keep="last")]
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def log_returns(bars: pd.DataFrame) -> pd.Series:
    """Log returns of the close price, computed *within* each calendar month.

    The monthly tick captures are concatenated for the count/time-series plots,
    but returns must never be differenced across a month boundary (that would
    inject a spurious gap return, e.g. the Oct-2022 hole in the sample). We
    therefore diff inside each month and drop the first observation of each.
    """
    log_close = np.log(bars["close"])
    return log_close.groupby(bars.index.to_period("M")).diff().dropna()


def jarque_bera(returns: pd.Series) -> float:
    """Jarque-Bera statistic (closed form). Lower => closer to Gaussian.

    JB = n/6 * (S^2 + (1/4) * K^2), with S the skewness and K the *excess*
    kurtosis. Asymptotically JB ~ chi2(2) under the normality null.
    """
    n = len(returns)
    s = returns.skew()
    k = returns.kurt()  # pandas returns excess kurtosis
    return n / 6.0 * (s**2 + 0.25 * k**2)


def summary_table(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-bar-type summary of the AFML statistical properties."""
    rows = []
    for b in BAR_TYPES:
        r = log_returns(bars[b])
        weekly = bars[b].resample("W").size()
        weekly = weekly[weekly > 0]
        monthly_var = r.resample("ME").var().dropna()
        rows.append(
            {
                "bar_type": b,
                "n_bars": len(bars[b]),
                "weekly_count_CoV": weekly.std() / weekly.mean(),
                "autocorr_lag1": r.autocorr(lag=1),
                "skew": r.skew(),
                "excess_kurtosis": r.kurt(),
                "jarque_bera": jarque_bera(r),
                "monthly_var_CoV": monthly_var.std() / monthly_var.mean(),
            }
        )
    return pd.DataFrame(rows).set_index("bar_type")


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_weekly_counts(bars: dict[str, pd.DataFrame], path: Path) -> None:
    """Adaptivity: information-driven bars expand in active markets and contract
    in quiet ones, whereas time bars are flat by construction (CoV in legend)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for b in BAR_TYPES:
        weekly = bars[b].resample("W").size()
        weekly = weekly[weekly > 0]
        cov = weekly.std() / weekly.mean()
        ax.plot(weekly.index, weekly.values, marker="o", ms=3, color=COLORS[b], label=f"{b} (CoV={cov:.2f})")
    ax.set_title("Weekly bar counts: information-driven bars track market activity, time bars ignore it")
    ax.set_ylabel("bars per week")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_return_distributions(bars: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    grid = np.linspace(-5, 5, 400)
    normal_pdf = np.exp(-(grid**2) / 2) / np.sqrt(2 * np.pi)
    for ax, b in zip(axes.ravel(), BAR_TYPES, strict=True):
        r = log_returns(bars[b])
        z = (r - r.mean()) / r.std()
        ax.hist(z, bins=200, density=True, color=COLORS[b], alpha=0.6, range=(-5, 5))
        ax.plot(grid, normal_pdf, "k--", lw=1.2, label="N(0,1)")
        ax.set_title(f"{b} bars  (JB={jarque_bera(r):,.0f}, excess kurt={r.kurt():.1f})")
        ax.set_xlim(-5, 5)
        ax.legend(fontsize=8)
    fig.suptitle("Standardized return distributions vs. standard normal", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="AFML Ch.2 bar statistics comparison.")
    parser.add_argument("--symbol", default="ES")
    parser.add_argument("--freq", default="5min", help="time-bar frequency (sampling anchor)")
    parser.add_argument("--months", nargs="*", default=DEFAULT_MONTHS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    logger.info("Pass 1/2: calibrating thresholds over %d months...", len(args.months))
    thresholds, available = scan_thresholds(args.symbol, args.months, args.freq)
    if not available:
        logger.error("No data files matched. Aborting.")
        return

    logger.info("Pass 2/2: building bars...")
    bars = build_all(args.symbol, available, args.freq, thresholds)

    table = summary_table(bars)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(f"\n=== AFML Chapter 2: bar statistics ({args.symbol}, {len(available)} months 2022) ===\n")
    print(table.to_string())
    table.to_csv(RESULTS_DIR / "bar_statistics.csv")

    plot_weekly_counts(bars, RESULTS_DIR / "bar_counts_weekly.png")
    plot_return_distributions(bars, RESULTS_DIR / "return_distributions.png")
    logger.info("Figures and table written to %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
