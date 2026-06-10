"""
Financial Data Processing Module
================================

This module handles the ingestion, cleaning, and preprocessing of high-frequency
financial data (tick data) stored in Parquet format. It serves as the primary
interface between raw storage and the quantitative analysis pipeline.

The pipeline is designed to be modular:
1. **Loading:** Efficiently loads data using PyArrow and attaches instrument metadata.
2. **Cleaning:** Filters for valid trades and detects contract rolls based on ID changes.
3. **Enrichment:** Infers trade direction using the Tick Rule algorithm.

Key Features:
-------------
- **Metadata-Aware Loading:** Automatically attaches 'point_value' and symbol details
  from `src.instrument_config`.
- **Roll Index Detection:** Identifies the exact integer row indices where contract
  rollovers occur. This is crucial for Bar Generators to reset accumulation logic
  precisely at contract boundaries.
- **Microstructure Tools:** Includes a vectorized implementation of the Tick Rule
  to infer aggressor side.

Dependencies:
-------------
- pandas
- numpy
- src.instrument_config

Usage Example:
--------------
>>> from src.data_processing import data_loader, data_cleaner_for_bars, tick_rule_creator
>>>
>>> # 1. Load raw data (metadata is attached to df_raw.attrs)
>>> df_raw = data_loader(data_dir="../data", symbol="ES", period_keyword="Mar2020")
>>>
>>> # 2. Clean data for bar generation and get roll indices
>>> roll_indices, df_trades = data_cleaner_for_bars(df_raw)
>>>
>>> # 3. (Optional) Apply Tick Rule to infer trade direction
>>> trade_signs = tick_rule_creator(df_trades['price'])
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.instrument_config import BAR_SAMPLING_CONFIG, FUTURE_SPECS

logger = logging.getLogger(__name__)


def data_cleaner_for_bars(ticks_dataset: pd.DataFrame) -> tuple[NDArray, pd.DataFrame]:
    """
    Prepares raw tick data for bar generation by filtering for executed trades and
    identifying roll timestamps.

    This function isolates actual trade events (action == 'T') from the raw dataset,
    retains only the columns necessary for constructing financial bars, and identifies
    the timestamps where the underlying contract changes (rollover).

    Args:
        ticks_dataset (pd.DataFrame): The raw tick data containing at least
            'action', 'instrument_id', 'ts_event', 'price', and 'size' columns.

    Returns:
        tuple[NDArray, pd.DataFrame]: A tuple containing:
            - **roll_dates** (NDArray): An array of timestamps (from 'ts_event')
              indicating when the contract ID changed.
            - **df_clean** (pd.DataFrame): A cleaned DataFrame containing only trade
              events with columns ['ts_event', 'price', 'size'], and a reset index.
    """

    # Filtering our dataset
    df_clean = ticks_dataset.loc[
        ticks_dataset["action"] == "T",
        ["instrument_id", "ts_event", "price", "size"],
    ].copy()

    # Deleting the indexes
    df_clean.reset_index(drop=True, inplace=True)

    # Find the rolling index
    id_changes = df_clean["instrument_id"] != df_clean["instrument_id"].shift(1)
    id_changes.iloc[0] = False  # Would always be true since the shift has an NA in the first index
    roll_date = df_clean.loc[id_changes, "ts_event"].values

    return roll_date, df_clean[["ts_event", "price", "size"]]


def data_loader(data_dir: str, symbol: str, period_keyword: str) -> pd.DataFrame:
    """
    Loads a specific financial dataset from a Parquet file and attaches instrument metadata.

    This function searches for a Parquet file matching the `{symbol}*{period_keyword}*` pattern
    within the specified directory. It efficiently loads the data using PyArrow and attaches
    relevant financial specs (point value, name) to the DataFrame attributes.

    Args:
        data_dir (str): The relative or absolute path to the directory containing the datasets.
        symbol (str): The short asset symbol (e.g., 'ES', 'NQ', 'CL') used for file matching.
        period_keyword (str): A specific keyword identifying the period to load (e.g., 'Mar2020').

    Returns:
        pd.DataFrame: The raw data with columns ['instrument_id', 'action', 'ts_event', 'price', 'size'].
        Metadata (point_value, symbol, name) is attached to `df.attrs`.

    Raises:
        FileNotFoundError: If no file matching the pattern is found in `data_dir`.
    """

    data_path = Path(data_dir)
    full_symbol_key = f"{symbol}.c.0"

    # Find the right file in our datasets
    found_files = list(data_path.glob(f"{symbol}*{period_keyword}*.parquet"))

    # If there are no files corresponding to what we are looking for, return error
    if not found_files:
        raise FileNotFoundError(
            f"No Parquet file matching '{symbol}*{period_keyword}*.parquet' in '{data_path}'"
        )

    # Use the good file (should be only one and load it)
    file_path = found_files[0]
    df = pd.read_parquet(
        path=file_path, columns=["instrument_id", "action", "ts_event", "price", "size"], engine="pyarrow"
    )

    # Attach the instrument specs to the loaded dataset
    if full_symbol_key in FUTURE_SPECS:
        df.attrs["point_value"] = FUTURE_SPECS[full_symbol_key]["point_value"]
        df.attrs["symbol"] = symbol
        df.attrs["name"] = FUTURE_SPECS[full_symbol_key]["name"]

    return df


def tick_rule_creator(price_series: pd.Series) -> NDArray[np.floating]:
    """
    Applies the Tick Rule algorithm to infer trade direction (aggressor side).

    The Tick Rule is a standard heuristic used in market microstructure to classify
    trades as buy-initiated or sell-initiated when the explicit aggressor side
    is unknown:

    1. **Uptick (+1):** Price is higher than the previous price (Buy).
    2. **Downtick (-1):** Price is lower than the previous price (Sell).
    3. **Zero-tick (0):** Price is unchanged. The sign of the previous non-zero
       price change is used (continuation).

    Args:
        price_series (pd.Series): A time series of prices (e.g., trade prices or close prices).

    Returns:
        NDArray[np.floating]: A numpy array of the same length containing:
            -  1.0 for Buy-initiated trades.
            - -1.0 for Sell-initiated trades.
            -  NaN for the first element(s) where direction cannot be determined.
    """

    # Pure NumPy: avoids pandas .replace() intermediate copy
    # np.asarray with dtype forces a writable copy (PyArrow-backed arrays are read-only)
    prices = np.asarray(price_series, dtype=np.float64)

    # Empty input -> empty output (no direction can be inferred)
    if prices.size == 0:
        return prices

    diff = np.empty(len(prices), dtype=np.float64)
    diff[0] = np.nan
    diff[1:] = np.sign(np.diff(prices))

    # Forward-fill zeros (unchanged prices keep previous direction)
    diff[diff == 0.0] = np.nan
    # .to_numpy() returns a writable array (unlike .values with pandas CoW)
    # diff[0] is already NaN so ffill preserves it — no extra assignment needed
    return pd.Series(diff).ffill().to_numpy(dtype=np.float64)


# ==============================================================================
# MULTI-ASSET PROCESSING EXTENSION
# ==============================================================================


def process_one_asset(
    asset: str, datadir: str, month: str, config: dict
) -> tuple[str, NDArray, pd.DataFrame]:
    """
    Worker unit: Loads, cleans, and generates bars for a specific asset using
    multiprocessing-safe logic.
    """
    # Local import to avoid a circular dependency with bars_creator.
    import src.bars_creator as bc

    no_rolls = np.array([], dtype="datetime64[ns]")

    try:
        # 1. Identify specific config (Partial match on name)
        # E.g.: If asset="ES.c.0", we look for key "ES"
        key = next((k for k in config.keys() if k in asset), "DEFAULT")
        params = config.get(key)

        if not params:
            logger.warning("No configuration found for %s; skipping.", asset)
            return asset, no_rolls, pd.DataFrame()

        logger.info("[%s] Processing... (span=%s)", key, params.get("span", "N/A"))

        # 2. Load raw data (Direct call to internal function)
        temp_df = data_loader(datadir, asset, month)
        if temp_df.empty:
            raise ValueError(f"Raw file for {asset} is empty or invalid.")

        # 3. Clean and get Rolling Dates (Direct call to internal function)
        roll_dates, temp_df = data_cleaner_for_bars(temp_df)

        # 4. Create Imbalance Bars
        bars = bc.dollar_imbalance_bar_creator(
            temp_df,
            roll_dates=roll_dates,
            min_bar_len=10,
            warmup_minutes=params["warmup_min"],
            warmup_bars_count=params["warmup_bars"],
            span=params["span"],
            include_warmup=False,
        )

        # 5. Post-processing
        # Keep only 'open' and 'close' (the columns consumed by the ETF Trick).
        cols_to_keep = ["open", "close"]
        existing_cols = [c for c in cols_to_keep if c in bars.columns]
        bars = bars[existing_cols].copy()

        # Safety de-duplication
        if bars.index.duplicated().any():
            bars = bars[~bars.index.duplicated(keep="last")]

        # Add suffix (e.g., close_ES, open_ES)
        bars = bars.add_suffix(f"_{key}")

        logger.info("[%s] Done: %d bars generated.", key, len(bars))
        return key, roll_dates, bars

    except Exception:
        logger.exception("Critical failure on %s", asset)
        return asset, no_rolls, pd.DataFrame()


def multi_assets_synchro_multi(
    assets: list[str], datadir: str = "../data", month: str = "Mar", config: dict = BAR_SAMPLING_CONFIG
) -> tuple[dict, pd.DataFrame]:
    """
    Orchestrates parallel creation and synchronization (Outer Join) of datasets
    for the ETF Trick.

    Args:
        assets: List of file names/symbols to process.
        datadir: Path to raw data directory.
        month: Month keyword to load (e.g., 'Mar').
        config: Configuration dictionary (Span, Warmup...). Default is BAR_SAMPLING_CONFIG.

    Returns:
        rolling_dates: Dict of roll dates per asset.
        df_final: Merged and synchronized DataFrame (ffill).
    """
    if not assets:
        logger.warning("Empty asset list provided.")
        return {}, pd.DataFrame()

    # Parallelization (max 1 worker per CPU or per asset)
    max_workers = min(len(assets), os.cpu_count() or 1)
    rolling_dates = {}
    processed_frames = []

    logger.info("Starting synchronization for %d assets...", len(assets))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # Submit tasks
        futures = {ex.submit(process_one_asset, asset, datadir, month, config): asset for asset in assets}

        # Collect results
        for future in as_completed(futures):
            try:
                asset_key, r_dates, frame = future.result()
                if not frame.empty:
                    rolling_dates[asset_key] = r_dates
                    processed_frames.append(frame)
            except Exception:
                logger.exception("Worker crashed")

    if not processed_frames:
        logger.error("No valid DataFrames generated.")
        return {}, pd.DataFrame()

    logger.info("Merging data (outer join)...")

    # 1. Merge on time index (Union of timestamps)
    df_final = pd.concat(processed_frames, axis=1, join="outer", sort=True)

    # 2. Forward Fill (Propagate last known value)
    df_final.ffill(inplace=True)

    # 3. Drop data before common start (where at least one asset is NaN)
    df_final.dropna(inplace=True)

    logger.info("Final dataset ready: %d rows x %d columns", df_final.shape[0], df_final.shape[1])
    return rolling_dates, df_final
