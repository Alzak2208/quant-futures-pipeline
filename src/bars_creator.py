"""
Financial Bars Creation Module
==============================

This module implements advanced sampling techniques to transform high-frequency 
tick data into OHLCV (Open, High, Low, Close, Volume) bars. 

Unlike standard Time Bars (sampling every minute/hour), the methods implemented here 
focus on **"Information-Driven"** sampling. They sample the market based on the 
arrival of information (activity, liquidity, or economic value), allowing for 
better statistical properties (normality, serial correlation) in Machine Learning models.

Theoretical Basis:
    Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado.

Key Features:
    1. **Standard Bars:**
       - `tick_bars_creator`: Sample every N transactions.
       - `volume_bars_creator`: Sample every N contracts exchanged.
       - `dollar_bars_creator`: Sample every $N exchanged (Recommended).

    2. **Imbalance Bars (Net Flow):**
       - Detects when the divergence between buy and sell flows exceeds expectations.
       - Available variants: `tick_imbalance`, `volume_imbalance`, `dollar_imbalance`.
       - Implementation: Recursive EWMA optimization via **Numba** (High Performance).

    3. **Runs Bars (Unilateral Intensity):**
       - Detects when a sequence of buys OR sells exceeds expectations.
       - Useful for detecting informed trading and "sweeping" of the order book.
       - Available variants: `tick_runs`, `volume_runs`, `dollar_runs`.

Dependencies:
    - pandas
    - numpy
    - numba (for JIT compilation of path-dependent recursive formulas)
    - src.data_processing (for tick_rule_creator)

Usage Example:
    >>> from src.bars_creator import dollar_bars_creator, dollar_imbalance_bar_creator
    >>> # 1. Standard Dollar Bars (e.g., every $10M)
    >>> bars_std = dollar_bars_creator(tick_data, threshold=10_000_000)
    >>> # 2. Advanced Imbalance Bars (Auto-adaptive)
    >>> bars_imb = dollar_imbalance_bar_creator(tick_data, num_init_ticks=1000)
"""

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from numba import njit

from src.data_processing import tick_rule_creator


def tick_bars_creator(ticks_dataset: pd.DataFrame, 
                      threshold: int = 1000) -> pd.DataFrame:
    """
    Creates Tick Bars from raw tick data using a fast vectorized approach.

    Tick bars are sampled every N transactions (ticks), regardless of the time elapsed.
    This synchronizes sampling with the arrival of information rather than the wall clock,
    which is particularly useful for analyzing periods of high volatility.

    The implementation uses NumPy reshaping to avoid slow loops, making it extremely 
    efficient for large datasets.

    Args:
        ticks_dataset (pd.DataFrame): The cleaned input DataFrame containing executed trades.
            Must contain columns: ['price', 'size', 'ts_event'].
            The DataFrame should be sorted by time and contain no index gaps relevant to the batch.
        threshold (int, optional): The number of ticks to aggregate into a single bar.
            Defaults to 1000.

    Returns:
        pd.DataFrame: A DataFrame containing OHLCV bars indexed by the timestamp 
        of the *last* tick in each bar.
        Columns: ['open', 'high', 'low', 'close', 'vwap', 'volume', 'tick_count'].
        
        If the dataset is smaller than the threshold, returns an empty DataFrame 
        with the correct column structure.
    """
    
    n_bars = len(ticks_dataset) // threshold
    
    # Case not enough datas
    if n_bars == 0:
        bars = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'vwap', 'volume', 'tick_count'])
        bars.index.name = "ts_event"
        return bars
    
    # Computing the number of ticks to keep
    nb_ticks_kept = n_bars * threshold
    
    # Creating the array versions of columns for computations
    arr_price = ticks_dataset["price"].values[:nb_ticks_kept]
    arr_size = ticks_dataset["size"].values[:nb_ticks_kept]
    arr_ts_event = ticks_dataset["ts_event"].values[:nb_ticks_kept]
    
    # Reshaping our arrays to only take into account the right prices
    price_matrix = arr_price.reshape(n_bars, threshold)
    size_matrix = arr_size.reshape(n_bars, threshold)
    ts_event_matrix = arr_ts_event.reshape(n_bars, threshold)
    
    # Compute volume and dollar values of ticks
    vol_sum = size_matrix.sum(axis=1)
    dollar = np.einsum("ij,ij->i", price_matrix, size_matrix)
    
    # Computing the tick bars
    bars = pd.DataFrame({
        'open': price_matrix[:, 0],
        'high': price_matrix.max(axis=1),
        'low': price_matrix.min(axis=1),
        'close': price_matrix[:, -1],
        'vwap': dollar / vol_sum,
        'volume': vol_sum,
        'tick_count': threshold
    }, index=ts_event_matrix[:, -1])
    bars.index.name = 'ts_event'
    
    return bars


def volume_bars_creator(ticks_dataset: pd.DataFrame, 
                        threshold: int = 6000) -> pd.DataFrame:
    """
    Creates Volume Bars from raw tick data by aggregating trades until a volume threshold is reached.

    Unlike Tick Bars (based on transaction count) or Time Bars (based on clock time), 
    Volume Bars are sampled every time a cumulative volume of `threshold` shares/contracts 
    is exchanged. This creates a sampling distribution that mimics the liquidity structure 
    of the market.

    Implementation details:
        - Uses vectorized cumulative sum and integer division to assign bar IDs.
        - Aggregates using pandas `groupby` (necessary as the number of ticks per bar varies).
        - Discards the final bar if it is less than 80% complete to avoid statistical noise.

    Args:
        ticks_dataset (pd.DataFrame): The cleaned input DataFrame containing executed trades.
            Must contain columns: ['price', 'size', 'ts_event'].
        threshold (int, optional): The aggregate volume required to form a single bar.
            Defaults to 6000.

    Returns:
        pd.DataFrame: A DataFrame containing OHLCV bars indexed by the timestamp 
        of the *last* tick in each bar.
        Columns: ['open', 'high', 'low', 'close', 'vwap', 'volume', 'tick_count'].
    """
    
    data = ticks_dataset[["ts_event", "price", "size"]].copy()

    # Vectorized Grouping
    data["cum_volume"] = data["size"].cumsum()
    data["bar_id"] = (data["cum_volume"] // threshold).astype(np.uint32)
    data["dollar_value"] = data['price'] * data['size']

    # Single groupby for all aggregations
    bars = data.groupby('bar_id').agg({
        'ts_event': 'last',
        'price': ['first', 'max', 'min', 'last'],
        'size': ['sum', 'count'],
        'dollar_value': 'sum'
    })

    # Naming bars columns
    bars.columns = ['ts_event', 'open', 'high', 'low', 'close', 'volume', 'tick_count', 'dollar_sum']

    # Computing vwap
    bars["vwap"] = bars['dollar_sum'] / bars['volume']

    # Setting index on bars
    bars.set_index(keys='ts_event', inplace=True)

    # If the last bar is incomplete delete it
    if not bars.empty:
        if bars.iloc[-1]['volume'] < 0.8 * threshold:
            bars = bars.iloc[:-1]

    return bars[['open', 'high', 'low', 'close', 'vwap', 'volume', 'tick_count']]


def dollar_bars_creator(ticks_dataset: pd.DataFrame, 
                        threshold: float = 1.4e7) -> pd.DataFrame:
    """
    Creates Dollar Bars (Dollar-Value Bars) from raw tick data.

    Dollar Bars are sampled every time a cumulative dollar value (Price * Size) of 
    `threshold` is exchanged. 
    
    Why use Dollar Bars?
    They are often considered robust to substantial price fluctuations. Unlike Volume Bars, 
    Dollar Bars adjust the sampling rate based on the asset's value. If the price doubles, 
    it takes half the volume to form a bar, keeping the economic activity per bar constant.
    This produces the best statistical properties (normality, serial correlation) for machine learning.

    Implementation details:
        - Calculates the dollar value per trade vectorially.
        - Uses cumulative sum bucketing for grouping.
        - Prunes the final incomplete bar to ensure data integrity.

    Args:
        ticks_dataset (pd.DataFrame): The cleaned input DataFrame containing executed trades.
            Must contain columns: ['price', 'size', 'ts_event'].
        threshold (float, optional): The aggregate dollar value required to form a single bar.
            Defaults to 14,000,000 (1.4e7), a common starting point for E-mini S&P 500.

    Returns:
        pd.DataFrame: A DataFrame containing OHLCV bars indexed by the timestamp 
        of the *last* tick in each bar.
        Columns: ['open', 'high', 'low', 'close', 'vwap', 'volume', 'tick_count'].
    """
    
    data = ticks_dataset[["ts_event", "price", "size"]].copy()
    
    data["dollar_value"] = data["price"] * data["size"]
    data["dollar_value_cum"] = data["dollar_value"].cumsum()
    data["bar_id"] = (data["dollar_value_cum"] // threshold).astype(np.uint32)
    
    bars = data.groupby(by="bar_id")
    bars = bars.agg({
        "ts_event": "last",
        "price": ['first', 'max', 'min', 'last'],
        "size": ['sum', 'count'],
        "dollar_value": ['sum']
    })
    bars.columns = ['ts_event', 'open', 'high', 'low', 'close', 'volume', 'tick_count', 'dollar_sum']
    
    # If the last bar is icomplete delete it
    if not bars.empty:
        if bars.iloc[-1]['dollar_sum'] < 0.8 * threshold:
            bars = bars.iloc[:-1]

    bars['vwap'] = bars['dollar_sum'] / bars['volume']
    bars.set_index(keys="ts_event", inplace=True)
    
    return bars[['open', 'high', 'low', 'close', 'vwap', 'volume', 'tick_count']]


@njit(cache=True, fastmath=True)
def get_imbalance_bars_numba(
    tick_signs: NDArray,
    roll_indices: NDArray[np.int32],
    initial_T: float = 1000.,
    initial_E_b: float = 0.,
    min_bar_length: int = 10,
    span_fast: int = 100,
    span_slow: int = 1000,
    switch_after_bars: int = 50
) -> NDArray[np.int32]:
    """
    Numba engine for Imbalance Bars with 2-speed EWMA:
      - fast adaptation for the first `switch_after_bars` bars
      - slower/stable adaptation afterward
    """

    T = len(tick_signs)

    # Roll flags: mark the roll at (roll_idx - 1) to close the bar right before the roll tick.
    is_roll = np.zeros(T, dtype=np.bool_)
    if len(roll_indices) > 0:
        for rr in roll_indices:
            r = rr - 1
            if 0 <= r < T:
                is_roll[r] = True

    bar_indices = np.zeros(T, dtype=np.int32)
    bar_count = 0

    theta = 0.0

    # Precompute alphas
    alpha_fast = 2.0 / (span_fast + 1.0)
    alpha_slow = 2.0 / (span_slow + 1.0)

    E_T = initial_T
    E_b = initial_E_b

    last_idx = -1

    # Floor for expected imbalance (constant floor derived from the initial estimate)
    min_expected_imbalance = np.maximum(np.abs(initial_E_b) * 1e-4, 1e-6)

    for i in range(T):
        theta += tick_signs[i]

        expected_imbalance = np.maximum(np.abs(E_b), min_expected_imbalance)
        threshold = E_T * expected_imbalance

        imbalance_trigger = (np.abs(theta) >= threshold) and ((i - last_idx) >= min_bar_length)
        roll_trigger = is_roll[i]

        if imbalance_trigger or roll_trigger:
            bar_indices[bar_count] = i
            # alpha choice: fast on first bars, then slow
            alpha = alpha_fast if bar_count < switch_after_bars else alpha_slow
            bar_count += 1

            current_T = i - last_idx
            E_T = (1.0 - alpha) * E_T + alpha * current_T

            if imbalance_trigger:
                current_b = theta / current_T
                E_b = (1.0 - alpha) * E_b + alpha * current_b

            theta = 0.0
            last_idx = i

    return bar_indices[:bar_count]


def apply_imbalance_bars(df: pd.DataFrame, bar_indices: NDArray[np.int32]) -> pd.DataFrame:
    """
    Aggregates tick data into OHLCV bars based on provided cut indices.

    This function serves as the "Wrapper" for the Imbalance/Run bars logic.
    It takes the raw tick DataFrame and the list of break points (indices) calculated
    by the Numba engine, and performs the aggregation to produce the final bars.

    Implementation details:
        - Uses NumPy ``reduceat`` for sum/max/min aggregations (no pandas groupby).
        - Calculates VWAP (Volume Weighted Average Price) accurately from raw data.

    Args:
        df (pd.DataFrame): The raw tick data containing ['ts_event', 'price', 'size'].
        bar_indices (NDArray[np.int32]): An array of integers representing the index
            of the *last* tick of each bar (output from `get_imbalance_bars_numba`).

    Returns:
        pd.DataFrame: A DataFrame containing OHLCV bars indexed by the timestamp
        of the *last* tick in each bar.
    """
    if len(bar_indices) == 0:
        out = pd.DataFrame(columns=["open","high","low","close","vwap","volume","tick_count"])
        out.set_index(pd.DatetimeIndex([], name="ts_event"), inplace=True)
        return out

    # Extract raw arrays (truncated to last bar — views, no copy)
    end = bar_indices[-1] + 1
    prices = df['price'].values[:end]
    sizes  = df['size'].values[:end]
    ts     = df['ts_event'].values[:end]

    # Build start indices for each bar (reduceat expects start-of-group positions)
    starts = np.empty(len(bar_indices), dtype=np.intp)
    starts[0] = 0
    starts[1:] = bar_indices[:-1] + 1

    # OHLCV via reduceat / direct indexing
    open_  = prices[starts]
    close_ = prices[bar_indices]
    high_  = np.maximum.reduceat(prices, starts)
    low_   = np.minimum.reduceat(prices, starts)
    volume = np.add.reduceat(sizes, starts)
    tick_count = bar_indices - starts + 1

    # VWAP = sum(price * size) / sum(size)
    dollar = prices * sizes
    dollar_sum = np.add.reduceat(dollar, starts)
    vwap = dollar_sum / volume

    # Timestamp = last tick of each bar
    ts_index = ts[bar_indices]

    bars = pd.DataFrame({
        'open': open_,
        'high': high_,
        'low': low_,
        'close': close_,
        'vwap': vwap,
        'volume': volume,
        'tick_count': tick_count,
    }, index=ts_index)
    bars.index.name = 'ts_event'

    return bars


def roll_indices_from_dates(data_ts: NDArray, roll_dates_ns: NDArray) -> NDArray[np.int32]:
    
    idx = np.searchsorted(data_ts, roll_dates_ns, side="left")

    # Keep valid indices strictly inside the array (0 excluded because later we do -1)
    idx = idx[(idx > 0) & (idx < len(data_ts))]

    # Remove duplicates (can happen if multiple roll_dates map to same tick)
    if idx.size > 1:
        idx = np.unique(idx)

    return idx.astype(np.int32)


def _prepare_data(ticks_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Common data preparation shared by all imbalance/runs bar creators.
    Copies the relevant columns, computes the tick rule, and drops NaN rows.
    """
    data = ticks_dataset[["ts_event", "price", "size"]].copy()
    data["tick_rule"] = tick_rule_creator(data["price"])
    data.dropna(subset=["tick_rule"], inplace=True)
    return data


def _ewma_last(values: np.ndarray, span: int) -> float:
    """Return the last EWMA value (same alpha as pandas ewm(span=...))."""
    if values.size == 0:
        return np.nan
    alpha = 2.0 / (span + 1.0)
    e = float(values[0])
    for x in values[1:]:
        e = (1.0 - alpha) * e + alpha * float(x)
    return e


def _warmup_timebars_and_init(
    data: pd.DataFrame,
    x_full: np.ndarray,                   # series fed to numba (tick signs, signed volume, signed $)
    warmup_minutes: int,
    warmup_bars_count: int,
    span: int
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, np.datetime64]:
    """
    Builds warmup time bars on the first warmup_minutes, returns:
      df_warmup, df_algo, initial_T, initial_E_b, cutoff_time
    where initial_* are EWMA estimates computed from warmup bars.
    """
    ts = data["ts_event"].to_numpy(dtype="datetime64[ns]")
    if ts.size == 0:
        raise ValueError("Empty dataset after preparation.")

    start_time = ts[0]
    cutoff_time = start_time + np.timedelta64(int(warmup_minutes), "m")

    # split warmup / algo
    mask_warmup = ts < cutoff_time
    if mask_warmup.sum() == 0:
        raise ValueError("Empty warmup window (no ticks within the warmup period).")
    if (~mask_warmup).sum() == 0:
        raise ValueError("Dataset is shorter than the requested warmup period.")

    df_warmup = data.loc[mask_warmup].copy()
    df_algo = data.loc[~mask_warmup].copy()

    # binning aligned to the first tick (robust, does not depend on resample/origin)
    warmup_seconds_total = float(warmup_minutes) * 60.0
    bar_seconds = warmup_seconds_total / max(int(warmup_bars_count), 1)
    freq_ns = int(max(round(bar_seconds * 1e9), 1))

    ts_w = df_warmup["ts_event"].to_numpy(dtype="datetime64[ns]").astype("datetime64[ns]")
    ts_w_ns = ts_w.astype(np.int64)
    start_ns = ts_w_ns[0]

    bin_id = ((ts_w_ns - start_ns) // freq_ns).astype(np.int64)
    df_warmup["_bin"] = bin_id

    # x for warmup (aligned)
    x_w = x_full[mask_warmup]
    df_warmup["_x"] = x_w

    # --- warmup stats per bar (for init) ---
    g = df_warmup.groupby("_bin", sort=False)
    tick_count = g["_x"].count().astype(np.float64).to_numpy()
    theta = g["_x"].sum().astype(np.float64).to_numpy()
    b_bar = theta / tick_count  # matches the engine update: current_b = theta / current_T

    initial_T = _ewma_last(tick_count, span=span)
    initial_E_b = _ewma_last(b_bar, span=span)

    # fallback safety
    if np.isnan(initial_T):
        initial_T = 1000.0
    if np.isnan(initial_E_b):
        initial_E_b = 0.0

    return df_warmup, df_algo, float(initial_T), float(initial_E_b), cutoff_time


def tick_imbalance_bar_creator(
    ticks_dataset: pd.DataFrame,
    roll_dates: NDArray,
    min_bar_len: int = 10,
    warmup_minutes: int = 3,
    warmup_bars_count: int = 3,
    span: int = 1000,
    include_warmup: bool = False
) -> pd.DataFrame:

    data = _prepare_data(ticks_dataset)

    b_full = data["tick_rule"].to_numpy(dtype=np.int8)

    # warmup -> init_T, init_E_b + split data
    df_warmup, df_algo, initial_T, initial_E_b, cutoff_time = _warmup_timebars_and_init(
        data=data,
        x_full=b_full.astype(np.float64),  # for sums/means (safe)
        warmup_minutes=warmup_minutes,
        warmup_bars_count=warmup_bars_count,
        span=span
    )

    # algo part
    b_algo = df_algo["tick_rule"].to_numpy(dtype=np.int8)

    roll_dates = np.asarray(roll_dates, dtype="datetime64[ns]")
    relevant_rolls = roll_dates[roll_dates >= cutoff_time]
    ts_algo = df_algo["ts_event"].to_numpy(dtype="datetime64[ns]")
    roll_indices = roll_indices_from_dates(ts_algo, relevant_rolls)

    bar_indices_algo = get_imbalance_bars_numba(
        b_algo,
        roll_indices=roll_indices,
        initial_T=initial_T,
        initial_E_b=initial_E_b,
        min_bar_length=min_bar_len,
        span_fast=max(span // 1000, 100),
        span_slow=span
    )

    bars_algo = apply_imbalance_bars(df_algo, bar_indices_algo)

    if not include_warmup:
        return bars_algo

    # build warmup OHLCV bars (same columns as imbalance output)
    dfw = df_warmup.copy()
    dfw["_dollar"] = dfw["price"].to_numpy() * dfw["size"].to_numpy()
    gw = dfw.groupby("_bin", sort=False)

    warmup_bars = pd.DataFrame({
        "open": gw["price"].first(),
        "high": gw["price"].max(),
        "low":  gw["price"].min(),
        "close": gw["price"].last(),
        "volume": gw["size"].sum(),
        "tick_count": gw["price"].count(),
        "dollar_sum": gw["_dollar"].sum(),
        "ts_event": gw["ts_event"].last(),
    }).set_index("ts_event")

    warmup_bars["vwap"] = warmup_bars["dollar_sum"] / warmup_bars["volume"]
    warmup_bars = warmup_bars[["open","high","low","close","vwap","volume","tick_count"]]

    return pd.concat([warmup_bars, bars_algo], axis=0)


def volume_imbalance_bar_creator(
    ticks_dataset: pd.DataFrame,
    roll_dates: NDArray,
    min_bar_len: int = 10,
    warmup_minutes: int = 3,
    warmup_bars_count: int = 3,
    span: int = 1000,
    include_warmup: bool = False
) -> pd.DataFrame:

    data = _prepare_data(ticks_dataset)

    # signed volume series (this is what theta accumulates)
    bv_full = (data["size"].to_numpy(dtype=np.float64) *
               data["tick_rule"].to_numpy(dtype=np.float64))

    # warmup -> init
    df_warmup, df_algo, initial_T, initial_E_b, cutoff_time = _warmup_timebars_and_init(
        data=data,
        x_full=bv_full,  # signed volume
        warmup_minutes=warmup_minutes,
        warmup_bars_count=warmup_bars_count,
        span=span
    )

    # algo part
    bv_algo = (df_algo["size"].to_numpy(dtype=np.float64) *
               df_algo["tick_rule"].to_numpy(dtype=np.float64))

    roll_dates = np.asarray(roll_dates, dtype="datetime64[ns]")
    relevant_rolls = roll_dates[roll_dates >= cutoff_time]
    ts_algo = df_algo["ts_event"].to_numpy(dtype="datetime64[ns]")
    roll_indices = roll_indices_from_dates(ts_algo, relevant_rolls)

    bar_indices_algo = get_imbalance_bars_numba(
        bv_algo,
        roll_indices=roll_indices,
        initial_T=initial_T,
        initial_E_b=initial_E_b,
        min_bar_length=min_bar_len,
        span_fast=max(span // 1000, 100),
        span_slow=span
    )

    bars_algo = apply_imbalance_bars(df_algo, bar_indices_algo)

    if not include_warmup:
        return bars_algo

    # warmup OHLCV bars
    dfw = df_warmup.copy()
    dfw["_dollar"] = dfw["price"].to_numpy() * dfw["size"].to_numpy()
    ts_w = dfw["ts_event"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    start_ns = ts_w[0]
    warmup_seconds_total = float(warmup_minutes) * 60.0
    bar_seconds = warmup_seconds_total / max(int(warmup_bars_count), 1)
    freq_ns = int(max(round(bar_seconds * 1e9), 1))
    dfw["_bin"] = ((ts_w - start_ns) // freq_ns).astype(np.int64)

    gw = dfw.groupby("_bin", sort=False)
    warmup_bars = pd.DataFrame({
        "open": gw["price"].first(),
        "high": gw["price"].max(),
        "low":  gw["price"].min(),
        "close": gw["price"].last(),
        "volume": gw["size"].sum(),
        "tick_count": gw["price"].count(),
        "dollar_sum": gw["_dollar"].sum(),
        "ts_event": gw["ts_event"].last(),
    }).set_index("ts_event")

    warmup_bars["vwap"] = warmup_bars["dollar_sum"] / warmup_bars["volume"]
    warmup_bars = warmup_bars[["open","high","low","close","vwap","volume","tick_count"]]

    return pd.concat([warmup_bars, bars_algo], axis=0)


def dollar_imbalance_bar_creator(
    ticks_dataset: pd.DataFrame,
    roll_dates: NDArray,
    min_bar_len: int = 10,
    warmup_minutes: int = 3,
    warmup_bars_count: int = 3,
    span: int = 1000,
    include_warmup: bool = False
) -> pd.DataFrame:
    """
    Dollar Imbalance Bars with Time-Bar warmup:
      1) Build `warmup_bars_count` time bars over the first `warmup_minutes`
      2) Compute initial_T and initial_E_b via EWMA over those warmup bars
      3) Run standard imbalance engine on the remaining ticks
      4) Optionally prepend warmup bars to output
    """

    # 1) Prep data
    data = _prepare_data(ticks_dataset)

    # signed dollar imbalance series
    bd_full = (
        data["price"].to_numpy(dtype=np.float64)
        * data["size"].to_numpy(dtype=np.float64)
        * data["tick_rule"].to_numpy(dtype=np.float64)
    )

    # 2) Warmup -> init + split
    df_warmup, df_algo, initial_T, initial_E_b, cutoff_time = _warmup_timebars_and_init(
        data=data,
        x_full=bd_full,
        warmup_minutes=warmup_minutes,
        warmup_bars_count=warmup_bars_count,
        span=span
    )

    # 3) Algo part inputs
    bd_algo = (
        df_algo["price"].to_numpy(dtype=np.float64)
        * df_algo["size"].to_numpy(dtype=np.float64)
        * df_algo["tick_rule"].to_numpy(dtype=np.float64)
    )

    roll_dates = np.asarray(roll_dates, dtype="datetime64[ns]")
    relevant_rolls = roll_dates[roll_dates >= cutoff_time]

    ts_algo = df_algo["ts_event"].to_numpy(dtype="datetime64[ns]")
    roll_indices = roll_indices_from_dates(ts_algo, relevant_rolls)

    # 4) Run engine
    bar_indices_algo = get_imbalance_bars_numba(
        bd_algo,
        roll_indices=roll_indices,
        initial_T=initial_T,
        initial_E_b=initial_E_b,
        min_bar_length=min_bar_len,
        span_fast=max(span/1000, 100),
        span_slow=span
    )

    bars_algo = apply_imbalance_bars(df_algo, bar_indices_algo)

    # 5) Optional prepend warmup bars
    if not include_warmup:
        return bars_algo

    # rebuild warmup bar ids (same logic as helper)
    dfw = df_warmup.copy()
    ts_w = dfw["ts_event"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    start_ns = ts_w[0]

    warmup_seconds_total = float(warmup_minutes) * 60.0
    bar_seconds = warmup_seconds_total / max(int(warmup_bars_count), 1)
    freq_ns = int(max(round(bar_seconds * 1e9), 1))

    dfw["_bin"] = ((ts_w - start_ns) // freq_ns).astype(np.int64)
    dfw["_dollar"] = dfw["price"].to_numpy(dtype=np.float64) * dfw["size"].to_numpy(dtype=np.float64)

    gw = dfw.groupby("_bin", sort=False)
    warmup_bars = pd.DataFrame({
        "open": gw["price"].first(),
        "high": gw["price"].max(),
        "low":  gw["price"].min(),
        "close": gw["price"].last(),
        "volume": gw["size"].sum(),
        "tick_count": gw["price"].count(),
        "dollar_sum": gw["_dollar"].sum(),
        "ts_event": gw["ts_event"].last(),
    }).set_index("ts_event")

    warmup_bars["vwap"] = warmup_bars["dollar_sum"] / warmup_bars["volume"]
    warmup_bars = warmup_bars[["open", "high", "low", "close", "vwap", "volume", "tick_count"]]

    return pd.concat([warmup_bars, bars_algo], axis=0)



@njit(cache=True, fastmath=True)
def get_runs_bars_numba(tick_signs: NDArray,
                        roll_indices: NDArray[np.int32], 
                        initial_T: float = 1000.,
                        initial_E_b_plus: float = 0.5,
                        initial_E_b_minus: float = 0.5, 
                        min_bar_length: int = 10,
                        span: int = 1000) -> NDArray[np.int32]:
    """
    Numba optimized engine for Runs Bars generation.
    
    Tracks separate accumulators for Buy runs (theta_plus) and Sell runs (theta_minus).
    A bar is sampled when the maximum of these two accumulators exceeds the expected 
    run size times the expected duration.
    """
    
    # Length of our tick dataset
    T = len(tick_signs)
    
    is_roll = np.zeros(T, dtype=np.bool_)
    if len(roll_indices) > 0:
        for r in (roll_indices - 1):
            if r < T:
                is_roll[r] = True
    
    # Initializing vector of zeros
    bar_indices = np.zeros(T, dtype=np.uint32)
    bar_count = 0
    
    # Initializing theta
    theta_plus = 0.
    theta_minus = 0.
    alpha = 2. / (span + 1)
    
    # Initializing expectations
    E_T = initial_T
    E_b_plus = initial_E_b_plus
    E_b_minus = initial_E_b_minus
    
    # initializing index of the last bar tick
    last_idx = -1
    
    # Computing min_threshold
    min_expected_plus = np.maximum(np.abs(initial_E_b_plus) * 1e-4, 1e-6)
    min_expected_minus = np.maximum(np.abs(initial_E_b_minus) * 1e-4, 1e-6)
    
    # Computing the bars indexes
    for i in range(T):
        # Computing theta plus and theta minus using simple conditions
        if tick_signs[i] > 0:
            theta_plus += tick_signs[i]
        else:
            theta_minus -= tick_signs[i]
        
        # Theta is the largest one
        theta = np.maximum(theta_plus, theta_minus)
        
        # Thresold computation
        expected_plus = np.maximum(np.abs(E_b_plus), min_expected_plus)
        expected_minus = np.maximum(np.abs(E_b_minus), min_expected_minus)
        threshold = E_T * np.maximum(expected_plus, expected_minus)
        
        # Triggers
        imbalance_trigger = theta >= threshold and (i - last_idx) >= min_bar_length
        roll_trigger = is_roll[i]
        
        if imbalance_trigger or roll_trigger:
            # add the index
            bar_indices[bar_count] = i
            bar_count += 1
            
            # update the current mean
            current_T = i - last_idx
            E_T = (1 - alpha) * E_T + alpha * current_T
            
            if imbalance_trigger:
                current_b_plus = theta_plus / current_T
                current_b_minus = theta_minus / current_T
                E_b_plus = (1 - alpha) * E_b_plus + alpha * current_b_plus
                E_b_minus = (1 - alpha) * E_b_minus + alpha * current_b_minus
            
            # Reset
            theta_plus = 0
            theta_minus = 0
            last_idx = i
            
    # return only the non empty values
    return bar_indices[:bar_count]


def tick_runs_bar_creator(ticks_dataset: pd.DataFrame,
                          roll_dates: NDArray,
                          min_bar_len: int = 10,
                          ticks_first_bar: int = 1000,
                          span: int = 1000) -> pd.DataFrame:
    """
    Orchestrates the creation of Tick Runs Bars.

    Runs Bars monitor the sequence of buys and sells separately.
    Unlike Imbalance bars which look at the net flow (Buy - Sell), Runs bars look at 
    the intensity of each side (Max(Buy sequence, Sell sequence)).
    
    This is useful to detect aggressive accumulation or distribution even if 
    the net imbalance is low due to noise.

    Args:
        ticks_dataset (pd.DataFrame): Raw tick data.
        roll_dates (NDArray): Timestamps of contract rolls.
        min_bar_len (int): Minimum ticks per bar.
        ticks_first_bar (int): Warmup period for probability estimation.
        span (int): EWMA window.

    Returns:
        pd.DataFrame: The final Runs OHLCV Bars.
    """
    
    data = _prepare_data(ticks_dataset)
    
    b = data["tick_rule"].values.astype(np.int8)
    
    W = min(len(b), ticks_first_bar)
    temp = b[:W]
    initial_E_b_plus = (temp > 0).mean()
    initial_E_b_minus = 1 - initial_E_b_plus
    
    roll_indices = roll_indices_from_dates(data['ts_event'].values, roll_dates)
    
    bar_indices = get_runs_bars_numba(b,
                                      roll_indices=roll_indices, 
                                      initial_T=float(W), 
                                      initial_E_b_plus=initial_E_b_plus, 
                                      initial_E_b_minus=initial_E_b_minus,
                                      span=span,
                                      min_bar_length=min_bar_len)
    
    return apply_imbalance_bars(data, bar_indices)


def volume_runs_bar_creator(ticks_dataset: pd.DataFrame,
                            roll_dates: NDArray,
                            min_bar_len: int = 10,
                            ticks_first_bar: int = 1000,
                            span: int = 1000) -> pd.DataFrame:
    """
    Orchestrates the creation of Volume Runs Bars.

    This method samples bars when the accumulated volume on one side (Buy OR Sell) 
    exceeds expectations. It is particularly effective at capturing periods of 
    heavy institutional accumulation or distribution.

    Args:
        ticks_dataset (pd.DataFrame): Raw tick data ['ts_event', 'price', 'size'].
        roll_dates (NDArray): Timestamps of contract rolls.
        min_bar_len (int): Minimum ticks per bar.
        ticks_first_bar (int): Warmup period for expectation estimation.
        span (int): EWMA window.

    Returns:
        pd.DataFrame: The final Volume Runs OHLCV Bars.
    """
    
    # 1. Prep data (copy + tick rule + dropna)
    data = _prepare_data(ticks_dataset)

    # 2. Calculate Signed Volume
    # bv contains: +Vol, -Vol, or 0
    bv = (data["size"].values * data["tick_rule"].values).astype(np.float64)
    
    # 5. Initialization (Specific to Volume Runs)
    # We estimate the average volume contribution per tick for each side.
    W = min(len(bv), ticks_first_bar)
    temp = bv[:W]
    
    # Mean of Buy Volume over ALL ticks (Treating sells as 0)
    initial_E_b_plus = np.maximum(temp, 0.).mean()
    # Mean of Sell Volume over ALL ticks (Treating buys as 0, taking abs value)
    initial_E_b_minus = np.maximum(-temp, 0.).mean()
    
    # 6. Re-align Roll Indices (Critical after dropna)
    roll_indices = roll_indices_from_dates(data['ts_event'].values, roll_dates)
    
    # 7. Run Numba Engine
    bar_indices = get_runs_bars_numba(bv, 
                                      roll_indices=roll_indices, 
                                      initial_T=float(W), 
                                      initial_E_b_plus=initial_E_b_plus, 
                                      initial_E_b_minus=initial_E_b_minus,
                                      span=span,
                                      min_bar_length=min_bar_len)
    
    # 8. Final Aggregation
    return apply_imbalance_bars(data, bar_indices)


def dollar_runs_bar_creator(ticks_dataset: pd.DataFrame,
                            roll_dates: NDArray,
                            min_bar_len: int = 10,
                            ticks_first_bar: int = 1000,
                            span: int = 1000) -> pd.DataFrame:
    """
    Orchestrates the creation of Dollar Runs Bars.

    This method samples bars when the accumulated dollar value on one side (Buy OR Sell) 
    exceeds expectations. 
    
    Why use this?
    It is arguably the most sophisticated sampling method in this module. It combines:
    1. The robustness of Dollar Bars (economic value).
    2. The ability of Runs Bars to detect one-sided institutional pressure (accumulation/distribution).

    Args:
        ticks_dataset (pd.DataFrame): Raw tick data ['ts_event', 'price', 'size'].
        roll_dates (NDArray): Timestamps of contract rolls.
        min_bar_len (int): Minimum ticks per bar.
        ticks_first_bar (int): Warmup period for expectation estimation.
        span (int): EWMA window.

    Returns:
        pd.DataFrame: The final Dollar Runs OHLCV Bars.
    """
    
    # 1. Prep data (copy + tick rule + dropna)
    data = _prepare_data(ticks_dataset)

    # 2. Calculate Signed Dollar Value (bd)
    # bd = Price * Size * Sign
    bd = (data['price'].values * data["size"].values * data["tick_rule"].values).astype(np.float64)
    
    # 5. Initialization (Specific to Dollar Runs)
    # We estimate the average dollar contribution per tick for each side.
    W = min(len(bd), ticks_first_bar)
    temp = bd[:W]
    
    # Mean of Buy Dollar Flow over ALL ticks
    initial_E_b_plus = np.maximum(temp, 0.).mean()
    # Mean of Sell Dollar Flow over ALL ticks
    initial_E_b_minus = np.maximum(-temp, 0.).mean()
    
    # 6. Re-align Roll Indices (Critical after dropna)
    roll_indices = roll_indices_from_dates(data['ts_event'].values, roll_dates)
    
    # 7. Run Numba Engine
    bar_indices = get_runs_bars_numba(bd, 
                                      roll_indices=roll_indices, 
                                      initial_T=float(W), 
                                      initial_E_b_plus=initial_E_b_plus, 
                                      initial_E_b_minus=initial_E_b_minus,
                                      span=span,
                                      min_bar_length=min_bar_len)
    
    # 8. Final Aggregation
    return apply_imbalance_bars(data, bar_indices)