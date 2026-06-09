"""
ETF Trick
=========

Splices a set of disjoint futures contracts into a single continuous, tradable
price series, following the "ETF Trick" from *Advances in Financial Machine
Learning* (Lopez de Prado, Section 2.4).

The output series K tracks the value of an initial $1 allocation as it is rolled
from one contract to the next without injecting artificial price gaps. At each
roll the P&L is taken from Close minus Open (instead of Close minus Close) and
the holdings h are rebalanced to keep dollar exposure constant. A Numba core
performs the recursive update of K and h. Both fixed and time-varying weights
are supported.

Dependencies:
    - numpy
    - pandas
    - numba (JIT compilation of the recursive core engine)
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Dict, List, Tuple, Union, Optional
from numpy.typing import NDArray

class ETFTrick:
    """
    Implements the 'ETF Trick' algorithm as described by Marcos López de Prado.
    
    This class generates a continuous price series (K) and a holdings matrix (h) 
    from a set of disjoint futures contracts. It handles the 'roll' (switch from 
    one contract to another) by rebalancing positions to maintain a constant 
    dollar exposure, effectively splicing the series without artificial price gaps.

    Attributes:
        future_specs (Dict): Dictionary containing contract metadata (point values, tickers).
        K (pd.Series): The resulting synthetic cumulative dollar value series.
        h (pd.DataFrame): The resulting number of contracts held for each asset over time.
        _is_runned (bool): Internal flag to check if the model has been fitted.
    """

    def __init__(self, future_specs: Dict[str, Dict[str, float]]):
        """
        Initializes the ETFTrick engine with contract specifications.

        Args:
            future_specs (Dict[str, Dict[str, float]]): A dictionary defining specifications 
                for the futures (e.g., {'ES.c.0': {'point_value': 50.0}}).
        """
        self.future_specs = future_specs
        self._is_runned = False
        self.K: Optional[pd.Series] = None
        self.h: Optional[pd.DataFrame] = None
    

    def run(self, 
            etf_data: pd.DataFrame, 
            rolling_dict: Dict[str, NDArray], 
            weights: Union[NDArray[np.float64], pd.DataFrame],
            value_one_point: NDArray = None
            ) -> 'ETFTrick':
        """
        Main orchestrator: prepares data, aligns rolls, and executes the Numba core engine.
        Acts as a 'fit' method, populating the internal state (K and h).

        Steps:
        1. Computes price deltas (Close - Close) and corrects them at roll dates (Close - Open).
        2. Aligns the provided roll dates to the dataframe index to create a boolean mask.
        3. Normalizes weights into a (N, M) matrix.
        4. Runs the recursive loop (Numba) to compute K and h.

        Args:
            etf_data (pd.DataFrame): Raw dataframe containing 'open_ASSET' and 'close_ASSET' columns.
            rolling_dict (Dict[str, NDArray]): Dictionary mapping asset names to their specific roll 
                timestamps (numpy arrays of datetime64).
            weights (Union[NDArray, pd.DataFrame]): Allocation weights. Can be a 1D array (fixed weights) 
                or a DataFrame/2D array (dynamic weights).
            value_one_point (NDArray, optional): Overrides the point values found in future_specs. 
                Defaults to None.

        Returns:
            ETFTrick: The instance itself (self), allowing for method chaining.
        """
              
        # Prepare datas
        data = etf_data
        assets, mat_open, mat_close = self._get_price_matrices(data)
        n_rows = len(data)
        
        index_dt64 = data.index.to_numpy(dtype="datetime64[ns]", copy=False)
        mat_delta, roll_mask = self._delta_and_rollmask_numpy(
            index_dt64=index_dt64,
            mat_close=mat_close,
            mat_open=mat_open,
            assets=assets,
            rolling_dict=rolling_dict
        )
        
        # Computing contigous array
        mat_open  = np.ascontiguousarray(mat_open,  dtype=np.float64)
        mat_close = np.ascontiguousarray(mat_close, dtype=np.float64)
        mat_delta = np.ascontiguousarray(mat_delta, dtype=np.float64)
        
        # 3. Prepare Weights & Point Values (Phi)
        sum_abs_weights, mat_weights = self._prepare_weights(weights, n_rows)
        phi = self._get_point_values(assets, value_one_point)
        
        # 4. Initialize loop value
        K = np.ones(n_rows, dtype=np.float64)
        h = np.zeros((n_rows, len(assets)), dtype=np.float64)
        
        # Initial condition
        h[0, :] = (K[0] * mat_weights[0, :]) / (mat_close[0, :] * phi * sum_abs_weights[0])
        
        # 5. Execute Numba Core Engine
        K, h = self._core_engine_numba(K, h, phi, mat_weights, mat_open, mat_delta, sum_abs_weights, roll_mask)
        
        # 6. Store results in Pandas format
        self.K = pd.Series(K, index=data.index, name="ETF_Trick")
        self.h = pd.DataFrame(h, index=data.index, columns=[f"h_{a}" for a in assets])
        self._is_runned = True
        
        return self
    

    def get_results(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Retrieves the results of the backtest.

        Raises:
            ValueError: If the run() method has not been called yet.

        Returns:
            Tuple[pd.Series, pd.DataFrame]: 
                - K: The synthetic ETF price series.
                - h: The holdings (number of contracts) dataframe.
        """
        if not self._is_runned:
            raise ValueError("Model has not been run yet. Please call .run() first.")
        return self.K, self.h
    
    
    @staticmethod
    @njit(cache=True, fastmath=True)
    def _core_engine_numba(K: NDArray[np.float64], 
                           h: NDArray[np.float64], 
                           phi: NDArray[np.float64], 
                           weights_mat: NDArray[np.float64],
                           mat_open: NDArray[np.float64],
                           mat_delta: NDArray[np.float64],
                           sum_abs_w: NDArray[np.float64],
                           roll_mask: NDArray[np.bool_]
                           ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        High-performance JIT-compiled core engine.
        
        It iterates through time to update the cumulative value K and the holdings h.
        Handles the 'roll' logic: if a roll is detected at t (via roll_mask), the position 
        is rebalanced using the Open price of t+1.

        Args:
            K (NDArray): Array to store the cumulative value series (initialized with ones).
            h (NDArray): Array to store the holdings (contracts) over time.
            phi (NDArray): Point values for each asset.
            weights_mat (NDArray): Matrix of weights (N, M).
            mat_open (NDArray): Matrix of Open prices.
            mat_delta (NDArray): Matrix of price changes (Close-Close or Close-Open).
            sum_abs_w (NDArray): Vector of the sum of absolute weights per timestamp.
            roll_mask (NDArray): Boolean mask indicating if a rebalance is needed at step t.

        Returns:
            Tuple[NDArray, NDArray]: The populated K and h arrays.
        """
        n = K.shape[0]
        n_asset = h.shape[1]
        
        inv_phi = 1. / phi
        
        for t in range(1, n):
            h_prev = h[t - 1]
            K_prev = K[t - 1]
            delta_t = mat_delta[t]
            pnl = 0.
            
            if roll_mask[t] and t < (n - 1):
                for i in range(n_asset):
                    pnl += h_prev[i] * phi[i] * delta_t[i]
            
                Kt = K_prev + pnl
                K[t] = Kt
                
                # Rebalance
                common = (Kt / sum_abs_w[t])
                w_t = weights_mat[t]
                open_next = mat_open[t + 1]
                for i in range(n_asset):
                    h[t, i] = (common * w_t[i]) * (inv_phi[i] / open_next[i])
            
            else:
                for i in range(n_asset):
                    hi = h_prev[i]
                    pnl += hi * phi[i] * delta_t[i]
                    h[t, i] = hi
                K[t] = K_prev + pnl
        
        return K, h
    
    
    def _get_point_values(self, assets: List[str], 
                          value_one_point: Optional[NDArray]
                          ) -> NDArray[np.float64]:
        """Retrieves or casts the point values (phi) for the assets."""
        if value_one_point is None:
            # Note: Assumes keys in future_specs match the format '{asset}.c.0'
            try:
                phi = np.array([self.future_specs[f"{asset}.c.0"]['point_value'] for asset in assets], 
                               dtype=np.float64)
            except KeyError as e:
                raise KeyError(f"Asset config not found in future_specs for key: {e}")
        else:
            phi = value_one_point.astype(np.float64)
        
        return phi
    
    
    def _get_price_matrices(self, 
                            data: pd.DataFrame
                            ) -> tuple[List[str], NDArray[np.float64], NDArray[np.float64]]:
        """
        Extracts asset names and converts price columns to numpy arrays.
        Ensures strict alignment between Close and Open matrices.
        """
        cols = data.columns
    
        # 1. Identify Close columns first (Master list)
        close_cols = [col for col in cols if col.startswith("close_")]
        close_cols.sort()
        
        if not close_cols:
            raise ValueError("No columns starting with 'close_' found in the dataframe.")
        
        # 2. Derive Asset names and Open columns
        prefix = "close_"
        assets = [col[len(prefix):] for col in close_cols]
        open_cols = [f"open_{a}" for a in assets]
        
        # 3. Check for missing Open columns before crashing in NumPy
        colset = set(cols)
        missing_opens = [c for c in open_cols if c not in colset]
        if missing_opens:
            raise ValueError(f"Missing corresponding open columns: {missing_opens}")
        
        # Convert to numpy arrays for Numba efficiency
        mat_close = np.ascontiguousarray(data[close_cols].to_numpy(dtype=np.float64, copy=False))
        mat_open = np.ascontiguousarray(data[open_cols].to_numpy(dtype=np.float64, copy=False))
        
        return assets, mat_open, mat_close
    
    
    def _delta_and_rollmask_numpy(self,
                                  index_dt64: NDArray,
                                  mat_close: NDArray[np.float64],
                                  mat_open: NDArray[np.float64],
                                  assets: List[str],
                                  rolling_dict: Dict[str, NDArray],
                                  ) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
        n, m = mat_close.shape
        
        # index as int64 without copy (nanoseconds)
        idx_int = index_dt64.view("i8")
        
        # 1) base delta: close[t] - close[t-1]
        mat_delta = np.empty_like(mat_close, dtype=np.float64)
        mat_delta[0, :] = 0.0
        mat_delta[1:, :] = mat_close[1:, :] - mat_close[:-1, :]
        
        # 2) roll mask (rebalance at t = roll_day - 1)
        roll_mask = np.zeros(n, dtype=np.bool_)
        
        # map asset -> column index
        asset_to_col = {a: j for j, a in enumerate(assets)}
        
        for asset, dates in rolling_dict.items():
            j = asset_to_col.get(asset, -1)
            if j == -1:
                continue
            if dates is None:
                continue
            d = np.asarray(dates)
            if d.size == 0:
                continue
            
            if d.dtype != "datetime64[ns]":
                d = d.astype("datetime64[ns]")
            d_int = d.view("i8")

            # insertion indices (robust if timestamp not exactly present)
            idx = np.searchsorted(idx_int, d_int)

            # keep indices inside bounds and allow idx-1
            idx = idx[(idx > 0) & (idx < n)]

            if idx.size == 0:
                continue

            # stitch delta on roll day: close_old(t) - open_new(t)
            mat_delta[idx, j] = mat_close[idx, j] - mat_open[idx, j]

            # rebalance the day before
            roll_mask[idx - 1] = True

        return mat_delta, roll_mask
    
    
    def _prepare_weights(self, weights, n_rows):
        # 1) Convert once
        if isinstance(weights, pd.DataFrame):
            w = weights.to_numpy(dtype=np.float64, copy=False)
        else:
            w = np.asarray(weights, dtype=np.float64)

        # 2) Fixed weights (1D): no tile
        if w.ndim == 1:
            sum_abs = np.abs(w).sum()
            if sum_abs == 0.0:
                raise ValueError("Sum of abs(weights) is zero (division by zero).")

            sum_weight_abs = np.full(n_rows, sum_abs, dtype=np.float64)
            mat_weights = np.broadcast_to(w, (n_rows, w.size))  # view, read-only

            return sum_weight_abs, mat_weights

        # 3) Dynamic weights (2D)
        mat_weights = np.ascontiguousarray(w)  # good for numba
        sum_weight_abs = np.abs(mat_weights).sum(axis=1).astype(np.float64, copy=False)

        return sum_weight_abs, mat_weights