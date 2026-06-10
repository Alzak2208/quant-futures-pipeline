"""
instrument_config.py

This file centralizes all static metadata for financial instruments used in the
ETF Trick implementation and position calculations.

SOURCES:
- CME Group Contract Specs (https://www.cmegroup.com)
- Databento Symbology
"""

# ==========================================
# 1. CONTRACT SPECIFICATIONS (FUTURES)
# ==========================================

# Key: The symbol as it appears in your Databento files (e.g., 'NQ.c.0')
# Values:
#   - point_value: Dollar value of a single full point movement (The "Multiplier").
#   - tick_size: The minimum price movement.
#   - currency: Quoting currency.
#   - asset_class: Asset class (used for Clustering).
#   - margin_req: Estimated % margin requirement (Maintenance Margin / Notional).
#                 (Backtest simplification: 10% to 15% is standard).

FUTURE_SPECS = {
    # --- INDICES (EQUITY) ---
    "ES.c.0": {
        "name": "E-mini S&P 500",
        "point_value": 50.0,
        "tick_size": 0.25,
        "asset_class": "Equity",
        "roll_frequency": "Quarterly",  # Mar, Jun, Sep, Dec
        "margin_req": 0.12,  # ~12%
    },
    "NQ.c.0": {
        "name": "E-mini Nasdaq 100",
        "point_value": 20.0,
        "tick_size": 0.25,
        "asset_class": "Equity",
        "roll_frequency": "Quarterly",
        "margin_req": 0.12,
    },
    "YM.c.0": {
        "name": "E-mini Dow Jones ($5)",
        "point_value": 5.0,
        "tick_size": 1.0,
        "asset_class": "Equity",
        "roll_frequency": "Quarterly",
        "margin_req": 0.10,
    },
    # --- RATES / BONDS ---
    "ZN.c.0": {
        "name": "10-Year T-Note",
        "point_value": 1000.0,
        "tick_size": 0.015625,  # 1/64 of a point (often displayed in decimals)
        "asset_class": "Rates",
        "roll_frequency": "Quarterly",
        "margin_req": 0.05,  # Bonds usually have lower margins (less volatile)
    },
    "ZF.c.0": {
        "name": "5-Year T-Note",
        "point_value": 1000.0,
        "tick_size": 0.0078125,  # 1/128
        "asset_class": "Rates",
        "roll_frequency": "Quarterly",
        "margin_req": 0.03,
    },
    # --- ENERGY ---
    "CL.c.0": {
        "name": "Crude Oil WTI",
        "point_value": 1000.0,
        "tick_size": 0.01,
        "asset_class": "Energy",
        "roll_frequency": "Monthly",  # Expires every month
        "margin_req": 0.15,  # Higher volatility
    },
    "RB.c.0": {
        "name": "RBOB Gasoline",
        "point_value": 42000.0,  # 42,000 Gallons.
        # WARNING: If price is 2.5000, Notional = 2.5 * 42000 = $105,000
        "tick_size": 0.0001,
        "asset_class": "Energy",
        "roll_frequency": "Monthly",
        "margin_req": 0.15,
    },
    # --- METALS ---
    "GC.c.0": {
        "name": "Gold (Comex)",
        "point_value": 100.0,
        "tick_size": 0.10,
        "asset_class": "Metals",
        "roll_frequency": "Bi-Monthly",  # Feb, Apr, Jun...
        "margin_req": 0.10,
    },
    "SI.c.0": {
        "name": "Silver (Comex)",
        "point_value": 5000.0,
        "tick_size": 0.005,
        "asset_class": "Metals",
        "roll_frequency": "Bi-Monthly",
        "margin_req": 0.12,
    },
    # --- FX ---
    "6E.c.0": {
        "name": "Euro FX",
        "point_value": 125000.0,  # 125,000 Euros per contract
        "tick_size": 0.00005,
        "asset_class": "FX",
        "roll_frequency": "Quarterly",
        "margin_req": 0.05,
    },
}

# ==========================================
# 2. DOLLAR IMBALANCE SAMPLING CONFIG
# ==========================================

# Calibrated parameters to achieve ~10,000 bars per month (March 2020 Data).
# Keys correspond to the asset symbol short code.

BAR_SAMPLING_CONFIG = {
    # --- INDICES ---
    # ES: The Benchmark (Target: ~10k bars)
    "ES": {"span": 250000, "warmup_min": 1440, "warmup_bars": 480},
    # NQ: Tech / High Beta (Target: ~12.8k bars)
    # Calibrated at 310k to manage high volatility.
    "NQ": {"span": 310000, "warmup_min": 1440, "warmup_bars": 480},
    # YM: Industrial / High Nominal Value (Target: ~10.6k bars)
    # Calibrated at 380k to compensate for the large point price.
    "YM": {"span": 380000, "warmup_min": 1440, "warmup_bars": 480},
    # --- ENERGY ---
    # CL: Energy Leader (Target: ~9.6k bars)
    "CL": {"span": 55000, "warmup_min": 1440, "warmup_bars": 480},
    # RB: Gasoline / Lower Liquidity (Target: ~6.7k bars)
    # Calibrated at 14k to filter noise while keeping signal.
    "RB": {"span": 14000, "warmup_min": 1440, "warmup_bars": 480},
    # --- FX ---
    # 6E: Major Currency (Target: ~13k bars)
    "6E": {"span": 50000, "warmup_min": 1440, "warmup_bars": 480},
}

# Final list of assets to process (excluding corrupted or illiquid contracts like GC, SI, ZF)
ACTIVE_ASSETS = ["ES", "NQ", "YM", "CL", "RB"]

# ==========================================
# 3. ECONOMIC PARAMETERS (GLOBAL)
# ==========================================

ECONOMIC_PARAMS = {
    # Default Risk-Free Rate if no FRED data is provided.
    # In March 2020, rates dropped towards 0.25%, previously ~1.5%.
    "default_risk_free_rate": 0.02,  # 2% per year (simplification)
    # Estimated transaction cost per trade (Slippage + Commission)
    # Expressed in "Ticks". E.g., 1 tick of slippage per roll.
    "transaction_cost_ticks": 1.0,
    # Dividends & Coupons
    # For Futures, these are implied (Future Price = Spot + Cost of Carry - Dividends).
    # We set to 0 because we trade the Future price which already discounts dividends.
    "implied_dividend_yield": 0.0,
}

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================


def get_point_value(symbol: str) -> float:
    """Returns the dollar value of a single point for a given symbol."""
    if symbol not in FUTURE_SPECS:
        raise ValueError(f"Unknown symbol in config: {symbol}")
    return FUTURE_SPECS[symbol]["point_value"]


def get_margin_rate(symbol: str) -> float:
    """Returns the estimated required margin rate."""
    return FUTURE_SPECS.get(symbol, {}).get("margin_req", 0.10)  # Default to 10%


def get_asset_class(symbol: str) -> str:
    """Returns the asset class (useful for clustering)."""
    return FUTURE_SPECS.get(symbol, {}).get("asset_class", "Unknown")


if __name__ == "__main__":
    # Small test to verify functionality
    test_sym = "NQ.c.0"
    print(f"Test for {test_sym} ({FUTURE_SPECS[test_sym]['name']}):")
    print(f" - Multiplier: ${get_point_value(test_sym)}")
    print(f" - Margin Req: {get_margin_rate(test_sym) * 100}%")

    # Display Sampling Config
    print("\n--- Sampling Configuration ---")
    for asset in ACTIVE_ASSETS:
        conf = BAR_SAMPLING_CONFIG.get(asset, "N/A")
        print(f"[{asset}] Span: {conf.get('span')}")
