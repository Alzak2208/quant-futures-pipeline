# ML Finance

Implementation of quantitative finance techniques from *Advances in Financial Machine Learning* (Marcos Lopez de Prado), applied to CME futures tick data.

## Project Structure

```
├── src/                    # Core library
│   ├── instrument_config.py   # Contract specs & sampling config
│   ├── data_processing.py     # Data loading, cleaning, tick rule
│   ├── bars_creator.py        # Standard, Imbalance & Runs bars (Numba)
│   ├── etf_trick.py           # ETF Trick for continuous series construction
│   └── allocation.py          # PCA-based portfolio weights
├── notebooks/              # Exploration & experiments
├── data/                   # Parquet tick data (not versioned)
├── docs/                   # Reference material
├── tests/                  # Unit tests
├── requirements.txt
└── pyproject.toml
```

## Pipeline

1. **Data Ingestion** — Load tick-level Parquet files (Databento format) via `data_loader`
2. **Cleaning** — Filter trades, detect contract rolls via `data_cleaner_for_bars`
3. **Bar Sampling** — Transform ticks into information-driven bars:
   - Standard: Tick, Volume, Dollar bars
   - Advanced: Imbalance bars (Tick/Volume/Dollar) with EWMA thresholds
   - Advanced: Runs bars (Tick/Volume/Dollar) for institutional flow detection
4. **ETF Trick** — Splice multiple futures into a continuous synthetic series
5. **Allocation** — PCA-based risk allocation across eigenportfolios

## Instruments

ES (S&P 500), NQ (Nasdaq 100), YM (Dow Jones), CL (Crude Oil), RB (Gasoline), ZN (10Y T-Note), ZF (5Y T-Note), GC (Gold), SI (Silver), 6E (Euro FX).

## Setup

```bash
# Create and activate the conda environment (installs all deps + editable package)
conda env create -f environment.yml
conda activate proj_ml_fin
```

## Tests

```bash
pytest tests/ -v
```
