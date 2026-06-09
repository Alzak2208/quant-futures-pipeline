# ML Finance

Implementation of quantitative finance techniques from *Advances in Financial Machine Learning* (Marcos Lopez de Prado), applied to CME futures tick data.

## Project Structure

```
├── src/                       # Core library
│   ├── instrument_config.py   # Contract specs & sampling config
│   ├── data_processing.py     # Data loading, cleaning, tick rule
│   ├── bars_creator.py        # Standard, Imbalance & Runs bars (Numba)
│   ├── etf_trick.py           # ETF Trick for continuous series construction
│   └── allocation.py          # PCA-based portfolio weights
├── analysis/                  # Reproducible studies (AFML Ch.2 bar statistics)
├── results/                   # Generated figures & tables
├── notebooks/                 # Exploration & experiments
├── data/                      # Parquet tick data (not versioned)
├── docs/                      # Reference material
├── tests/                     # Unit tests
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

## Results — do information-driven bars actually deliver?

Following *AFML* Chapter 2, I tested whether information-driven bars produce returns with better statistical properties than ordinary time bars, on **~105 M E-mini S&P 500 (ES) trades across ten months of 2022**. Every scheme is calibrated to the same average frequency (~58,900 bars) so the comparison reflects sampling *quality*, not quantity. Reproduce end-to-end with:

```bash
python -m analysis.bar_statistics      # writes the figures and table below to results/
```

### 1. Sampling adapts to information flow

![Weekly bar counts](results/bar_counts_weekly.png)

Time bars emit a near-constant number of bars every week (coefficient of variation just **0.06**) regardless of what the market is doing — they oversample quiet periods and undersample bursts of activity. Information-driven bars instead expand to ~2,500/week during the volatile Sep–Oct 2022 market bottom and contract below ~400 in calm weeks. Sampling follows information, which is the property we want *before* feeding a model.

### 2. Returns are closer to Gaussian

![Standardized return distributions](results/return_distributions.png)

| bar type | excess kurtosis ↓ | Jarque–Bera ↓ | skew | lag-1 autocorr | monthly-var CoV ↓ |
|----------|------------------:|--------------:|-----:|---------------:|------------------:|
| time     | 45.3 | 5,035,015 | −0.61 | −0.0104 | 0.31 |
| tick     | **27.7** | **1,878,181** | 0.34 | 0.0028 | **0.26** |
| volume   | 36.0 | 3,178,764 | 0.39 | **−0.0008** | 0.30 |
| dollar   | 32.0 | 2,514,354 | **−0.09** | 0.0022 | 0.29 |

- **All three information-driven schemes beat time bars on normality** (lower Jarque–Bera *and* lower kurtosis), confirming the AFML thesis on real data.
- **Dollar bars are the most symmetric** (skew ≈ 0) and the best all-rounder — which is why they are the default for everything downstream.
- **Variance is more stationary** month-to-month for tick/dollar bars (lower CoV).
- **Serial correlation is negligible for every scheme.** On an instrument as liquid as the E-mini there is little microstructure noise left to remove at this frequency — the gain here is in the *distribution*, not in autocorrelation. Reporting this honestly matters more than overselling it.

> The heavy tails that remain (excess kurtosis ≫ 0) are **not** data artifacts: the largest bar returns line up with scheduled macro releases — e.g. the **+2.7 % spike on 2022-12-13** at the CPI print — i.e. genuine information events, not bad ticks.

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
