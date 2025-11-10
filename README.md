# 🔭 Hubble.AI - Treasury Cash-Flow Forecasting Platform

> *"Bringing vision and intelligence to your cash horizon."*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready AI-driven treasury cash flow forecasting platform for weekly 8-week horizons with hierarchical reconciliation.

## 📋 Overview

**Hubble.AI** is a comprehensive forecasting system that:

- **Forecasts** weekly cash flows 8 weeks ahead
- **Hierarchical Structure**: Entity × {TRR, TRP} → Entity Net → Group
- **Probabilistic**: Generates quantile forecasts (p85, p90, p95, p99)
- **Reconciles**: Ensures coherence via MinT/OLS reconciliation
- **Monitors**: Tracks performance against per-horizon targets
- **Supports Multiple Backends**: Local, AWS S3, Azure Blob, Databricks, DENODO

### Key Features

✅ **118 Features** - Daily patterns, lags, rolling stats, calendar, trends, LP
✅ **4 ML Learners** - LightGBM, XGBoost, LSTM, SARIMAX
✅ **3 Multi-Horizon Strategies** - Recursive, Direct, DirRec
✅ **Anti-Leakage** - Rolling-origin CV with embargo
✅ **Hierarchical Reconciliation** - MinT/OLS for coherent forecasts
✅ **Performance Monitoring** - Per-horizon WAPE targets
✅ **CLI + Web UI** - Command-line and Streamlit interface

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd hubble-ai

# Install dependencies
pip install -e .

# Install dev dependencies (for testing)
pip install -e ".[dev]"

# Copy credentials template
cp configs/credentials.example.yml configs/credentials.yml
```

### Configuration

Edit `configs/config.yml` to configure:
- Data backend (local/s3/azure_blob/databricks/denodo)
- Feature settings
- Model hyperparameters
- WAPE targets per horizon

### Running Forecasts

#### Command Line

```bash
# Full pipeline
python -m cf_forecast.cli run --config configs/config.yml --asof 2024-11-12

# Individual steps
python -m cf_forecast.cli ingest --config configs/config.yml --asof 2024-11-12
python -m cf_forecast.cli features --config configs/config.yml --asof 2024-11-12
python -m cf_forecast.cli forecast --config configs/config.yml --asof 2024-11-12
```

#### Using Makefile

```bash
# Run full forecast for specific date
make run-forecast ASOF=2024-11-12

# Run tests
make test

# Launch Streamlit UI
make run-ui
```

#### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Then navigate to http://localhost:8501

## 📂 Project Structure

```
hubble-ai/
├── configs/                 # Configuration files
│   ├── config.yml          # Main configuration
│   └── credentials.yml     # Backend credentials
├── data/
│   ├── raw/                # Raw data: actuals, LP, FX
│   ├── intermediate/       # Validated parquet
│   ├── features/           # Feature snapshots
│   └── reference/          # Entity-liquidity mapping
├── artifacts/
│   ├── models/             # Trained models
│   ├── metrics/            # Performance metrics
│   └── forecasts/          # Output forecasts
├── src/cf_forecast/
│   ├── io.py               # Data backends
│   ├── validation.py       # Data quality checks
│   ├── preprocessing.py    # Daily→weekly aggregation
│   ├── features.py         # 118 features
│   ├── metrics.py          # WAPE/MAE/RMSE/etc
│   ├── cv.py               # Rolling-origin CV
│   ├── modeling/
│   │   ├── baselines.py    # Naïve, LP, SES
│   │   ├── learners.py     # LGBM, XGB, LSTM, SARIMAX
│   │   ├── multi_horizon.py# Recursive, Direct, DirRec
│   │   └── ensemble.py     # Stacking
│   ├── reconciliation.py   # MinT/OLS
│   ├── monitor.py          # Performance tracking
│   ├── pipeline.py         # Orchestration
│   └── cli.py              # CLI interface
├── app/
│   └── streamlit_app.py    # Web UI
├── tests/                  # Test suite
│   ├── test_validation.py
│   ├── test_features.py
│   ├── test_leakage.py     # Critical leakage tests
│   ├── test_metrics.py
│   └── test_reconciliation.py
├── pyproject.toml          # Dependencies
├── Makefile                # Common commands
└── README.md               # This file
```

## 🎯 Architecture

### Forecast Hierarchy

```
Group Net (1 series)
├── Group TRR (1 series)
├── Group TRP (1 series)
└── Entities (15 entities)
    ├── Entity Net = TRR + TRP (15 series)
    └── Bottom Level: Entity × {TRR, TRP} (30 series)
```

**Bottom up forecasting:**
1. Forecast 30 bottom series
2. Reconcile to ensure Entity Net = TRR + TRP
3. Reconcile to ensure Group = Σ Entities

### Feature Engineering

**Order of operations (CRITICAL):**

1. **Daily-level (8 features)** - Calculated BEFORE weekly aggregation
   - Transaction concentration, volatility, counts

2. **Weekly-level (110 features)** - Calculated after aggregation
   - Lags (52): Lag_1 to Lag_52
   - Rolling stats (18): 4w/12w/26w/52w windows
   - Calendar (12): Week, month, quarter, holidays
   - Trends (8): Slopes, momentum, acceleration
   - LP (4): LP_W1 to LP_W4
   - Entity encoding (15): One-hot top 15 entities
   - Liquidity group (1): Is_TRR flag

**Total: 118 features**

### Multi-Horizon Strategies

1. **Recursive**: Train 1-step model, roll forward W1→W8 using predictions as features
2. **Direct**: Train 8 separate models, one per horizon
3. **DirRec**: Hybrid - Direct for W1-W4, Recursive for W5-W8

### Reconciliation

Uses **MinT (Minimum Trace) with OLS** covariance estimation:

```
y_reconciled = S @ (S' @ W^-1 @ S)^-1 @ S' @ W^-1 @ y_base
```

Where:
- `S` = Summing matrix (hierarchy definition)
- `W` = Residual covariance matrix (from CV folds)

### Performance Targets

| Horizon | WAPE Target | Accuracy Target |
|---------|-------------|-----------------|
| W1      | ≤ 5%        | ≥ 95%           |
| W2      | ≤ 7.5%      | ≥ 92.5%         |
| W3      | ≤ 10%       | ≥ 90%           |
| W4      | ≤ 12.5%     | ≥ 87.5%         |
| W5      | ≤ 15%       | ≥ 85%           |
| W6      | ≤ 17.5%     | ≥ 82.5%         |
| W7      | ≤ 20%       | ≥ 80%           |
| W8      | ≤ 22.5%     | ≥ 77.5%         |

**Additional metrics:**
- Directionality ≥ 85% (W1-W2)
- PICP ≥ 80% (prediction interval coverage)

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run fast tests only
pytest tests/ -v -m "not slow and not integration"

# Run leakage tests (critical!)
pytest tests/test_leakage.py -v

# With coverage
pytest tests/ -v --cov=cf_forecast --cov-report=html
```

### Test Coverage

- ✅ Data validation
- ✅ Feature engineering (118 features)
- ✅ Metrics calculations
- ✅ Cross-validation (rolling-origin + embargo)
- ✅ **Data leakage prevention** (critical)
- ✅ Reconciliation (MinT/OLS)

## 📊 Data Requirements

### Input Data

1. **Actuals** (daily transactions)
   - Entity ID
   - Posting Date
   - Amount (EUR)
   - Liquidity Group (TRR/TRP)

2. **Liquidity Plan** (weekly forecasts)
   - Entity ID
   - Week Start
   - LP Amount (EUR)
   - Liquidity Group

3. **Entity Mapping**
   - Entity ID
   - TRR Active (boolean)
   - TRP Active (boolean)

### Data Backends

Supported backends:
- **Local**: Filesystem (CSV/Parquet/Excel)
- **AWS S3**: S3 buckets
- **Azure Blob**: Azure Blob Storage
- **Databricks**: Delta Lake
- **DENODO**: SQL views

Configure in `configs/config.yml`:

```yaml
data_backend:
  profile: "local"  # or s3, azure_blob, databricks_delta, denodo
  options:
    s3_bucket: "your-bucket"
    s3_prefix: "hubble-ai/"
    # ... other backend-specific options
```

## 🔧 Configuration

Key configuration sections in `configs/config.yml`:

### Features
```yaml
features:
  daily_patterns:
    enabled: true
  lags:
    enabled: true
    max_lag: 52
  rolling_stats:
    enabled: true
    windows: [4, 12, 26, 52]
```

### Modeling
```yaml
modeling:
  horizons: [1,2,3,4,5,6,7,8]
  learners: ["lightgbm","xgboost","lstm","sarimax"]
  quantiles: [0.85, 0.90, 0.95, 0.99]
  strategy:
    recursive: true
    direct_horizons: [1,2,3,4,5,6,7,8]
    dirrec: true
```

### Monitoring
```yaml
monitoring:
  wape_targets:
    "1": 0.05
    "2": 0.075
    "3": 0.10
    # ...
  directionality_min:
    "1": 0.85
    "2": 0.85
  picp_target: 0.80
```

## 📈 Output Formats

Forecasts are saved in two formats:

### 1. Parquet (for programmatic access)
```
artifacts/forecasts/forecasts_20241112.parquet
```

Partitioned by `asof_week`

### 2. Excel (for business users)
```
artifacts/forecasts/forecasts_20241112.xlsx
```

Sheets:
- **Bottom_Leaf**: 30 series (Entity × {TRR, TRP})
- **Entity_Net**: 15 series (Entity Net)
- **Group_TRR**: 1 series (Total TRR)
- **Group_TRP**: 1 series (Total TRP)
- **Group_Net**: 1 series (Total Net)
- **Metrics**: Performance metrics

Each forecast includes:
- Point forecast (p50)
- Quantiles: p85, p90, p95, p99
- Metadata: asof_date, entity, liquidity_group, horizon

## 🚨 Critical Design Rules

### 1. No Data Leakage
- Features use data ≤ forecast cut-off
- Recursive strategy uses predictions, not future actuals
- Rolling-origin CV with embargo=1 week

### 2. Feature Order
- Daily patterns computed BEFORE weekly aggregation
- Weekly features computed AFTER aggregation

### 3. Hierarchical Coherence
- Forecasts reconciled bottom-up
- Entity Net = TRR + TRP (enforced)
- Group = Σ Entities (enforced)

### 4. Reproducibility
- Fixed random seeds
- Data hash tracking
- Feature snapshot IDs
- Model versioning
- Git SHA in artifacts

## 🤝 Contributing

1. Follow the coding standards (black, ruff)
2. Add tests for new features
3. Ensure `test_leakage.py` passes
4. Update documentation

```bash
# Format code
make format

# Lint code
make lint

# Run tests
make test
```

## 📝 License

MIT License - see LICENSE file for details

## 👥 Authors

**Aperam S.A. Treasury Analytics Team**

## 📞 Support

For questions or issues:
1. Check documentation in `docs/`
2. Review `Claude.md` for architecture details
3. Open an issue on GitHub

---

**Version**: 1.0.0
**Last Updated**: 2025-11-10
**Status**: Production-Ready

*Bringing vision and intelligence to your cash horizon.* 🔭
