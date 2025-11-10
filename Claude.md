# Treasury Cash Flow Forecasting Application

## Overview

Production-ready Python application for weekly 8-week cash flow forecasting at Aperam S.A. Forecast hierarchy: Group Net ➜ Entity ➜ Liquidity Group (TRR, TRP). Bottom level: all valid (Entity × {TRR, TRP}) pairs from `data/reference/entity_liquidity_map.csv`.

**Goal:** Weekly run ingests actuals and LP data, validates, engineers ~118 features (daily patterns before weekly aggregation), trains/evaluates 4 learners (LightGBM, XGBoost, LSTM, SARIMAX), produces 8-week forecasts at leaf level, reconciles to Entity Net and Group totals with MinT/OLS, publishes probabilistic intervals p85/p90/p95/p99 (optional p75), monitors performance vs per-horizon targets, persists outputs to pluggable storage.

Includes CLI and Streamlit UI. Cloud scheduling stubs (Azure ML, SageMaker) provided but disabled.

---

## Critical Reference Documents

**Location:** `docs/` (project documentation)

Claude Code MUST review these documents before implementation:

1. **`TCFF_Conversation.docx`** - Full project context, business requirements, technical decisions, accuracy targets
2. **`Comprehensive_Feature_List_for_Forecasting_Model.docx`** - Complete feature engineering specification (~118 features)
3. **`Business case`**
4. **`Requirements doc`**

**Location:** `data/reference/`
4. **`entity_liquidity_map.csv`** - Entity → {TRR, TRP} mapping (defines bottom hierarchy)

**Location:** `data/raw/` (Excel files for V1)
5. **Actuals data** - Daily transaction-level cash flows
6. **Liquidity Plan (LP) data** - Treasury's 4-week forward forecasts  
7. **FX rates** - Historical EUR exchange rates for currency conversion

**Note for V1:** Data provided as Excel extracts. Future: direct connection to data sources (DENODO, or Databricks, or S3 or Blob storage etc.) with full pipeline from scratch each week to handle data/model drift.

---

## Repository Structure

```
cashflow-forecasting/
├── README.md
├── Claude.md                    # This file
├── pyproject.toml
├── Makefile
├── configs/
│   ├── config.yml              # Main configuration
│   └── credentials.example.yml
├── data/
│   ├── raw/                    # Excel extracts: actuals, LP, FX
│   ├── intermediate/           # Validated & conformed parquet
│   ├── features/               # Feature snapshots by as-of date
│   └── reference/
│       └── entity_liquidity_map.csv
├── artifacts/
│   ├── models/
│   ├── metrics/
│   └── forecasts/
├── app/
│   └── streamlit_app.py        # Treasury UI
├── notebooks/
│   └── EDA.ipynb               # Exploratory only (read-only for prod)
├── src/
│   └── cf_forecast/
│       ├── __init__.py
│       ├── io.py               # DataBackend (FS/S3/Blob/Databricks/Denodo)
│       ├── validation.py       # Data quality checks
│       ├── preprocessing.py    # Daily→weekly aggregation
│       ├── features.py         # ~118 features (daily patterns, then weekly)
│       ├── metrics.py          # WAPE/MAE/RMSE/Directionality/PICP
│       ├── cv.py               # Rolling-origin CV + embargo
│       ├── modeling/
│       │   ├── baselines.py    # Naïve, LP-as-is, SES
│       │   ├── learners.py     # LGBM, XGB, LSTM, SARIMAX
│       │   ├── multi_horizon.py# Recursive, Direct(1–8), DirRec
│       │   └── ensemble.py     # Stacking (ridge) across learners
│       ├── reconciliation.py   # S builder + MinT/OLS
│       ├── monitor.py          # Per-horizon thresholds, drift, fallback
│       ├── pipeline.py         # Orchestrates ingest→forecast→publish
│       └── cli.py              # CLI entrypoints
└── tests/
    ├── test_validation.py
    ├── test_features.py
    ├── test_cv.py
    ├── test_metrics.py
    ├── test_modeling.py
    ├── test_reconciliation.py
    └── test_leakage.py
```

---

## Non-Negotiable Design Rules

1. **No leakage:** Features use data ≤ forecast cut-off; Recursive strategy uses its own predictions, not future actuals
2. **CV:** Rolling-origin with embargo=1 week (buffer between train and test)
3. **Coherence:** Forecast leaves → reconcile to Entity Net, Group TRR/TRP, Group Net via MinT/OLS
4. **Probabilistic:** Publish p85, p90, p95, p99 (optional p75) and track PICP per horizon
5. **Reproducibility:** Fixed seeds, data hash, feature snapshot ID, model version, git SHA in artifacts
6. **Constraints:** TRR ≥ 0, TRP ≤ 0 (train on magnitudes if preferred, reapply sign; clip residual violations)
7. **Feature Engineering Order:** Calculate daily pattern features (8) from daily data BEFORE aggregating to weekly level

---

## Configuration (configs/config.yml)

```yaml
project:
  timezone: "Europe/Luxembourg"
  week_close: "Tuesday 12:00"        # Weekly cut-off for ingestion

paths:
  raw: "data/raw/"
  intermediate: "data/intermediate/"
  features: "data/features/"
  reference: "data/reference/"
  artifacts: "artifacts/"

data_backend:
  profile: "local"                   # local | s3 | azure_blob | databricks_delta | denodo
  options:
    s3_bucket: ""
    s3_prefix: ""
    azure_container: ""
    azure_account_url: ""
    databricks_catalog: ""
    denodo_dsn: ""

modeling:
  horizons: [1,2,3,4,5,6,7,8]
  learners: ["lightgbm","xgboost","lstm","sarimax"]
  quantiles: [0.85, 0.90, 0.95, 0.99]
  quantile_optional: [0.75]
  strategy:
    recursive: true                  # 1-step model, predictions replace lags
    direct_horizons: [1,2,3,4,5,6,7,8]
    dirrec: true
  hierarchy:
    type: "grouped"                  # Grouped time-series via S matrix
    leaves: ["TRR","TRP"]            # Per entity
  reconciliation: "mint_ols"

cv:
  n_splits: 5
  embargo_weeks: 1
  min_history_weeks: 52

monitoring:
  # Per-horizon entity-level WAPE targets
  # Business requirement: W1-W2: 5%, W3-W4: 10%, W5-W6: 15%, W7-W8: 20%
  wape_targets:
    "1": 0.05    # 95% accuracy
    "2": 0.075   # 92.5% accuracy
    "3": 0.10    # 90% accuracy
    "4": 0.125   # 87.5% accuracy
    "5": 0.15    # 85% accuracy
    "6": 0.175   # 82.5% accuracy
    "7": 0.20    # 80% accuracy
    "8": 0.225   # 77.5% accuracy
  wow_wape_deterioration_pp: 3
  directionality_min:
    "1": 0.85
    "2": 0.85
  picp_target: 0.80
  drift_psi_threshold: 0.20

publishing:
  write_parquet: true
  write_excel: true
  partition_by: ["asof_week"]
  excel_filename: "forecasts_{asof}.xlsx"

ui:
  enable_streamlit: true

cloud_stubs:
  enable_azure_ml_stub: false        # Stubs provided but disabled
  enable_sagemaker_stub: false
```

**Note on hierarchy:**
- Bottom (leaves): (Entity, TRR), (Entity, TRP) from entity_liquidity_map.csv
- Entity Net = TRR + TRP
- Group TRR = Σ TRR_e, Group TRP = Σ TRP_e
- Group Net = Σ (TRR_e + TRP_e)

---

## Data Contract

### Inputs

**Actuals (Daily transactions):**
```
entity_id: str
posting_date: date
payment_currency: str
amount_functional: float (EUR)
liquidity_group: str  # TRR or TRP
...
```

**Liquidity Plan (Weekly forecasts):**
```
entity_id: str
liquidity_group: str
week_start: date (ISO Monday)
lp_amount: float (EUR)
```

**Reference:** `data/reference/entity_liquidity_map.csv`
```
entity_id, trr_active, trp_active
Entity_057, TRUE, TRUE
Entity_11G5, TRUE, TRUE
...
```
*If entity not in map: defaults to TRR=TRUE, TRP=TRUE*

### Validation Rules (validation.py)

- Schema & dtypes enforcement
- No duplicates: (entity_id, liquidity_group, week_start)
- Sign policy: TRR ≥ 0, TRP ≤ 0
- Date continuity: ≤1% gaps per entity × liquidity_group
- FX source consistency
- Fail-fast on schema; log issues on quality

---

## Feature Engineering (~118 Features)

**CRITICAL:** Follow exact specification in `Comprehensive_Feature_List_for_Forecasting_Model.docx`

### Feature Calculation Order

1. **Daily-level (8 features):** Calculate from raw daily transactions BEFORE aggregation
   - Transaction concentration (4): Pct_Txn_Month_End_Day, Pct_Txn_First_Day, Pct_Txn_Last_Day, Pct_Txn_Friday
   - Daily volatility (2): Daily_Std_Within_Week, Daily_CV_Within_Week
   - Transaction patterns (2): Num_Days_With_Txn, Num_Transactions_Week

2. **Aggregate to weekly:** Sum daily transactions → weekly level

3. **Weekly-level (110 features):**
   - Lag features (52): Lag_1 to Lag_52
   - Rolling statistics (18): Means, medians, stds, min/max, range (4wk, 12wk, 26wk windows)
   - Calendar features (12): Week_of_Year, Month, Quarter, binary flags (month-end, quarter-end, holidays, etc.)
   - Trend features (8): Slopes, momentum, acceleration, relative position
   - Liquidity Plans (4): LP_W1, LP_W2, LP_W3, LP_W4 (treasury's forward forecasts)
   - Entity encoding (15): One-hot for 15 Tier 1 entities
   - Liquidity group (1): Is_TRR binary flag

**Total: 8 + 52 + 18 + 12 + 8 + 4 + 15 + 1 = 118 features**

### Cut-off Discipline

**For forecast as-of Week N:**
- ✅ Can use: Weeks N-1, N-2, ..., N-52 (historical actuals)
- ✅ Can use: Weeks N+1, N+2, N+3, N+4 (current treasury LP)
- ❌ Cannot use: Weeks N+1, N+2, ... actuals (future!)

### Recursive Strategy Impact

During Week 2+ predictions:
- **Lag_1 becomes W1 prediction** (not actual)
- **Lag_2 becomes W2 prediction**, and W1 pred shifts to Lag_1
- **Lag_3 becomes W3 prediction**, W2 pred shifts to Lag_2, W1 pred shifts to Lag_1
- Continue for W4-W8
- Rolling statistics recalculate using predictions
- LP features shift: LP_W2 becomes new LP_W1 for W2 forecast

---

## Modeling

### Learners (modeling/learners.py)

**LightGBM & XGBoost:**
- p50 via MAE objective
- Quantile heads: p75 (opt), p85, p90, p95, p99 via pinball loss
- Separate models for TRR and TRP

**LSTM:**
- Sequence length: 52 weeks
- Features: All ~118 as multivariate input
- p50 head (MAE) + quantile heads (p85/p90/p95/p99, optional p75)
- 2-3 layers, dropout, early stopping

**SARIMAX:**
- Per-leaf model with exogenous LP & calendar
- Point + residual variance → quantiles via normal approx or empirical

### Multi-Horizon Strategies (modeling/multi_horizon.py)

**Recursive:** 1-step model rolled forward W1→W8 (uses prior predictions as features)

**Direct:** 8 separate models, one per horizon

**DirRec:** Direct for W1-W4, Recursive W5-W8 (hybrid)

Each strategy outputs: `{h: DataFrame[entity_id, liquidity_group, week_start, p75?, p85, p90, p95, p99, p50]}`

### Ensemble (modeling/ensemble.py)

Stacked ridge across 4 learners per horizon using out-of-fold predictions. Optional dynamic per-leaf weights.

### Constraints

Post-prediction enforcement:
- TRR ≥ 0 (clip or restore sign)
- TRP ≤ 0 (clip or restore sign)

---

## Cross-Validation (cv.py)

**Rolling-origin:** 5 splits, embargo=1 week, min_history=52 weeks

Per-horizon evaluation (W1-W8):
- WAPE, MAE, RMSE, Directionality, PICP
- Compare vs targets from config
- Output: `artifacts/metrics/cv_results_{timestamp}.parquet`

---

## Metrics (metrics.py)

### Primary: WAPE (Weighted Absolute Percentage Error)
```python
WAPE = sum(|actual - forecast|) / sum(|actual|)
Accuracy = 1 - WAPE
```
Targets: W1: 5%, W2: 7.5%, W3: 10%, W4: 12.5%, W5: 15%, W6: 17.5%, W7: 20%, W8: 22.5%

### Secondary Metrics
- **MAE:** Mean absolute error (€)
- **RMSE:** Root mean squared error
- **Directionality:** % correct sign predictions (target ≥85% for W1-W2)
- **PICP:** Prediction interval coverage probability (target ≥80%)

---

## Reconciliation (reconciliation.py)

### Summing Matrix S

**Bottom (30 series):** 15 entities × 2 liquidity groups (TRR, TRP)

**Aggregates:**
- Entity Net (15): Net_e = TRR_e + TRP_e
- Group TRR (1): Σ_e TRR_e
- Group TRP (1): Σ_e TRP_e
- Group Net (1): Σ_e Net_e

**Total:** 48 series (30 bottom + 18 aggregates)

**S matrix:** (48 × 30)

### MinT/OLS

Uses residual covariance from CV folds to minimally adjust forecasts ensuring sum consistency.

**Per-quantile reconciliation:** Apply separately for p75?, p85, p90, p95, p99, p50

**Enforce monotonicity:** p75 ≤ p85 ≤ p90 ≤ p95 ≤ p99

---

## Monitoring & Alerts (monitor.py)

**Triggers:**
1. Threshold breach: Entity WAPE exceeds target
2. WoW deterioration: WAPE worsens by ≥3pp week-over-week
3. Directionality drop: <85% for W1-W2
4. Coverage failure: PICP <80%
5. Data drift: PSI >0.20

**Fallback Logic:**
If W5-W8 severely underperform (WAPE > target+15pp OR Directionality <70%):
- Publish ML forecasts for W1-W4
- Use LP extrapolation for W5-W8 (clearly labeled)

**Output:** `artifacts/metrics/monitoring_{asof}.json`

---

## Pipeline & CLI (pipeline.py, cli.py)

### Full Pipeline
```bash
python -m cf_forecast.cli run --config configs/config.yml --asof 2025-11-12
```

### Individual Steps
```bash
python -m cf_forecast.cli ingest    --config configs/config.yml --asof 2025-11-12
python -m cf_forecast.cli features  --config configs/config.yml --asof 2025-11-12
python -m cf_forecast.cli train     --config configs/config.yml --asof 2025-11-12
python -m cf_forecast.cli forecast  --config configs/config.yml --asof 2025-11-12
python -m cf_forecast.cli reconcile --config configs/config.yml --asof 2025-11-12
python -m cf_forecast.cli monitor   --config configs/config.yml --asof 2025-11-12
python -m cf_forecast.cli publish   --config configs/config.yml --asof 2025-11-12 --excel true
```

---

## Streamlit UI (app/streamlit_app.py)

### Features

1. **Run Forecast Wizard**
   - Select as-of date (default: this Tuesday)
   - If no data provided by user: use raw data files from repo, recommend appropriate as-of dates
   - Choose backend profile, learners, strategies, quantiles
   - Progress indicators during run

2. **Data Preview**
   - Row counts, date ranges, coverage stats
   - Sign checks (TRR ≥ 0, TRP ≤ 0)
   - Late posting flags

3. **Feature Controls**
   - Enable/disable families
   - Confirm lags 1-52, daily patterns calculated first
   - Cut-off discipline banner

4. **Model Options**
   - Select learners: LGBM, XGB, LSTM, SARIMAX
   - Strategy: Recursive / Direct / DirRec

5. **Metrics Dashboard**
   - Per-horizon WAPE/MAE/RMSE/Directionality/PICP vs targets
   - Green/amber/red status indicators
   - Per-entity heatmaps

6. **Reconciliation Checker**
   - Pre/post reconciliation sums
   - Delta table
   - Quantile monotonicity verification

7. **Downloads**
   - Parquet (partitioned by asof_week)
   - Excel (multi-sheet: Bottom, EntityNet, GroupTRR, GroupTRP, GroupNet, Metrics)

8. **Backtest Mode**
   - Rolling-origin CV over date range
   - Leaderboard comparing learners/strategies

9. **What-If LP (Sandbox)**
   - Adjust treasury LP by ±X%
   - Re-forecast and compare (non-production)

10. **Audit Trail**
    - Config snapshot, data hash, git SHA
    - Run timestamps per step

---

## Testing (pytest)

### Critical Tests

**test_leakage.py** ⚠️
- Recursive uses predictions not actuals for lags
- Features computed as-of Week N use only data ≤ Week N
- LP future (W+1 to W+4) in features, but W+5+ LP excluded

**test_features.py**
- Lags 1-52 respect cut-off
- Daily patterns calculated before weekly aggregation
- Feature count = 118

**test_reconciliation.py**
- S matrix shape correct
- Sums exact after reconciliation
- Quantile monotonicity enforced

**test_cv.py**
- Rolling-origin indices sequential
- Embargo=1 week enforced

**test_metrics.py**
- WAPE/MAE/Directionality/PICP formulas correct

**test_modeling.py**
- All strategies produce 8 horizons
- All quantiles emitted: p85, p90, p95, p99, p50 (optional p75)

---

## Outputs

### Parquet (partitioned by asof_week)
- `bottom_leaf_forecasts.parquet` (30 series)
- `entity_net.parquet` (15 series)
- `group_trr.parquet`, `group_trp.parquet`, `group_net.parquet`
- `metrics_{asof}.parquet`

### Excel (forecasts_{asof}.xlsx)
Sheets: Bottom, EntityNet, GroupTRR, GroupTRP, GroupNet, Metrics

---

## What Claude Code Must Follow (Non-Negotiable)

1. **Feature Engineering:**
   - Exactly 118 features per `Comprehensive_Feature_List_for_Forecasting_Model.docx`
   - Daily pattern features (8) calculated BEFORE weekly aggregation
   - Cut-off discipline: no future data in features

2. **WAPE Targets:**
   - W1: 5%, W2: 7.5%, W3: 10%, W4: 12.5%, W5: 15%, W6: 17.5%, W7: 20%, W8: 22.5%

3. **Recursive Strategy:**
   - W1 prediction replaces Lag_1
   - W2 prediction replaces Lag_2, W1 pred becomes Lag_1
   - W3 prediction replaces Lag_3, W2 pred becomes Lag_2, W1 pred becomes Lag_1
   - Continue for W4-W8

4. **Quantiles:**
   - Every prediction includes: p85, p90, p95, p99, p50 (optional: p75)

5. **Hierarchy:**
   - Bottom: (Entity, TRR), (Entity, TRP) per entity_liquidity_map.csv
   - Reconcile to Entity Net → Group aggregates via MinT/OLS

6. **Testing:**
   - test_leakage.py must pass (no future data)
   - test_features.py confirms 118 features
   - test_reconciliation.py validates sums + monotonicity

---

## Where Claude Code Has Creative Freedom

1. **Architecture Choices:**
   - LSTM layers/units (2-3 layers recommended, but flexible)
   - XGBoost/LightGBM hyperparameters (can use defaults or tune)
   - Ensemble meta-learner beyond ridge (could try ElasticNet, weighted avg)

2. **Implementation Details:**
   - Data loading efficiency (pandas vs polars vs dask)
   - Caching strategies for intermediate outputs
   - Parallel processing for CV folds
   - Logging verbosity and format

3. **UI Enhancements:**
   - Visualization choices (Plotly vs Altair vs Matplotlib)
   - Layout and styling in Streamlit
   - Additional exploratory views (if time permits)

4. **Error Handling:**
   - Retry logic for data backends
   - Graceful degradation strategies
   - User-friendly error messages

5. **Optimizations:**
   - Feature storage format (parquet partitioning scheme)
   - Model serialization approach (joblib vs pickle vs cloudpickle)
   - Memory management for large datasets

**Guideline:** Follow the spec for "what" (features, metrics, targets, hierarchy) but be creative about "how" (implementation efficiency, code organization, user experience).

---

## Installation & Quick Start

```bash
# Clone and install
git clone <repo-url>
cd cashflow-forecasting
pip install -e .

# Configure
cp configs/credentials.example.yml configs/credentials.yml
# Edit credentials.yml with your backend settings

# Run full pipeline
python -m cf_forecast.cli run --config configs/config.yml --asof 2025-11-12

# Launch UI
streamlit run app/streamlit_app.py

# Run tests
pytest tests/ -v --cov=cf_forecast
```

---

## Cloud Deployment (Stubs Provided, Disabled by Default)

**Azure ML:** `azureml/pipeline_stub.py`, `azureml/schedule.json`

**SageMaker:** `sagemaker/pipeline_stub.py`, `sagemaker/eventbridge_cron_stub.json`

Enable via `config.yml` when ready for cloud deployment.

---

## Future Enhancements (Post-V1)

When additional data becomes available:
- Direct connection to data sources (DENODO, REVAL TMS)
- AR/AP schedules integration (invoice due dates)
- Customer behavior analytics (top 20 customers)
- External data feeds (FX rates, steel prices, PMI indices)
- Category-level breakdowns (when treasury provides)

V1 establishes infrastructure. Model accuracy improves as data enriches.

---

**Last Updated:** 2025-11-10  
**Version:** 1.0  
**Project:** Treasury Cash Flow Forecasting, Aperam S.A.
