# TRAINING DATA SPECIFICATION
## Hubble.AI Treasury Cash-Flow Forecasting Platform

**File:** `data/intermediate/training_data_enhanced.csv`
**Created:** Stage 9 (Advanced Feature Engineering)
**Last Updated:** November 12, 2025

---

## 1. Dataset Overview

| Attribute | Value |
|-----------|-------|
| **Total Rows** | 4,646 |
| **Total Columns** | 204 |
| **Memory Usage** | 7.77 MB |
| **Missing Values** | 0 (0.00%) |
| **Date Range** | January 3, 2022 - September 29, 2025 |
| **Total Weeks** | 196 weeks (~3.8 years) |
| **Scope** | Tier 1 combinations only (20 combos, 96.97% coverage) |

---

## 2. Entity-Liquidity Combinations

**Total Entities:** 16
**Liquidity Groups:** 2 (TRR, TRP)
**Entity-Liq Combinations:** 24

### Coverage by Combination

| Rank | Entity-Liq | Weeks | Coverage | Notes |
|------|------------|-------|----------|-------|
| 1-19 | 19 combinations | 196 weeks | Full | Complete timeseries |
| 20 | 17C7-TRP | 195 weeks | 99.5% | 1 week missing |
| 21 | V756-TRP | 193 weeks | 98.5% | 3 weeks missing |
| 22 | 4B9-TRP | 190 weeks | 96.9% | 6 weeks missing |
| 23 | 11G5-TRP | 184 weeks | 93.9% | 12 weeks missing |
| 24 | 057-TRP | 160 weeks | 81.6% | 36 weeks missing |

**Note:** All 24 combinations are Tier 1 (included in forecasting). 14 Tier 2 combinations were excluded during data preparation.

---

## 3. Column Breakdown

### 3.1 Metadata Columns (4)

| Column | Type | Description |
|--------|------|-------------|
| `entity_id` | object | Entity identifier (e.g., V798, 97C) |
| `liquidity_group` | object | TRR (Treasury Receipts) or TRP (Treasury Payments) |
| `week_start` | datetime64[ns] | Week starting date (Monday) |
| `amount_eur` | float64 | **Target variable** - weekly cash flow amount in EUR |

### 3.2 Feature Columns (200)

**Total Features:** 200 (152 base + 48 advanced from Stage 9)

---

## 4. Feature Categories

### 4.1 Lag Features (52)

Time-lagged values of target variable for capturing temporal dependencies.

**Examples:**
- `lag_1` to `lag_52`: Previous 1 to 52 weeks

**Purpose:** Capture autocorrelation and seasonality

---

### 4.2 Rolling Window Features (48)

Aggregations over various time windows.

**Windows:** 7d, 14d, 30d, 4w, 8w, 12w, 26w, 52w

**Aggregations:**
- `rolling_*_sum_last`: Sum over window
- `rolling_*_mean_last`: Mean over window
- `rolling_*_std_last`: Standard deviation over window
- `rolling_*_min_last`: Minimum value
- `rolling_*_max_last`: Maximum value

**Examples:**
- `rolling_7d_mean_last`: 7-day rolling mean
- `rolling_4w_mean_x_lp_w1`: 4-week mean × liquidity plan W1
- `rolling_12w_std_last`: 12-week rolling std dev

**Purpose:** Capture trend and volatility patterns

---

### 4.3 Calendar Features (22)

Date-based features for seasonality and business cycles.

**Month/Quarter Features:**
- `month` (1-12)
- `quarter` (1-4)
- `week_of_month` (1-5)
- `week_of_year` (1-52)
- `day_of_week` (0-6)

**Binary Flags:**
- `is_month_start_max`, `is_month_end_max`
- `is_quarter_start_max`, `is_quarter_end_max`
- `is_year_start_max`, `is_year_end_max`
- `is_weekend_sum`
- `is_last_week_of_month`
- `is_last_week_of_quarter`
- `is_last_week_of_year`

**Continuous:**
- `days_to_month_end`

**Fourier Seasonality (Stage 9):**
- `fourier_52w_sin`, `fourier_52w_cos`: Annual cycle
- `fourier_26w_sin`, `fourier_26w_cos`: Semi-annual cycle
- `fourier_13w_sin`, `fourier_13w_cos`: Quarterly cycle

**Purpose:** Capture seasonal patterns, fiscal periods, month-end effects

---

### 4.4 Statistical Features (6) - Stage 9 Advanced

Higher-order statistical moments for distribution characterization.

**Skewness (asymmetry):**
- `amount_skew_12w`: 12-week skewness
- `amount_skew_26w`: 26-week skewness

**Kurtosis (tail heaviness):**
- `amount_kurt_12w`: 12-week kurtosis
- `amount_kurt_26w`: 26-week kurtosis

**Mean Absolute Deviation:**
- `mad_12w`: 12-week MAD
- `mad_26w`: 26-week MAD

**Purpose:** Capture distribution shape and outlier patterns

---

### 4.5 Exponential Features (5) - Stage 9 Advanced

Exponentially weighted metrics for recency bias.

**Exponentially Weighted Moving Average:**
- `ewma_4w`: 4-week EWMA
- `ewma_8w`: 8-week EWMA
- `ewma_12w`: 12-week EWMA

**Momentum:**
- `momentum_exp_4_8`: EWMA_4w - EWMA_8w
- `momentum_exp_8_12`: EWMA_8w - EWMA_12w

**Purpose:** Give more weight to recent observations, capture momentum

---

### 4.6 Logarithmic Features (4) - Stage 9 Advanced

Log-transformed metrics for handling scale and volatility.

**Log Returns:**
- `log_return_1w`: Log return over 1 week
- `log_return_4w`: Log return over 4 weeks

**Log Volatility:**
- `log_volatility_4w`: 4-week log volatility
- `log_volatility_12w`: 12-week log volatility

**Purpose:** Normalize large values, measure relative changes

---

### 4.7 Normalized Features (4) - Stage 9 Advanced

Standardized metrics for cross-entity comparability.

**Z-Score:**
- `amount_zscore_12w`: (value - mean_12w) / std_12w
- `amount_zscore_52w`: (value - mean_52w) / std_52w

**Percentile Rank:**
- `amount_percentile_26w`: Percentile within 26-week window
- `amount_percentile_52w`: Percentile within 52-week window

**Purpose:** Normalize across different scales, identify relative position

---

### 4.8 Ratio Features (3)

Derived ratios for relative metrics.

**Examples:**
- Ratio of current to historical averages
- Ratio of actual to liquidity plan

**Purpose:** Capture deviations from norms

---

### 4.9 Growth Features (1)

Period-over-period growth metrics.

**Examples:**
- `pct_change_*`: Percentage change over periods

**Purpose:** Capture acceleration/deceleration

---

## 5. Target Variable Statistics (amount_eur)

| Statistic | Value |
|-----------|-------|
| **Mean** | €1,221,399.87 |
| **Median** | €-13,768.72 |
| **Std Dev** | €13,931,452.28 |
| **Min** | €-124,940,978.40 |
| **Max** | €+95,792,411.28 |
| **25th Percentile** | €-1,046,934.26 |
| **75th Percentile** | €+2,680,481.43 |

### Sign Distribution

| Sign | Count | Percentage |
|------|-------|------------|
| **Positive** | 2,156 | 46.4% |
| **Negative** | 2,490 | 53.6% |
| **Zero** | 0 | 0.0% |

**Key Insight:**
- Median is negative (€-13,768.72) but mean is positive (€1,221,399.87), indicating **right-skewed distribution** with large positive outliers
- Slightly more negative transactions (53.6%) vs positive (46.4%)
- High standard deviation (€13.9M) indicates significant volatility

---

## 6. Data Types Distribution

| Data Type | Count |
|-----------|-------|
| **float64** | 181 (88.7%) |
| **int64** | 14 (6.9%) |
| **bool** | 6 (2.9%) |
| **object** | 2 (1.0%) |
| **datetime64** | 1 (0.5%) |

**Total:** 204 columns

---

## 7. Sample Data

### Example: V798-TRR (Last 5 Weeks)

| Week Start | Amount (EUR) | Lag 1 | Lag 2 | Lag 4 |
|------------|--------------|-------|-------|-------|
| 2025-09-01 | €12,699,459 | €12,537,204 | €13,491,781 | €14,905,565 |
| 2025-09-08 | €14,128,704 | €12,699,459 | €12,537,204 | €13,130,898 |
| 2025-09-15 | €11,380,863 | €14,128,704 | €12,699,459 | €13,491,781 |
| 2025-09-22 | €13,330,970 | €11,380,863 | €14,128,704 | €12,537,204 |
| 2025-09-29 | €6,512,448 | €13,330,970 | €11,380,863 | €12,699,459 |

**Pattern:** V798-TRR shows consistent weekly cash flows around €12-14M with occasional spikes

---

## 8. Data Quality

### 8.1 Completeness

| Metric | Value |
|--------|-------|
| **Total Cells** | 947,784 |
| **Missing Cells** | 0 (0.00%) |
| **Completeness** | 100% |

**✅ Perfect Data Quality:** No missing values after feature engineering and forward-fill imputation.

### 8.2 Known Issues

1. **Near-Zero Actuals:** Some rows have very small amounts (€0.01), likely placeholders for missing data
   - **Impact:** Causes extreme WAPE outliers
   - **Mitigation:** Filtered during evaluation (|actual| ≥ €1,000)

2. **Partial Coverage for Some Combinations:**
   - 057-TRP: Only 160/196 weeks (81.6%)
   - 11G5-TRP: Only 184/196 weeks (93.9%)
   - **Impact:** Less training data for these combinations
   - **Mitigation:** Excluded from Tier 1 or flagged as high uncertainty

3. **High Volatility:**
   - Std Dev (€13.9M) > Mean (€1.2M)
   - Some entities show extreme week-to-week swings
   - **Impact:** Harder to forecast consistently
   - **Mitigation:** Advanced features (volatility, MAD, log transforms)

---

## 9. Feature Engineering Pipeline

### Stage 1-8: Base Features (152)
- Transaction aggregations (mean, std, sum, count, min, max)
- Lag features (1-52 weeks)
- Rolling windows (4w, 8w, 12w, 26w, 52w)
- Calendar features (month, quarter, week indicators)
- Liquidity plan integration
- Growth and ratio metrics

### Stage 9: Advanced Features (+48)
- **Statistical:** Skewness, kurtosis, MAD, quantiles
- **Exponential:** EWMA (4w, 8w, 12w), momentum
- **Logarithmic:** Log returns, log volatility
- **Calendar Enhancement:** Fourier seasonality (annual, semi-annual, quarterly)
- **Normalized:** Z-scores, percentile ranks

**Total:** 200 features

---

## 10. Usage in Backtesting

### Walk-Forward Validation

**Training Window:** Expanding
- Week 1 validation: Train on all data before Week 1
- Week 13 validation: Train on all data before Week 13 (includes 12 additional weeks)

**Validation Period (3-month):**
- **Start:** July 7, 2025
- **End:** September 29, 2025
- **Weeks:** 13
- **Training Size at Start:** ~150 weeks per entity
- **Training Size at End:** ~162 weeks per entity

**Entity-Specific Models:**
- Each entity-liquidity combination gets separate model
- 24 combinations × 13 weeks = **312 model training sessions**
- Per model: 8 horizon-specific models (Direct) or 1 recursive model

---

## 11. Tier Classification

### Tier 1 (Forecasting - ML Models)
**Count:** 20 combinations (24 in data, 4 flagged as high uncertainty)
**Coverage:** 96.97% of portfolio
**Scope:** All 24 combinations in this dataset
**Horizon:** W1-W8 (8-week forecasts)

### High Uncertainty Flagged (within Tier 1)
**Count:** 4 combinations
**List:** 4B9-TRP, V756-TRP, 057-TRP, V508-TRP
**Coverage:** 0.86% of portfolio
**WAPE:** 40-60% (poor performance)
**Recommendation:** Consider Tier 2 escalation

### Tier 2 (LP Passthrough - Not in this dataset)
**Count:** 14 combinations (excluded during Stage 8)
**Coverage:** 3.03% of portfolio
**Horizon:** W1-W4 (4-week LP passthrough)
**Reason:** Insufficient data, poor historical WAPE, or business logic

---

## 12. File Format

**Format:** CSV (Comma-Separated Values)
**Encoding:** UTF-8
**Delimiter:** `,`
**Line Terminator:** `\n`
**Header:** Yes (first row)
**File Size:** ~7.77 MB

### Loading Instructions

```python
import pandas as pd

# Load data
df = pd.read_csv('data/intermediate/training_data_enhanced.csv')

# Convert week_start to datetime
df['week_start'] = pd.to_datetime(df['week_start'])

# Convert features to numeric (important for modeling)
feature_cols = [col for col in df.columns
                if col not in ['entity_id', 'liquidity_group', 'week_start', 'amount_eur']]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[feature_cols] = df[feature_cols].fillna(0)
```

---

## 13. Version History

| Version | Date | Changes | Features |
|---------|------|---------|----------|
| v1.0 | Nov 2025 | Stage 1-8 base features | 152 |
| v2.0 | Nov 2025 | Stage 9 advanced features | 200 |
| v2.1 | Nov 2025 | Tier 1 filtering, 3-month validation | 200 |

**Current Version:** v2.1

---

## 14. Related Files

### Input Files
- `data/raw/transactions_2022_2025.csv` (original transaction data)
- `data/raw/liquidity_plan_2025.csv` (liquidity plan forecasts)
- `data/raw/entity_liquidity_reference_map.csv` (entity metadata)

### Intermediate Files
- `data/intermediate/training_data_base.csv` (Stages 1-8, 152 features)
- `data/intermediate/training_data_enhanced.csv` (Stage 9, 200 features) ⭐ **This file**

### Output Files
- `artifacts/backtesting/3month/lightgbm_direct_3month_*.csv` (backtest results)
- `artifacts/backtesting/3month/FINAL_MODEL_COMPARISON_REPORT.md` (model comparison)

---

## 15. Key Takeaways

✅ **Strengths:**
- 100% data completeness (no missing values)
- 3.8 years of historical data (196 weeks)
- Rich feature set (200 features across 9 categories)
- Entity-specific modeling (24 combinations)
- Advanced features for pattern capture

⚠️ **Limitations:**
- 5 combinations have incomplete coverage (<196 weeks)
- High volatility (std dev > mean)
- Near-zero actuals in some records
- July-Sept period appears harder to forecast (+0.76 pp WAPE vs full period)

📊 **Best Use:**
- Primary training dataset for ML forecasting models
- Walk-forward validation with expanding window
- Entity-specific LightGBM or XGBoost models
- Direct strategy (separate model per horizon)

---

**Document Version:** 1.0
**Last Updated:** November 12, 2025
**Author:** Claude (AI Assistant)
**Session ID:** claude/hubble-ai-scaffold-implementation-011CUzgHryjwTB3Y5ZH6yd43
