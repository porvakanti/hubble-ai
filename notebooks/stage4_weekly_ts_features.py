# Stage 4: Feature Engineering Part 2 - Weekly Time-Series Features
# To be added after Stage 3 in the EDA workflow

"""
Stage 4: Feature Engineering - Weekly Time-Series Features

Creates ~40 time-series features on weekly aggregated data:
- Lag features (lag_1 through lag_8, lag_52)
- Rolling window statistics (4w, 8w, 12w, 52w)
- Week-over-week changes and trends
- Volatility and stability metrics
- Seasonal indices

Input: weekly_actuals_featured.csv (from Stage 3)
Output: weekly_actuals_ts_features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("\n" + "="*80)
print("STAGE 4: FEATURE ENGINEERING - WEEKLY TIME-SERIES FEATURES")
print("="*80)

# Load weekly actuals from Stage 3
weekly_actuals = pd.read_csv('../data/intermediate/weekly_actuals_featured.csv')
weekly_actuals['week_start'] = pd.to_datetime(weekly_actuals['week_start'])

print(f"\nLoaded weekly actuals:")
print(f"  Rows: {len(weekly_actuals):,}")
print(f"  Columns: {len(weekly_actuals.columns)}")
print(f"  Date range: {weekly_actuals['week_start'].min().date()} to {weekly_actuals['week_start'].max().date()}")

# Sort by entity, liquidity group, week
weekly_actuals = weekly_actuals.sort_values(['entity_id', 'liquidity_group', 'week_start'])

# =============================================================================
# 4.1: LAG FEATURES
# =============================================================================

print("\n" + "="*80)
print("4.1: LAG FEATURES")
print("="*80)

# Group by entity × liquidity group
grouped = weekly_actuals.groupby(['entity_id', 'liquidity_group'])

# Create lags 1-52 (full year of historical lags)
lag_periods = list(range(1, 53))  # lag_1 through lag_52

print(f"\nCreating lag features for {len(lag_periods)} periods (lag_1 to lag_52)...")

for lag in lag_periods:
    if lag % 10 == 0:
        print(f"  lag_{lag}...")
    weekly_actuals[f'lag_{lag}'] = grouped['amount_eur'].shift(lag)

lag_features_created = len(lag_periods)  # 52 lags
print(f"\n✓ Created {lag_features_created} lag features (lag_1 to lag_52)")

# =============================================================================
# 4.2: ROLLING WINDOW FEATURES
# =============================================================================

print("\n" + "="*80)
print("4.2: ROLLING WINDOW FEATURES")
print("="*80)

# Rolling windows: 4-week, 8-week, 12-week, 52-week
rolling_windows = [4, 8, 12, 52]

print(f"\nCreating rolling window features for {len(rolling_windows)} windows...")

for window in rolling_windows:
    print(f"  {window}-week window...")

    # Rolling mean
    weekly_actuals[f'rolling_{window}w_mean'] = grouped['amount_eur'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    # Rolling std
    weekly_actuals[f'rolling_{window}w_std'] = grouped['amount_eur'].transform(
        lambda x: x.rolling(window, min_periods=1).std()
    )

    # Rolling min
    weekly_actuals[f'rolling_{window}w_min'] = grouped['amount_eur'].transform(
        lambda x: x.rolling(window, min_periods=1).min()
    )

    # Rolling max
    weekly_actuals[f'rolling_{window}w_max'] = grouped['amount_eur'].transform(
        lambda x: x.rolling(window, min_periods=1).max()
    )

    # Rolling sum
    weekly_actuals[f'rolling_{window}w_sum'] = grouped['amount_eur'].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )

rolling_features_created = len(rolling_windows) * 5  # mean, std, min, max, sum
print(f"\n✓ Created {rolling_features_created} rolling window features")

# =============================================================================
# 4.3: TREND FEATURES
# =============================================================================

print("\n" + "="*80)
print("4.3: TREND FEATURES")
print("="*80)

# Week-over-week change
weekly_actuals['wow_change'] = grouped['amount_eur'].diff()
weekly_actuals['wow_pct_change'] = grouped['amount_eur'].pct_change()

# 4-week trend (linear regression slope)
def compute_trend(series):
    """Compute linear trend over window"""
    if len(series) < 2:
        return np.nan
    x = np.arange(len(series))
    y = series.values
    if np.all(np.isnan(y)) or np.std(y) == 0:
        return 0
    # Simple slope calculation
    slope = np.polyfit(x, y, 1)[0]
    return slope

weekly_actuals['trend_4w'] = grouped['amount_eur'].transform(
    lambda x: x.rolling(4, min_periods=2).apply(compute_trend, raw=False)
)

weekly_actuals['trend_8w'] = grouped['amount_eur'].transform(
    lambda x: x.rolling(8, min_periods=2).apply(compute_trend, raw=False)
)

weekly_actuals['trend_12w'] = grouped['amount_eur'].transform(
    lambda x: x.rolling(12, min_periods=2).apply(compute_trend, raw=False)
)

# Trend direction (positive/negative/stable)
weekly_actuals['trend_direction_4w'] = np.sign(weekly_actuals['trend_4w'])
weekly_actuals['trend_direction_8w'] = np.sign(weekly_actuals['trend_8w'])

trend_features_created = 7
print(f"\n✓ Created {trend_features_created} trend features")
print(f"  - wow_change, wow_pct_change")
print(f"  - trend_4w, trend_8w, trend_12w (linear slopes)")
print(f"  - trend_direction_4w, trend_direction_8w")

# =============================================================================
# 4.4: VOLATILITY & STABILITY FEATURES
# =============================================================================

print("\n" + "="*80)
print("4.4: VOLATILITY & STABILITY FEATURES")
print("="*80)

# Coefficient of variation (CV) = std / mean
# Lower CV = more stable
for window in [4, 8, 12]:
    mean_col = f'rolling_{window}w_mean'
    std_col = f'rolling_{window}w_std'

    weekly_actuals[f'cv_{window}w'] = (
        weekly_actuals[std_col] / weekly_actuals[mean_col].abs()
    ).replace([np.inf, -np.inf], np.nan)

# Stability score (inverse of CV, normalized 0-1)
weekly_actuals['stability_4w'] = 1 / (1 + weekly_actuals['cv_4w'])
weekly_actuals['stability_8w'] = 1 / (1 + weekly_actuals['cv_8w'])

# Range (max - min) as volatility measure
for window in [4, 8]:
    min_col = f'rolling_{window}w_min'
    max_col = f'rolling_{window}w_max'
    weekly_actuals[f'range_{window}w'] = (
        weekly_actuals[max_col] - weekly_actuals[min_col]
    )

volatility_features_created = 8
print(f"\n✓ Created {volatility_features_created} volatility/stability features")
print(f"  - cv_4w, cv_8w, cv_12w (coefficient of variation)")
print(f"  - stability_4w, stability_8w")
print(f"  - range_4w, range_8w")

# =============================================================================
# 4.5: RATIO & INTERACTION FEATURES
# =============================================================================

print("\n" + "="*80)
print("4.5: RATIO & INTERACTION FEATURES")
print("="*80)

# Current vs historical average ratios
weekly_actuals['amount_vs_4w_avg'] = (
    weekly_actuals['amount_eur'] / weekly_actuals['rolling_4w_mean']
)

weekly_actuals['amount_vs_8w_avg'] = (
    weekly_actuals['amount_eur'] / weekly_actuals['rolling_8w_mean']
)

weekly_actuals['amount_vs_52w_avg'] = (
    weekly_actuals['amount_eur'] / weekly_actuals['rolling_52w_mean']
)

# Transaction count vs amount (average transaction size)
weekly_actuals['avg_txn_size'] = (
    weekly_actuals['amount_eur'] / weekly_actuals['txn_count']
).replace([np.inf, -np.inf], np.nan)

# Deviation from rolling mean (z-score like)
weekly_actuals['deviation_from_4w_mean'] = (
    (weekly_actuals['amount_eur'] - weekly_actuals['rolling_4w_mean']) /
    weekly_actuals['rolling_4w_std']
).replace([np.inf, -np.inf], np.nan)

weekly_actuals['deviation_from_8w_mean'] = (
    (weekly_actuals['amount_eur'] - weekly_actuals['rolling_8w_mean']) /
    weekly_actuals['rolling_8w_std']
).replace([np.inf, -np.inf], np.nan)

ratio_features_created = 6
print(f"\n✓ Created {ratio_features_created} ratio/interaction features")
print(f"  - amount_vs_4w/8w/52w_avg (current vs historical)")
print(f"  - avg_txn_size")
print(f"  - deviation_from_4w/8w_mean (standardized)")

# =============================================================================
# 4.6: SEASONAL INDICES
# =============================================================================

print("\n" + "="*80)
print("4.6: SEASONAL INDICES")
print("="*80)

# Compute seasonal index by week_number (1-52)
# Seasonal index = average for that week / overall average

seasonal_indices = weekly_actuals.groupby(['entity_id', 'liquidity_group', 'week_number']).agg({
    'amount_eur': 'mean'
}).reset_index()

seasonal_indices.columns = ['entity_id', 'liquidity_group', 'week_number', 'week_avg']

# Calculate overall average per entity × liquidity group
overall_avg = weekly_actuals.groupby(['entity_id', 'liquidity_group'])['amount_eur'].mean().reset_index()
overall_avg.columns = ['entity_id', 'liquidity_group', 'overall_avg']

# Merge and compute seasonal index
seasonal_indices = seasonal_indices.merge(overall_avg, on=['entity_id', 'liquidity_group'])
seasonal_indices['seasonal_index'] = seasonal_indices['week_avg'] / seasonal_indices['overall_avg']

# Merge back to main dataset
weekly_actuals = weekly_actuals.merge(
    seasonal_indices[['entity_id', 'liquidity_group', 'week_number', 'seasonal_index']],
    on=['entity_id', 'liquidity_group', 'week_number'],
    how='left'
)

seasonal_features_created = 1
print(f"\n✓ Created {seasonal_features_created} seasonal feature")
print(f"  - seasonal_index (week-specific pattern)")

# =============================================================================
# 4.7: EXPONENTIAL WEIGHTED MOVING AVERAGE (EWMA)
# =============================================================================

print("\n" + "="*80)
print("4.7: EXPONENTIAL WEIGHTED FEATURES")
print("="*80)

# EWMA gives more weight to recent observations
# Span of 4 weeks (alpha ≈ 0.4)
weekly_actuals['ewma_4w'] = grouped['amount_eur'].transform(
    lambda x: x.ewm(span=4, adjust=False).mean()
)

# Span of 8 weeks (alpha ≈ 0.22)
weekly_actuals['ewma_8w'] = grouped['amount_eur'].transform(
    lambda x: x.ewm(span=8, adjust=False).mean()
)

ewma_features_created = 2
print(f"\n✓ Created {ewma_features_created} EWMA features")
print(f"  - ewma_4w, ewma_8w (exponentially weighted moving average)")

# =============================================================================
# 4.8: STAGE 4 SUMMARY
# =============================================================================

print("\n" + "="*80)
print("STAGE 4 SUMMARY")
print("="*80)

stage4_features = (
    lag_features_created +
    rolling_features_created +
    trend_features_created +
    volatility_features_created +
    ratio_features_created +
    seasonal_features_created +
    ewma_features_created
)

print(f"\n✓ WEEKLY TIME-SERIES FEATURES CREATED: {stage4_features}")
print(f"  - Lag features: {lag_features_created} (lag_1 through lag_52)")
print(f"  - Rolling windows: {rolling_features_created}")
print(f"  - Trend features: {trend_features_created}")
print(f"  - Volatility/Stability: {volatility_features_created}")
print(f"  - Ratio/Interaction: {ratio_features_created}")
print(f"  - Seasonal: {seasonal_features_created}")
print(f"  - EWMA: {ewma_features_created}")

# Total feature count so far
stage2_features = 38  # From Stage 2
stage3_features = 12  # From Stage 3 aggregation
total_features = stage2_features + stage3_features + stage4_features

print(f"\n✓ CUMULATIVE FEATURE COUNT:")
print(f"  - Stage 2 (Daily): {stage2_features}")
print(f"  - Stage 3 (Aggregation): {stage3_features}")
print(f"  - Stage 4 (Time-Series): {stage4_features} (includes 52 lags)")
print(f"  - Total so far: {total_features}")
print(f"  - Target: ~118")
print(f"  - Remaining for Stages 5-7: ~{max(0, 118 - total_features)} (LP features + cross-features)")

# Data quality check
print(f"\n✓ FINAL DATASET DIMENSIONS:")
print(f"  - Rows: {len(weekly_actuals):,}")
print(f"  - Columns: {len(weekly_actuals.columns)}")
print(f"  - Entities: {weekly_actuals['entity_id'].nunique()}")
print(f"  - Weeks: {weekly_actuals['week_start'].nunique()}")
print(f"  - Date range: {weekly_actuals['week_start'].min().date()} to {weekly_actuals['week_start'].max().date()}")

# Check for NaN in key features
print(f"\n✓ DATA QUALITY CHECK:")
key_features = ['amount_eur', 'lag_1', 'rolling_4w_mean', 'trend_4w', 'seasonal_index']
for feat in key_features:
    null_pct = weekly_actuals[feat].isnull().sum() / len(weekly_actuals) * 100
    print(f"  - {feat}: {null_pct:.1f}% null")

# Save intermediate output
weekly_actuals.to_csv('../data/intermediate/weekly_actuals_ts_features.csv', index=False)
print(f"\n✓ Saved to: data/intermediate/weekly_actuals_ts_features.csv")

print("\n" + "="*80)
print("✓ STAGE 4 COMPLETE")
print("="*80)
print("\nNext: Stage 5 - LP Processing & Pivoting")
