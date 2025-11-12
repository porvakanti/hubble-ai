"""
Stage 9: Advanced Feature Engineering

Adds advanced features for improved forecasting:
- Statistical: skewness, kurtosis, quantiles, MAD
- Exponential: EWMA, momentum, time decay
- Logarithmic: log returns, log volatility
- Calendar: month-end, quarter-end, Fourier seasonality
- Interaction: lag×rolling, volatility-adjusted
- Normalized: z-score, percentile rank

Prerequisites: Stage 8 output (training_data_final.csv with 152 features)
Output: training_data_enhanced.csv with ~200 features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for entity_mapping
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from cf_forecast.entity_mapping import filter_tier_1_combinations

print("=" * 80)
print("STAGE 9: ADVANCED FEATURE ENGINEERING")
print("=" * 80)
print()

# Load Stage 8 output
print("Loading Stage 8 output...")
training_data = pd.read_csv('../data/intermediate/training_data_final.csv')
training_data['week_start'] = pd.to_datetime(training_data['week_start'])

print(f"  Loaded: {len(training_data):,} rows × {training_data.shape[1]} columns")
print(f"  Date range: {training_data['week_start'].min().date()} to {training_data['week_start'].max().date()}")
print()

# Filter to Tier 1 only (ML forecasting)
print("Filtering to Tier 1 combinations only...")
training_data_tier1 = filter_tier_1_combinations(training_data)
print(f"  Kept: {len(training_data_tier1):,} rows ({len(training_data_tier1)/len(training_data)*100:.1f}%)")
print(f"  Excluded: {len(training_data) - len(training_data_tier1):,} rows (Tier 2)")
print()

# Work with Tier 1 data
df = training_data_tier1.copy()
initial_features = df.shape[1]

print(f"Starting features: {initial_features}")
print()

# ============================================================================
# 1. STATISTICAL FEATURES
# ============================================================================
print("Adding statistical features...")

# Skewness and Kurtosis (distribution shape)
for window in [12, 26]:
    print(f"  Skewness & kurtosis ({window}w)...")
    df[f'amount_skew_{window}w'] = df.groupby(['entity_id', 'liquidity_group'])['amount_eur'].transform(
        lambda x: x.rolling(window).skew()
    )
    df[f'amount_kurt_{window}w'] = df.groupby(['entity_id', 'liquidity_group'])['amount_eur'].transform(
        lambda x: x.rolling(window).apply(lambda y: y.kurtosis() if len(y) >= 4 else np.nan)
    )

# Quantile-based features (robust to outliers)
for window in [12, 52]:
    print(f"  Quantiles ({window}w)...")
    df[f'amount_p25_{window}w'] = df.groupby(['entity_id', 'liquidity_group'])['amount_eur'].transform(
        lambda x: x.rolling(window).quantile(0.25)
    )
    df[f'amount_p75_{window}w'] = df.groupby(['entity_id', 'liquidity_group'])['amount_eur'].transform(
        lambda x: x.rolling(window).quantile(0.75)
    )
    df[f'iqr_{window}w'] = df[f'amount_p75_{window}w'] - df[f'amount_p25_{window}w']

# Median Absolute Deviation (MAD - outlier-robust)
print("  Median absolute deviation...")
for window in [12, 26]:
    df[f'mad_{window}w'] = df.groupby(['entity_id', 'liquidity_group'])['amount_eur'].transform(
        lambda x: (x - x.rolling(window).median()).abs().rolling(window).median()
    )

statistical_features = df.shape[1] - initial_features
print(f"  Added: {statistical_features} statistical features")
print()

# ============================================================================
# 2. EXPONENTIAL FEATURES
# ============================================================================
print("Adding exponential features...")

# Exponentially Weighted Moving Average (EWMA)
for span in [4, 8, 12]:
    print(f"  EWMA (span={span})...")
    df[f'ewma_{span}w'] = df.groupby(['entity_id', 'liquidity_group'])['amount_eur'].transform(
        lambda x: x.ewm(span=span, adjust=False).mean()
    )
    df[f'ewm_std_{span}w'] = df.groupby(['entity_id', 'liquidity_group'])['amount_eur'].transform(
        lambda x: x.ewm(span=span, adjust=False).std()
    )

# Exponential momentum
print("  Exponential momentum...")
df['momentum_exp_4_8'] = df['ewma_4w'] / (df['ewma_8w'] + 1e-6) - 1
df['momentum_exp_8_12'] = df['ewma_8w'] / (df['ewma_12w'] + 1e-6) - 1

exponential_features = df.shape[1] - initial_features - statistical_features
print(f"  Added: {exponential_features} exponential features")
print()

# ============================================================================
# 3. LOGARITHMIC FEATURES
# ============================================================================
print("Adding logarithmic features...")

# Log transformation (handle large magnitude variations)
print("  Log transformation...")
df['amount_log'] = np.sign(df['amount_eur']) * np.log1p(df['amount_eur'].abs())

# Log returns
print("  Log returns...")
df['log_return_1w'] = df.groupby(['entity_id', 'liquidity_group'])['amount_log'].diff()
df['log_return_4w'] = df.groupby(['entity_id', 'liquidity_group'])['amount_log'].diff(4)

# Log volatility
print("  Log volatility...")
for window in [4, 12]:
    df[f'log_volatility_{window}w'] = df.groupby(['entity_id', 'liquidity_group'])['log_return_1w'].transform(
        lambda x: x.rolling(window).std()
    )

logarithmic_features = df.shape[1] - initial_features - statistical_features - exponential_features
print(f"  Added: {logarithmic_features} logarithmic features")
print()

# ============================================================================
# 4. CALENDAR FEATURES
# ============================================================================
print("Adding calendar features...")

# Extract date components
print("  Date components...")
df['day_of_week'] = df['week_start'].dt.dayofweek
df['week_of_month'] = ((df['week_start'].dt.day - 1) // 7 + 1).clip(upper=5)
df['week_of_year'] = df['week_start'].dt.isocalendar().week

# Days to month-end (treasury operations spike)
print("  Days to month/quarter end...")
df['days_to_month_end'] = (df['week_start'] + pd.offsets.MonthEnd(0) - df['week_start']).dt.days

# Is last week of month
df['is_last_week_of_month'] = (df['days_to_month_end'] <= 7).astype(int)

# Is last week of quarter
df['is_last_week_of_quarter'] = (
    (df['week_start'].dt.month.isin([3, 6, 9, 12])) &
    (df['days_to_month_end'] <= 7)
).astype(int)

# Is last week of year
df['is_last_week_of_year'] = (
    (df['week_start'].dt.month == 12) &
    (df['days_to_month_end'] <= 14)
).astype(int)

# Fourier features for seasonality
print("  Fourier seasonality components...")
# Annual cycle (52 weeks)
df['fourier_52w_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
df['fourier_52w_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

# Semi-annual cycle (26 weeks)
df['fourier_26w_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 26)
df['fourier_26w_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 26)

# Quarterly cycle (13 weeks)
df['fourier_13w_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 13)
df['fourier_13w_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 13)

calendar_features = df.shape[1] - initial_features - statistical_features - exponential_features - logarithmic_features
print(f"  Added: {calendar_features} calendar features")
print()

# ============================================================================
# 5. INTERACTION FEATURES
# ============================================================================
print("Adding interaction features...")

# Lag × Rolling average (trend-lag interaction)
print("  Lag × Rolling interactions...")
if 'lag_1' in df.columns and 'rolling_4w_mean' in df.columns:
    df['lag1_x_ma4'] = df['lag_1'] * df['rolling_4w_mean']
if 'lag_4' in df.columns and 'rolling_4w_mean' in df.columns:
    df['lag4_x_ma4'] = df['lag_4'] * df['rolling_4w_mean']

# Volatility-adjusted amount
print("  Volatility-adjusted...")
if 'rolling_4w_std' in df.columns:
    df['vol_adjusted_amount'] = df['amount_eur'] / (df['rolling_4w_std'] + 1)

# Trend acceleration (recent vs historical)
print("  Trend acceleration...")
if all(col in df.columns for col in ['rolling_4w_mean', 'rolling_12w_mean', 'rolling_52w_mean']):
    df['trend_acceleration'] = (
        (df['rolling_4w_mean'] - df['rolling_12w_mean']) /
        (df['rolling_52w_mean'].abs() + 1)
    )

# LP × Actual interactions (if LP features exist)
print("  LP × Actual interactions...")
if 'W1_Forecast' in df.columns:
    df['actual_to_lp_ratio_sqr'] = (df['amount_eur'] / (df['W1_Forecast'].abs() + 1)) ** 2

interaction_features = df.shape[1] - initial_features - statistical_features - exponential_features - logarithmic_features - calendar_features
print(f"  Added: {interaction_features} interaction features")
print()

# ============================================================================
# 6. NORMALIZED FEATURES
# ============================================================================
print("Adding normalized features...")

# Z-score normalization
print("  Z-score...")
for window in [12, 52]:
    if f'rolling_{window}w_mean' in df.columns and f'rolling_{window}w_std' in df.columns:
        df[f'amount_zscore_{window}w'] = (
            (df['amount_eur'] - df[f'rolling_{window}w_mean']) /
            (df[f'rolling_{window}w_std'] + 1)
        )

# Percentile rank (where are we in historical distribution?)
print("  Percentile rank...")
for window in [26, 52]:
    df[f'amount_percentile_{window}w'] = df.groupby(['entity_id', 'liquidity_group'])['amount_eur'].transform(
        lambda x: x.rolling(window).rank(pct=True)
    )

# Distance from typical (relative to IQR)
print("  Distance from typical...")
if 'iqr_52w' in df.columns and 'rolling_52w_mean' in df.columns:
    df['distance_from_typical'] = (
        (df['amount_eur'] - df['rolling_52w_mean']).abs() /
        (df['iqr_52w'] + 1)
    )

normalized_features = df.shape[1] - initial_features - statistical_features - exponential_features - logarithmic_features - calendar_features - interaction_features
print(f"  Added: {normalized_features} normalized features")
print()

# ============================================================================
# FINALIZE
# ============================================================================

# Fill NaN with 0 for all new features
print("Filling NaN values...")
feature_cols = [col for col in df.columns if col not in ['entity_id', 'liquidity_group', 'week_start', 'amount_eur']]

# Count NaN before
nan_before = df[feature_cols].isna().sum().sum()
print(f"  NaN values before: {nan_before:,}")

df[feature_cols] = df[feature_cols].fillna(0)

# Count NaN after
nan_after = df[feature_cols].isna().sum().sum()
print(f"  NaN values after: {nan_after:,}")
print()

# Summary
final_features = df.shape[1]
new_features = final_features - initial_features

print("=" * 80)
print("FEATURE SUMMARY")
print("=" * 80)
print(f"Initial features: {initial_features}")
print(f"  Statistical: +{statistical_features}")
print(f"  Exponential: +{exponential_features}")
print(f"  Logarithmic: +{logarithmic_features}")
print(f"  Calendar: +{calendar_features}")
print(f"  Interaction: +{interaction_features}")
print(f"  Normalized: +{normalized_features}")
print(f"Total new features: +{new_features}")
print(f"Final features: {final_features}")
print()

# Save enhanced training data
output_path = '../data/intermediate/training_data_enhanced.csv'
print(f"Saving enhanced training data to: {output_path}")
df.to_csv(output_path, index=False)
print(f"  Saved: {len(df):,} rows × {df.shape[1]} columns")
print()

# Validation
print("=" * 80)
print("VALIDATION")
print("=" * 80)
print(f"Rows: {len(df):,}")
print(f"Columns: {df.shape[1]}")
print(f"Date range: {df['week_start'].min().date()} to {df['week_start'].max().date()}")
print(f"Entities: {df['entity_id'].nunique()}")
print(f"Liquidity groups: {df['liquidity_group'].nunique()}")
print(f"Entity-liq combinations: {df.groupby(['entity_id', 'liquidity_group']).ngroups}")
print()

# Check dtypes
print("Data types:")
print(f"  Numeric: {df.select_dtypes(include=[np.number]).shape[1]}")
print(f"  Object: {df.select_dtypes(include=['object']).shape[1]}")
print(f"  Datetime: {df.select_dtypes(include=['datetime64']).shape[1]}")
print()

print("=" * 80)
print("STAGE 9 COMPLETE")
print("=" * 80)
print(f"✓ Enhanced training data saved: {output_path}")
print(f"✓ Ready for backtesting with {final_features} features")
