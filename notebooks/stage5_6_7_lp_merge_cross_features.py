# Stages 5-7: LP Processing, Merge, and Cross-Features
# Final feature engineering stages to reach ~118 features

"""
Stage 5: LP Processing & Pivoting
Stage 6: Merge Actuals + LP
Stage 7: Cross-Features (Actuals × LP interactions)

Completes feature engineering pipeline with LP forecasts as input features
and cross-feature generation for final training dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# STAGE 5: LP PROCESSING & PIVOTING
# =============================================================================

print("\n" + "="*80)
print("STAGE 5: LP PROCESSING & PIVOTING")
print("="*80)

# Load raw LP data
lp_raw = pd.read_csv('../data/raw/LP_17C7.csv')

print(f"\nRaw LP data loaded:")
print(f"  Rows: {len(lp_raw):,}")
print(f"  Columns: {len(lp_raw.columns)}")

# -----------------------------------------------------------------------------
# 5.1: Standardize LP Column Names
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("5.1: STANDARDIZING LP COLUMNS")
print("="*80)

# Expected columns from EDA:
# Entity, Entity Name, Year Title, Liquidity Group/Super Liquidity Group, Item's Date, Amount, etc.

lp_filtered = lp_raw[[
    "Entity",
    "Entity Name",
    "Liquidity Group/Super Liquidity Group",
    "Year Title",
    "Item's Date",
    "Amount",
    "Currency",
    "Plan Currency",
    "Amount in plan currency",
    "Rate"
]].copy()

lp_filtered = lp_filtered.rename(columns={
    "Liquidity Group/Super Liquidity Group": "Liquidity_Group"
})

print(f"\n✓ Standardized LP columns")
print(f"  Rows: {len(lp_filtered):,}")

# -----------------------------------------------------------------------------
# 5.2: FX Conversion (if needed)
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("5.2: FX CONVERSION")
print("="*80)

# Load FX rates
fx = pd.read_csv('../data/raw/eurofxref-hist.csv')
fx['Date'] = pd.to_datetime(fx['Date'])
fx_lookup = fx.set_index('Date')[['USD', 'CHF']].to_dict('index')

def convert_to_eur(row):
    """Convert to EUR using FX rates"""
    if pd.isna(row['Plan Currency']) or row['Plan Currency'] == 'EUR':
        return row['Amount in plan currency']

    item_date = pd.to_datetime(row["Item's Date"])
    currency = row['Plan Currency']

    # Find closest FX rate
    while item_date not in fx_lookup and item_date >= fx['Date'].min():
        item_date -= pd.Timedelta(days=1)

    if item_date in fx_lookup:
        rate = fx_lookup[item_date].get(currency)
        if rate and not pd.isna(rate):
            return row['Amount'] / float(rate)

    return row['Amount in plan currency']

lp_filtered['Amount_EUR'] = lp_filtered.apply(convert_to_eur, axis=1)

print(f"\n✓ FX conversion complete")
print(f"  Rows with EUR conversion: {(lp_filtered['Plan Currency'] != 'EUR').sum():,}")

# -----------------------------------------------------------------------------
# 5.3: Pivot to Wide Format (W1-W4 Forecasts)
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("5.3: PIVOT TO WIDE FORMAT")
print("="*80)

# Parse dates
lp_filtered['Item_Date'] = pd.to_datetime(lp_filtered["Item's Date"])

# Sort by date within each group
lp_filtered = lp_filtered.sort_values([
    'Entity', 'Entity Name', 'Year Title', 'Liquidity_Group', 'Item_Date'
])

# Create week number (1, 2, 3, 4) within each group
lp_filtered['Week_Num'] = lp_filtered.groupby(
    ['Entity', 'Entity Name', 'Year Title', 'Liquidity_Group']
).cumcount() + 1

# Keep only first 4 weeks
lp_filtered = lp_filtered[lp_filtered['Week_Num'] <= 4]

# Pivot
lp_wide = lp_filtered.pivot_table(
    index=['Entity', 'Entity Name', 'Year Title', 'Liquidity_Group'],
    columns='Week_Num',
    values='Amount_EUR',
    aggfunc='first'
).reset_index()

# Rename columns
lp_wide.columns = ['Entity', 'Entity_Name', 'Year_Title', 'Liquidity_Group',
                    'W1_Forecast', 'W2_Forecast', 'W3_Forecast', 'W4_Forecast']

# Standardize entity names
lp_wide['Entity'] = lp_wide['Entity'].astype(str).str.replace('57', '057', regex=False)

print(f"\n✓ LP pivoted to wide format:")
print(f"  Rows: {len(lp_wide):,}")
print(f"  Columns: {list(lp_wide.columns)}")

# -----------------------------------------------------------------------------
# 5.4: Forward-Fill Missing Forecasts
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("5.4: HANDLING MISSING FORECASTS")
print("="*80)

# Forward-fill missing W2, W3, W4
lp_wide['W2_Forecast'] = lp_wide['W2_Forecast'].fillna(lp_wide['W1_Forecast'])
lp_wide['W3_Forecast'] = lp_wide['W3_Forecast'].fillna(lp_wide['W2_Forecast'])
lp_wide['W4_Forecast'] = lp_wide['W4_Forecast'].fillna(lp_wide['W3_Forecast'])

missing_before = lp_filtered.isnull().sum().sum()
missing_after = lp_wide[['W1_Forecast', 'W2_Forecast', 'W3_Forecast', 'W4_Forecast']].isnull().sum().sum()

print(f"\n✓ Forward-fill complete:")
print(f"  Missing values before: {missing_before}")
print(f"  Missing values after: {missing_after}")

# -----------------------------------------------------------------------------
# 5.5: Convert Year_Title to Week_Start
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("5.5: CONVERTING YEAR_TITLE TO WEEK_START")
print("="*80)

# Year_Title format: '2021/CW50' → need to convert to week_start (Monday)

def year_title_to_week_start(year_title):
    """Convert Year_Title format to week_start Monday"""
    try:
        parts = year_title.split('/CW')
        year = int(parts[0])
        week_num = int(parts[1])

        # ISO week to date
        # Week 1 starts on the Monday nearest to Jan 1
        jan_4 = pd.Timestamp(year=year, month=1, day=4)  # Week 1 contains Jan 4
        week_1_monday = jan_4 - pd.Timedelta(days=jan_4.weekday())

        week_start = week_1_monday + pd.Timedelta(weeks=week_num-1)
        return week_start
    except:
        return pd.NaT

lp_wide['week_start'] = lp_wide['Year_Title'].apply(year_title_to_week_start)

print(f"\n✓ Year_Title converted to week_start:")
print(f"  Sample mappings:")
sample = lp_wide[['Year_Title', 'week_start']].drop_duplicates().head(5)
print(sample.to_string(index=False))

# -----------------------------------------------------------------------------
# 5.6: Create LP Features
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("5.6: CREATING LP-SPECIFIC FEATURES")
print("="*80)

# Forecast availability flags
lp_wide['W1_Available'] = lp_wide['W1_Forecast'].notna()
lp_wide['W2_Available'] = lp_wide['W2_Forecast'].notna()
lp_wide['W3_Available'] = lp_wide['W3_Forecast'].notna()
lp_wide['W4_Available'] = lp_wide['W4_Forecast'].notna()

# Count of available forecasts
lp_wide['LP_Available_Count'] = (
    lp_wide[['W1_Available', 'W2_Available', 'W3_Available', 'W4_Available']].sum(axis=1)
)

# LP volatility (std across W1-W4)
lp_wide['LP_Volatility'] = lp_wide[['W1_Forecast', 'W2_Forecast', 'W3_Forecast', 'W4_Forecast']].std(axis=1)

# LP trend (slope from W1 to W4)
lp_wide['LP_Trend'] = (
    (lp_wide['W4_Forecast'] - lp_wide['W1_Forecast']) / 3
).fillna(0)

# LP average magnitude
lp_wide['LP_Avg_Magnitude'] = lp_wide[['W1_Forecast', 'W2_Forecast', 'W3_Forecast', 'W4_Forecast']].mean(axis=1)

lp_features_created = 9  # 4 forecasts + 5 derived features
print(f"\n✓ Created {lp_features_created} LP features")

# -----------------------------------------------------------------------------
# 5.7: Save LP Wide Format
# -----------------------------------------------------------------------------

# Standardize column names for merge
lp_wide = lp_wide.rename(columns={
    'Entity': 'entity_id',
    'Liquidity_Group': 'liquidity_group'
})

lp_wide.to_csv('../data/intermediate/lp_curated_pivoted.csv', index=False)
print(f"\n✓ Saved: data/intermediate/lp_curated_pivoted.csv")

print("\n✓ Stage 5 complete")

# =============================================================================
# STAGE 6: MERGE ACTUALS + LP
# =============================================================================

print("\n" + "="*80)
print("STAGE 6: MERGE ACTUALS + LP")
print("="*80)

# Load weekly actuals with time-series features from Stage 4
weekly_actuals = pd.read_csv('../data/intermediate/weekly_actuals_ts_features.csv')
weekly_actuals['week_start'] = pd.to_datetime(weekly_actuals['week_start'])

print(f"\nWeekly actuals loaded:")
print(f"  Rows: {len(weekly_actuals):,}")
print(f"  Columns: {len(weekly_actuals.columns)}")

# -----------------------------------------------------------------------------
# 6.1: Merge on entity_id, liquidity_group, week_start
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("6.1: MERGING ACTUALS WITH LP")
print("="*80)

training_data = weekly_actuals.merge(
    lp_wide[['entity_id', 'liquidity_group', 'week_start',
             'W1_Forecast', 'W2_Forecast', 'W3_Forecast', 'W4_Forecast',
             'LP_Volatility', 'LP_Trend', 'LP_Avg_Magnitude', 'LP_Available_Count']],
    on=['entity_id', 'liquidity_group', 'week_start'],
    how='left'
)

print(f"\n✓ Merge complete:")
print(f"  Rows: {len(training_data):,}")
print(f"  Columns: {len(training_data.columns)}")
print(f"  Rows with LP data: {training_data['W1_Forecast'].notna().sum():,}")
print(f"  Rows without LP: {training_data['W1_Forecast'].isna().sum():,}")

# =============================================================================
# STAGE 7: CROSS-FEATURES (ACTUALS × LP)
# =============================================================================

print("\n" + "="*80)
print("STAGE 7: CROSS-FEATURES (ACTUALS × LP INTERACTIONS)")
print("="*80)

# -----------------------------------------------------------------------------
# 7.1: Historical LP Accuracy Features
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("7.1: HISTORICAL LP ACCURACY FEATURES")
print("="*80)

# LP W1 vs actual (lag 1 week) - how accurate was last week's LP forecast?
grouped = training_data.groupby(['entity_id', 'liquidity_group'])

training_data['lp_w1_vs_actual_lag1'] = (
    training_data['W1_Forecast'] - grouped['amount_eur'].shift(1)
)

training_data['lp_error_pct_lag1'] = (
    training_data['lp_w1_vs_actual_lag1'].abs() /
    grouped['amount_eur'].shift(1).abs()
).replace([np.inf, -np.inf], np.nan)

# Rolling 4-week LP error rate
training_data['lp_error_rolling_4w'] = grouped['lp_error_pct_lag1'].transform(
    lambda x: x.rolling(4, min_periods=1).mean()
)

# LP bias (tends to over/under forecast?)
training_data['lp_bias_4w'] = grouped['lp_w1_vs_actual_lag1'].transform(
    lambda x: x.rolling(4, min_periods=1).mean()
)

lp_accuracy_features = 4
print(f"\n✓ Created {lp_accuracy_features} LP accuracy features")

# -----------------------------------------------------------------------------
# 7.2: Divergence Features
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("7.2: DIVERGENCE FEATURES (ACTUAL VS LP)")
print("="*80)

# Current actual vs LP W1 difference
training_data['actual_minus_lp_w1'] = training_data['amount_eur'] - training_data['W1_Forecast']

# Actual vs LP as ratio
training_data['actual_to_lp_ratio'] = (
    training_data['amount_eur'] / training_data['W1_Forecast']
).replace([np.inf, -np.inf], np.nan)

# Actual trend vs LP trend divergence
training_data['trend_divergence'] = training_data['trend_4w'] - training_data['LP_Trend']

# Volatility comparison
training_data['volatility_actual_vs_lp'] = training_data['cv_4w'] - (
    training_data['LP_Volatility'] / training_data['LP_Avg_Magnitude'].abs()
)

divergence_features = 4
print(f"\n✓ Created {divergence_features} divergence features")

# -----------------------------------------------------------------------------
# 7.3: Interaction Features
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("7.3: INTERACTION FEATURES")
print("="*80)

# Lag × LP interactions
training_data['lag1_x_lp_w1'] = training_data['lag_1'] * training_data['W1_Forecast']
training_data['lag4_x_lp_w1'] = training_data['lag_4'] * training_data['W1_Forecast']

# Rolling mean × LP
training_data['rolling_4w_mean_x_lp_w1'] = training_data['rolling_4w_mean'] * training_data['W1_Forecast']
training_data['rolling_8w_mean_x_lp_w2'] = training_data['rolling_8w_mean'] * training_data['W2_Forecast']

# Trend × LP trend
training_data['trend_4w_x_lp_trend'] = training_data['trend_4w'] * training_data['LP_Trend']

interaction_features = 5
print(f"\n✓ Created {interaction_features} interaction features")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("STAGES 5-7 SUMMARY")
print("="*80)

stage5_features = lp_features_created  # 9
stage6_features = 0  # Merge only, no new features
stage7_features = lp_accuracy_features + divergence_features + interaction_features  # 13

total_new_features = stage5_features + stage7_features

print(f"\n✓ NEW FEATURES CREATED IN STAGES 5-7: {total_new_features}")
print(f"  - Stage 5 (LP features): {stage5_features}")
print(f"  - Stage 6 (Merge): {stage6_features} (merge only)")
print(f"  - Stage 7 (Cross-features): {stage7_features}")

# Update cumulative count
stage2_features = 38
stage3_features = 12
stage4_features = 62
cumulative_total = stage2_features + stage3_features + stage4_features + total_new_features

print(f"\n✓ CUMULATIVE FEATURE COUNT:")
print(f"  - Stage 2 (Daily): {stage2_features}")
print(f"  - Stage 3 (Aggregation): {stage3_features}")
print(f"  - Stage 4 (Time-Series): {stage4_features}")
print(f"  - Stages 5-7 (LP + Cross): {total_new_features}")
print(f"  - TOTAL: {cumulative_total}")
print(f"  - Target: ~118")
print(f"  - Status: {'✓ TARGET REACHED' if cumulative_total >= 118 else f'Need {118 - cumulative_total} more'}")

# Final dataset stats
print(f"\n✓ FINAL TRAINING DATASET:")
print(f"  - Rows: {len(training_data):,}")
print(f"  - Columns: {len(training_data.columns)}")
print(f"  - Entities: {training_data['entity_id'].nunique()} (Tier 1 only)")
print(f"  - Weeks: {training_data['week_start'].nunique()}")
print(f"  - Date range: {training_data['week_start'].min().date()} to {training_data['week_start'].max().date()}")

# Save final training dataset
training_data.to_csv('../data/intermediate/training_data_complete.csv', index=False)
print(f"\n✓ Saved: data/intermediate/training_data_complete.csv")

print("\n" + "="*80)
print("✓ STAGES 5-7 COMPLETE - FEATURE ENGINEERING DONE!")
print("="*80)
print("\nReady for Stage 8: Final Analysis & Feature Selection")
