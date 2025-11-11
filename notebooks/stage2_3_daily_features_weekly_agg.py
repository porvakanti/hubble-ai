# Stage 2 & 3: Feature Engineering - Daily Level & Weekly Aggregation
# To be added to EDA_v2_Comprehensive.ipynb

"""
Stage 2: Feature Engineering Part 1 (Daily Level)
Stage 3: Weekly Aggregation with Feature Preservation

This script creates ~118 features following a multi-stage approach:
1. Identify entity tiers (Tier 1 vs Tier 2)
2. Create daily-level features BEFORE aggregation
3. Aggregate to weekly preserving feature statistics
4. Filter to Tier 1 entities for ML training
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load raw data
print("\nLoading raw actuals data...")
actuals = pd.read_csv('../data/raw/actuals_curated.csv')
actuals['Value Date'] = pd.to_datetime(actuals['Value Date'])

print(f"  Rows: {len(actuals):,}")
print(f"  Date range: {actuals['Value Date'].min().date()} to {actuals['Value Date'].max().date()}")

# =============================================================================
# STAGE 2: FEATURE ENGINEERING - DAILY LEVEL
# =============================================================================

print("\n" + "="*80)
print("STAGE 2: FEATURE ENGINEERING (DAILY LEVEL)")
print("="*80)

# -----------------------------------------------------------------------------
# 2.1: Entity Tiering - Identify Tier 1 vs Tier 2
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("2.1: ENTITY TIERING ANALYSIS")
print("="*80)

# Analyze data coverage by entity
entity_coverage = actuals.groupby('Entity').agg({
    'Value Date': ['min', 'max', 'count'],
    'Amount Functional Currency': 'count'
}).reset_index()

entity_coverage.columns = ['entity', 'first_date', 'last_date', 'days_count', 'txn_count']

# Calculate coverage metrics
entity_coverage['data_span_days'] = (
    pd.to_datetime(entity_coverage['last_date']) -
    pd.to_datetime(entity_coverage['first_date'])
).dt.days

entity_coverage['coverage_since_2022'] = (
    pd.to_datetime(entity_coverage['first_date']) <= pd.Timestamp('2022-06-01')
)

# Tier assignment criteria:
# Tier 1: Data coverage >= Dec 2023 (at least 1 year of history)
# Tier 2: Data coverage < Dec 2023 (limited history, use LP fallback)
tier_cutoff_date = pd.Timestamp('2023-12-01')

entity_coverage['tier'] = entity_coverage['first_date'].apply(
    lambda x: 1 if pd.to_datetime(x) <= tier_cutoff_date else 2
)

print("\nEntity Tiering Summary:")
print(f"  Tier 1 (ML Forecast): {(entity_coverage['tier'] == 1).sum()} entities")
print(f"  Tier 2 (LP Fallback): {(entity_coverage['tier'] == 2).sum()} entities")

print("\nTier 1 Entities (sufficient history for ML):")
tier1_entities = entity_coverage[entity_coverage['tier'] == 1]['entity'].tolist()
print(f"  {tier1_entities}")

print("\nTier 2 Entities (limited history, use LP):")
tier2_entities = entity_coverage[entity_coverage['tier'] == 2]['entity'].tolist()
print(f"  {tier2_entities}")

print("\nDetailed Entity Coverage:")
print(entity_coverage[['entity', 'first_date', 'last_date', 'data_span_days', 'txn_count', 'tier']].to_string(index=False))

# Save entity tier mapping
entity_tier_map = entity_coverage[['entity', 'tier']].copy()
entity_tier_map.to_csv('../data/reference/entity_tier_mapping.csv', index=False)
print("\n✓ Entity tier mapping saved to: data/reference/entity_tier_mapping.csv")

# Filter actuals to Tier 1 entities only for ML training
actuals_tier1 = actuals[actuals['Entity'].isin(tier1_entities)].copy()
print(f"\n✓ Filtered to Tier 1 entities: {len(actuals_tier1):,} rows (was {len(actuals):,})")

# -----------------------------------------------------------------------------
# 2.2: Temporal Features (Daily Level)
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("2.2: TEMPORAL FEATURES (DAILY LEVEL)")
print("="*80)

# Create comprehensive temporal features
actuals_tier1['year'] = actuals_tier1['Value Date'].dt.year
actuals_tier1['month'] = actuals_tier1['Value Date'].dt.month
actuals_tier1['quarter'] = actuals_tier1['Value Date'].dt.quarter
actuals_tier1['week_of_year'] = actuals_tier1['Value Date'].dt.isocalendar().week
actuals_tier1['day_of_week'] = actuals_tier1['Value Date'].dt.dayofweek  # 0=Monday
actuals_tier1['day_of_month'] = actuals_tier1['Value Date'].dt.day
actuals_tier1['day_of_year'] = actuals_tier1['Value Date'].dt.dayofyear

# Binary temporal indicators
actuals_tier1['is_month_start'] = actuals_tier1['day_of_month'] <= 5
actuals_tier1['is_month_end'] = actuals_tier1['day_of_month'] >= 25
actuals_tier1['is_quarter_start'] = (actuals_tier1['month'].isin([1, 4, 7, 10])) & (actuals_tier1['day_of_month'] <= 5)
actuals_tier1['is_quarter_end'] = (actuals_tier1['month'].isin([3, 6, 9, 12])) & (actuals_tier1['day_of_month'] >= 25)
actuals_tier1['is_year_start'] = (actuals_tier1['month'] == 1) & (actuals_tier1['day_of_month'] <= 5)
actuals_tier1['is_year_end'] = (actuals_tier1['month'] == 12) & (actuals_tier1['day_of_month'] >= 25)
actuals_tier1['is_weekend'] = actuals_tier1['day_of_week'] >= 5

# Week of month (1-5)
actuals_tier1['week_of_month'] = ((actuals_tier1['day_of_month'] - 1) // 7) + 1

print(f"\n✓ Created {12} temporal features:")
temporal_features = [
    'year', 'month', 'quarter', 'week_of_year', 'day_of_week', 'day_of_month',
    'day_of_year', 'week_of_month', 'is_month_start', 'is_month_end',
    'is_quarter_start', 'is_quarter_end', 'is_year_start', 'is_year_end',
    'is_weekend'
]
for feat in temporal_features[:5]:
    print(f"  - {feat}")
print(f"  ... and {len(temporal_features) - 5} more")

# -----------------------------------------------------------------------------
# 2.3: Transaction-Level Features
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("2.3: TRANSACTION-LEVEL FEATURES")
print("="*80)

# Transaction amount characteristics
actuals_tier1['amount_abs'] = actuals_tier1['Amount Functional Currency'].abs()
actuals_tier1['amount_sign'] = np.sign(actuals_tier1['Amount Functional Currency'])
actuals_tier1['amount_log'] = np.log1p(actuals_tier1['amount_abs'])  # log(1+x) to handle zeros

# Transaction size categories (per liquidity group)
for liq_group in ['TRR', 'TRP']:
    mask = actuals_tier1['Liquidity Group'] == liq_group
    q33 = actuals_tier1.loc[mask, 'amount_abs'].quantile(0.33)
    q67 = actuals_tier1.loc[mask, 'amount_abs'].quantile(0.67)

    actuals_tier1.loc[mask, 'txn_size_category'] = pd.cut(
        actuals_tier1.loc[mask, 'amount_abs'],
        bins=[0, q33, q67, float('inf')],
        labels=['small', 'medium', 'large']
    )

print(f"\n✓ Created {4} transaction-level features:")
print(f"  - amount_abs, amount_sign, amount_log")
print(f"  - txn_size_category (small/medium/large)")

# -----------------------------------------------------------------------------
# 2.4: Rolling Daily Features (per entity × liquidity group)
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("2.4: ROLLING DAILY FEATURES")
print("="*80)

# Sort by entity, liquidity group, date
actuals_tier1 = actuals_tier1.sort_values(['Entity', 'Liquidity Group', 'Value Date'])

# Rolling windows: 7, 14, 30 days
rolling_windows = [7, 14, 30]

print(f"\nComputing rolling features for {len(tier1_entities)} entities...")

for window in rolling_windows:
    print(f"  Rolling {window}-day features...")

    # Group by entity × liquidity group
    grouped = actuals_tier1.groupby(['Entity', 'Liquidity Group'])

    # Rolling sum
    actuals_tier1[f'rolling_{window}d_sum'] = grouped['Amount Functional Currency'].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )

    # Rolling mean
    actuals_tier1[f'rolling_{window}d_mean'] = grouped['Amount Functional Currency'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    # Rolling std
    actuals_tier1[f'rolling_{window}d_std'] = grouped['Amount Functional Currency'].transform(
        lambda x: x.rolling(window, min_periods=1).std()
    )

    # Rolling count (number of transactions)
    actuals_tier1[f'rolling_{window}d_count'] = grouped['Amount Functional Currency'].transform(
        lambda x: x.rolling(window, min_periods=1).count()
    )

rolling_features_created = len(rolling_windows) * 4  # 4 metrics per window
print(f"\n✓ Created {rolling_features_created} rolling daily features")

# -----------------------------------------------------------------------------
# 2.5: Day-of-Week Pattern Features
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("2.5: DAY-OF-WEEK PATTERN FEATURES")
print("="*80)

# Compute average amount by day of week (per entity × liquidity group)
dow_patterns = actuals_tier1.groupby(['Entity', 'Liquidity Group', 'day_of_week']).agg({
    'Amount Functional Currency': ['mean', 'std', 'count']
}).reset_index()

dow_patterns.columns = ['entity', 'liq_group', 'dow', 'dow_avg_amount', 'dow_std_amount', 'dow_txn_count']

# Pivot to create one column per day of week
dow_pivoted = dow_patterns.pivot_table(
    index=['entity', 'liq_group'],
    columns='dow',
    values='dow_avg_amount'
).reset_index()

# Rename columns
dow_pivoted.columns = ['entity', 'liq_group'] + [f'dow_{int(d)}_avg_amount' for d in dow_pivoted.columns[2:]]

# Merge back to actuals
actuals_tier1 = actuals_tier1.merge(
    dow_pivoted,
    left_on=['Entity', 'Liquidity Group'],
    right_on=['entity', 'liq_group'],
    how='left'
).drop(['entity', 'liq_group'], axis=1)

dow_features_created = 7  # One per day of week
print(f"\n✓ Created {dow_features_created} day-of-week pattern features")

# -----------------------------------------------------------------------------
# 2.6: Stage 2 Summary
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("STAGE 2 SUMMARY")
print("="*80)

daily_features_count = (
    len(temporal_features) +  # 12 temporal
    4 +  # transaction-level
    rolling_features_created +  # 12 rolling
    dow_features_created  # 7 day-of-week
)

print(f"\n✓ DAILY-LEVEL FEATURES CREATED: {daily_features_count}")
print(f"  - Temporal features: {len(temporal_features)}")
print(f"  - Transaction-level: 4")
print(f"  - Rolling daily features: {rolling_features_created}")
print(f"  - Day-of-week patterns: {dow_features_created}")

print(f"\n✓ Data prepared for weekly aggregation:")
print(f"  - Rows: {len(actuals_tier1):,} (Tier 1 entities only)")
print(f"  - Columns: {len(actuals_tier1.columns)}")
print(f"  - Entities: {actuals_tier1['Entity'].nunique()} (Tier 1 only)")

print("\n✓ Stage 2 complete. Ready for Stage 3: Weekly Aggregation")

# =============================================================================
# STAGE 3: WEEKLY AGGREGATION
# =============================================================================

print("\n" + "="*80)
print("STAGE 3: WEEKLY AGGREGATION WITH FEATURE PRESERVATION")
print("="*80)

# -----------------------------------------------------------------------------
# 3.1: Create Week Start (Monday)
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("3.1: WEEK NORMALIZATION TO MONDAY")
print("="*80)

# Calculate week_start (Monday of each week)
actuals_tier1['week_start'] = actuals_tier1['Value Date'] - pd.to_timedelta(
    actuals_tier1['Value Date'].dt.dayofweek, unit='D'
)

print(f"\n✓ Week normalization complete:")
print(f"  - All dates normalized to Monday of their week")
print(f"  - Sample mappings:")
sample_weeks = actuals_tier1[['Value Date', 'week_start']].drop_duplicates().head(10)
print(sample_weeks.to_string(index=False))

# -----------------------------------------------------------------------------
# 3.2: Weekly Aggregation Strategy
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("3.2: WEEKLY AGGREGATION")
print("="*80)

# Define aggregation dictionary
agg_dict = {
    'Amount Functional Currency': ['sum', 'mean', 'std', 'count', 'min', 'max'],
    'amount_abs': ['sum', 'mean'],
    'is_month_start': 'max',  # Any day in week was month start?
    'is_month_end': 'max',
    'is_quarter_start': 'max',
    'is_quarter_end': 'max',
    'is_year_start': 'max',
    'is_year_end': 'max',
    'is_weekend': 'sum',  # Count of weekend transactions
}

# Add rolling features (take last value of week)
for window in rolling_windows:
    agg_dict[f'rolling_{window}d_sum'] = 'last'
    agg_dict[f'rolling_{window}d_mean'] = 'last'
    agg_dict[f'rolling_{window}d_std'] = 'last'
    agg_dict[f'rolling_{window}d_count'] = 'last'

# Add day-of-week averages (take mean across week)
for dow in range(7):
    col = f'dow_{dow}_avg_amount'
    if col in actuals_tier1.columns:
        agg_dict[col] = 'mean'

print(f"\nAggregating to weekly level...")
print(f"  Aggregation metrics: {len(agg_dict)} feature groups")

# Aggregate
weekly_actuals = actuals_tier1.groupby([
    'Entity', 'Liquidity Group', 'week_start'
]).agg(agg_dict).reset_index()

# Flatten column names
weekly_actuals.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                          for col in weekly_actuals.columns]

# Rename core columns
weekly_actuals = weekly_actuals.rename(columns={
    'Entity': 'entity_id',
    'Liquidity Group': 'liquidity_group',
    'Amount Functional Currency_sum': 'amount_eur',
    'Amount Functional Currency_mean': 'amount_mean',
    'Amount Functional Currency_std': 'amount_std',
    'Amount Functional Currency_count': 'txn_count',
    'Amount Functional Currency_min': 'amount_min',
    'Amount Functional Currency_max': 'amount_max',
})

print(f"\n✓ Weekly aggregation complete:")
print(f"  - Rows: {len(weekly_actuals):,}")
print(f"  - Columns: {len(weekly_actuals.columns)}")
print(f"  - Date range: {weekly_actuals['week_start'].min()} to {weekly_actuals['week_start'].max()}")
print(f"  - Weeks: {weekly_actuals['week_start'].nunique()}")

# -----------------------------------------------------------------------------
# 3.3: Add Week-Level Temporal Features
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("3.3: WEEK-LEVEL TEMPORAL FEATURES")
print("="*80)

weekly_actuals['week_year'] = weekly_actuals['week_start'].dt.isocalendar().year
weekly_actuals['week_number'] = weekly_actuals['week_start'].dt.isocalendar().week
weekly_actuals['month'] = weekly_actuals['week_start'].dt.month
weekly_actuals['quarter'] = weekly_actuals['week_start'].dt.quarter
weekly_actuals['year'] = weekly_actuals['week_start'].dt.year

print(f"\n✓ Added 5 week-level temporal features")

# -----------------------------------------------------------------------------
# 3.4: Stage 3 Summary
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print("STAGE 3 SUMMARY")
print("="*80)

print(f"\n✓ WEEKLY AGGREGATED DATASET READY:")
print(f"  - Rows: {len(weekly_actuals):,}")
print(f"  - Columns: {len(weekly_actuals.columns)}")
print(f"  - Entities: {weekly_actuals['entity_id'].nunique()} (Tier 1 only)")
print(f"  - Liquidity Groups: {sorted(weekly_actuals['liquidity_group'].unique())}")
print(f"  - Weeks: {weekly_actuals['week_start'].nunique()}")
print(f"  - Date range: {weekly_actuals['week_start'].min().date()} to {weekly_actuals['week_start'].max().date()}")

# Sample preview
print(f"\nSample data (first 5 rows):")
print(weekly_actuals.head(5)[['entity_id', 'liquidity_group', 'week_start', 'amount_eur', 'txn_count']].to_string(index=False))

# Save intermediate output
weekly_actuals.to_csv('../data/intermediate/weekly_actuals_featured.csv', index=False)
print(f"\n✓ Saved to: data/intermediate/weekly_actuals_featured.csv")

print("\n" + "="*80)
print("✓ Stages 2 & 3 Complete!")
print("="*80)
print("\nNext steps:")
print("  - Stage 4: Feature Engineering Part 2 (Weekly Time-Series Features)")
print("  - Stage 5: LP Processing & Pivoting")
print("  - Stage 6: Merge Actuals + LP")
print("  - Stage 7: Feature Engineering Part 3 (Cross-features)")
print("  - Stage 8: Final Analysis & Feature Selection")
