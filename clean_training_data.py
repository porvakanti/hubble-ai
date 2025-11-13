#!/usr/bin/env python3
"""
Clean Training Dataset - Conservative Approach

Changes:
1. Start from week 53 (Feb 27, 2023) - after all lags fill
2. Replace zeros in LP features with NaN
3. Remove forecast-vs-actual comparison features
4. Remove zero-variance features
5. Drop questionable/unsure features
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("CLEANING TRAINING DATASET")
print("=" * 80)
print()

# Load original data
print("1. Loading original training data...")
df = pd.read_csv('data/intermediate/training_data_enhanced.csv')
df['week_start'] = pd.to_datetime(df['week_start'])

print(f"   Original shape: {df.shape}")
print(f"   Date range: {df['week_start'].min()} to {df['week_start'].max()}")
print()

# Identify metadata and feature columns
metadata_cols = ['entity_id', 'liquidity_group', 'week_start', 'amount_eur']
all_feature_cols = [col for col in df.columns if col not in metadata_cols]

print(f"2. Original feature count: {len(all_feature_cols)}")
print()

# Step 1: Start from week 53 (Feb 27, 2023)
print("3. Filtering to week 53+ (after lag_52 fills)...")
cutoff_date = pd.Timestamp('2023-02-27')
df_clean = df[df['week_start'] >= cutoff_date].copy()

print(f"   Rows before: {len(df)}")
print(f"   Rows after: {len(df_clean)}")
print(f"   Weeks removed: {df['week_start'].nunique() - df_clean['week_start'].nunique()}")
print(f"   New date range: {df_clean['week_start'].min()} to {df_clean['week_start'].max()}")
print()

# Step 2: Replace zeros in LP features with NaN
print("4. Replacing zeros in LP features with NaN...")
lp_features = ['W1_Forecast', 'W2_Forecast', 'W3_Forecast', 'W4_Forecast']

zeros_replaced = 0
for col in lp_features:
    if col in df_clean.columns:
        zero_count = (df_clean[col] == 0).sum()
        df_clean.loc[df_clean[col] == 0, col] = np.nan
        zeros_replaced += zero_count
        print(f"   {col}: {zero_count} zeros replaced with NaN")

if zeros_replaced == 0:
    print("   No zeros found in LP forecast features")
print()

# Step 3: Remove forecast-vs-actual comparison features
print("5. Removing forecast-vs-actual comparison features...")
vs_actual_features = [col for col in all_feature_cols if 'vs_actual' in col.lower() or 'v_actual' in col.lower()]

if vs_actual_features:
    print(f"   Found {len(vs_actual_features)} features to remove:")
    for col in vs_actual_features:
        print(f"   - {col}")
    df_clean = df_clean.drop(columns=vs_actual_features)
else:
    print("   No vs_actual features found")
print()

# Step 4: Remove zero-variance features
print("6. Removing zero-variance features...")
zero_var_features = []

for col in all_feature_cols:
    if col in df_clean.columns and col not in vs_actual_features:
        # Check variance
        if df_clean[col].dtype in ['int64', 'float64']:
            if df_clean[col].nunique() <= 1:
                zero_var_features.append(col)

if zero_var_features:
    print(f"   Found {len(zero_var_features)} zero-variance features:")
    for col in zero_var_features:
        print(f"   - {col}")
    df_clean = df_clean.drop(columns=zero_var_features)
else:
    print("   No zero-variance features found")
print()

# Step 5: Remove questionable/unsure features
print("7. Removing questionable features (if unsure, drop)...")

questionable_features = []

# Check for features I'm unsure about
potential_questionable = ['volume_adjusted_amount', 'distance_from_typical', 'anomaly_score']

for col in potential_questionable:
    if col in df_clean.columns:
        questionable_features.append(col)

if questionable_features:
    print(f"   Found {len(questionable_features)} questionable features:")
    for col in questionable_features:
        print(f"   - {col}")
    df_clean = df_clean.drop(columns=questionable_features)
else:
    print("   No questionable features found")
print()

# Final cleanup
print("8. Final statistics:")
print("-" * 80)

remaining_features = [col for col in df_clean.columns if col not in metadata_cols]

print(f"   Final shape: {df_clean.shape}")
print(f"   Features removed: {len(all_feature_cols) - len(remaining_features)}")
print(f"   Features remaining: {len(remaining_features)}")
print(f"   Rows: {len(df_clean):,}")
print(f"   Entities: {df_clean['entity_id'].nunique()}")
print(f"   Entity-liq combinations: {df_clean.groupby(['entity_id', 'liquidity_group']).ngroups}")
print(f"   Weeks: {df_clean['week_start'].nunique()}")
print()

# Check for NaN values
nan_counts = df_clean[remaining_features].isnull().sum()
nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

if len(nan_counts) > 0:
    print("   Features with NaN values:")
    for col, count in nan_counts.items():
        print(f"   - {col}: {count} ({count/len(df_clean)*100:.1f}%)")
else:
    print("   No NaN values in features (except LP where zeros replaced)")
print()

# Save cleaned data
output_path = Path('data/intermediate/training_data_clean.csv')
df_clean.to_csv(output_path, index=False)

print("=" * 80)
print(f"CLEAN DATASET SAVED: {output_path}")
print("=" * 80)
print()

# Summary of changes
print("SUMMARY OF CHANGES:")
print("-" * 80)
print(f"1. ✅ Started from week 53 (Feb 27, 2023)")
print(f"2. ✅ Replaced zeros in LP features with NaN: 4 features, {zeros_replaced} zeros total")
print(f"3. ✅ Removed forecast-vs-actual features: {len(vs_actual_features)} features")
print(f"4. ✅ Removed zero-variance features: {len(zero_var_features)} features")
print(f"5. ✅ Removed questionable features: {len(questionable_features)} features")
print()
print(f"Total features removed: {len(all_feature_cols) - len(remaining_features)}")
print(f"Total features remaining: {len(remaining_features)}")
print(f"Rows removed: {len(df) - len(df_clean):,} ({(len(df) - len(df_clean))/len(df)*100:.1f}%)")
print()
