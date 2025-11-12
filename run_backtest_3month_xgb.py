#!/usr/bin/env python3
"""
Run XGBoost + Recursive Strategy Backtest
3-month validation period for faster execution.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
sys.path.append('src')

from cf_forecast.backtesting import WalkForwardBacktester

print("=" * 80)
print("XGBOOST + RECURSIVE BACKTEST (3-MONTH VALIDATION)")
print("=" * 80)
print()

# Load enhanced training data
print("Loading enhanced training data...")
training_data = pd.read_csv('data/intermediate/training_data_enhanced.csv')

# Convert week_start to datetime
training_data['week_start'] = pd.to_datetime(training_data['week_start'])

# Convert features to numeric
print("Converting features to numeric...")
feature_cols = [col for col in training_data.columns
                if col not in ['entity_id', 'liquidity_group', 'week_start', 'amount_eur']]

for col in feature_cols:
    training_data[col] = pd.to_numeric(training_data[col], errors='coerce')
training_data[feature_cols] = training_data[feature_cols].fillna(0)

print(f"  Rows: {len(training_data):,}")
print(f"  Features: {len(feature_cols)}")
print(f"  Entities: {training_data['entity_id'].nunique()}")
print(f"  Entity-liq combinations: {training_data.groupby(['entity_id', 'liquidity_group']).ngroups}")
print()

# Backtest configuration - 3 MONTHS
start_date = pd.Timestamp('2025-07-07')
end_date = pd.Timestamp('2025-09-29')
weeks = (end_date - start_date).days // 7 + 1

print("Backtest period (3 months):")
print(f"  Start: {start_date.date()}")
print(f"  End: {end_date.date()}")
print(f"  Weeks: {weeks}")
print()

print("Configuration:")
print("  Model: XGBoost")
print("  Strategy: Recursive")
print("  Horizons: W1-W8")
print()

print("Estimated time: ~15 minutes")
print()

# Initialize backtester
config = {
    'target_column': 'amount_eur',
    'entity_column': 'entity_id',
    'liquidity_group_column': 'liquidity_group',
    'date_column': 'week_start'
}
backtester = WalkForwardBacktester(config)

# Run backtest
print("=" * 80)
print("STARTING BACKTEST...")
print("=" * 80)
print()

results = backtester.run_backtest(
    training_data=training_data,
    start_date=start_date,
    end_date=end_date,
    models=['xgboost'],
    strategies=['recursive']
)

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = Path('artifacts/backtesting/3month')
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f'xgboost_recursive_3month_{timestamp}.csv'
results.to_csv(output_file, index=False)

print()
print("=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
print()
print(f"Results saved to: {output_file}")
print()

# Quick analysis
print("=" * 80)
print("QUICK ANALYSIS")
print("=" * 80)
print()

# Filter near-zero actuals
results_clean = results[results['actual'].abs() >= 1000].copy()

print(f"Total forecasts: {len(results):,}")
print(f"After filtering |actual| < €1,000: {len(results_clean):,}")
print()

# Portfolio WAPE by horizon
print("Portfolio WAPE by Horizon:")
print("-" * 80)
print(f"{'Horizon':<10} {'WAPE':<10} {'Target':<10} {'Gap':<10} {'Status':<10}")
print("-" * 80)

targets = {1: 5.0, 2: 7.5, 3: 10.0, 4: 12.5, 5: 15.0, 6: 17.5, 7: 20.0, 8: 22.5}

wapes = []
for horizon in range(1, 9):
    h_data = results_clean[results_clean['horizon'] == horizon]
    total_actual = h_data['actual'].abs().sum()
    total_error = h_data['abs_error'].sum()
    wape = (total_error / total_actual * 100) if total_actual > 0 else 0
    wapes.append(wape)

    target = targets[horizon]
    gap = wape - target
    status = "✅ PASS" if wape <= target else "❌ FAIL"

    print(f"W{horizon:<9} {wape:>6.2f}%    ≤{target:>4.1f}%    {gap:>+6.2f}%   {status:<10}")

print()

# Directionality
results_clean['direction_actual'] = np.where(results_clean['actual'] > 0, 1, -1)
results_clean['direction_forecast'] = np.where(results_clean['forecast_p90'] > 0, 1, -1)
results_clean['direction_correct'] = (results_clean['direction_actual'] == results_clean['direction_forecast']).astype(int)
directionality = results_clean['direction_correct'].mean() * 100

print(f"Overall Directionality: {directionality:.2f}%")
print()

# Compare to LightGBM 3-month
lgb_w1_wape = 19.79
xgb_w1_wape = wapes[0]
difference = lgb_w1_wape - xgb_w1_wape

print("Comparison to LightGBM (3-month):")
print("-" * 80)
print(f"LightGBM W1 WAPE:  {lgb_w1_wape:.2f}%")
print(f"XGBoost W1 WAPE:   {xgb_w1_wape:.2f}%")
print(f"Difference:        {difference:+.2f} pp {'(XGBoost better)' if difference > 0 else '(LightGBM better)'}")
print()

# Entity performance (W1)
print("Entity-Liquidity Performance (W1 WAPE):")
print("-" * 80)

w1_data = results_clean[results_clean['horizon'] == 1]
entity_perf = []

for (entity_id, liquidity_group), group in w1_data.groupby(['entity_id', 'liquidity_group']):
    total_actual = group['actual'].abs().sum()
    total_error = group['abs_error'].sum()
    wape = (total_error / total_actual * 100) if total_actual > 0 else 0

    entity_perf.append({
        'entity_id': entity_id,
        'liquidity_group': liquidity_group,
        'wape': wape,
        'total_actual': total_actual
    })

entity_df = pd.DataFrame(entity_perf).sort_values('wape')

print(f"{'Rank':<6} {'Entity-Liq':<15} {'W1 WAPE':<12} {'Total (€M)':<15}")
print("-" * 80)

# Top 5
print("TOP 5 PERFORMERS:")
for idx, row in entity_df.head(5).iterrows():
    combo = f"{row['entity_id']}-{row['liquidity_group']}"
    rank = entity_df.index.get_loc(idx) + 1
    print(f"{rank:<6} {combo:<15} {row['wape']:>8.2f}%   {row['total_actual']/1e6:>11.2f}")

print()
print("BOTTOM 5 PERFORMERS:")
for idx, row in entity_df.tail(5).iterrows():
    combo = f"{row['entity_id']}-{row['liquidity_group']}"
    rank = entity_df.index.get_loc(idx) + 1
    print(f"{rank:<6} {combo:<15} {row['wape']:>8.2f}%   {row['total_actual']/1e6:>11.2f}")

print()
print("=" * 80)
print("XGBOOST COMPLETE - READY FOR MODEL COMPARISON")
print("=" * 80)
