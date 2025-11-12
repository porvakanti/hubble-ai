"""
Simple Backtest: LightGBM + Recursive Strategy

Optimized for speed and reliability:
- Single model: LightGBM only
- Single strategy: Recursive only
- Checkpoint saving every 5 weeks
- Expected time: ~30 minutes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from cf_forecast.backtesting import WalkForwardBacktester

print("="*80)
print("LIGHTGBM + RECURSIVE BACKTEST")
print("="*80)
print()

# Config
config = {
    'target_column': 'amount_eur',
    'entity_column': 'entity_id',
    'liquidity_group_column': 'liquidity_group',
    'week_column': 'week_start'
}

# Load enhanced training data
print("Loading enhanced training data...")
training_data = pd.read_csv('data/intermediate/training_data_enhanced.csv')
training_data['week_start'] = pd.to_datetime(training_data['week_start'])

# Convert features to numeric
print("Converting features to numeric...")
metadata_cols = ['entity_id', 'liquidity_group', 'week_start']
feature_cols = [col for col in training_data.columns if col not in metadata_cols and col != 'amount_eur']

for col in feature_cols:
    training_data[col] = pd.to_numeric(training_data[col], errors='coerce')

training_data[feature_cols] = training_data[feature_cols].fillna(0)
training_data['amount_eur'] = pd.to_numeric(training_data['amount_eur'], errors='coerce')

print(f"  Rows: {len(training_data):,}")
print(f"  Features: {len(feature_cols)}")
print(f"  Entities: {training_data['entity_id'].nunique()}")
print(f"  Entity-liq combinations: {training_data.groupby(['entity_id', 'liquidity_group']).ngroups}")
print()

# Backtest period
end_date = training_data['week_start'].max()
start_date = end_date - pd.DateOffset(months=6)

print(f"Backtest period:")
print(f"  Start: {start_date.date()}")
print(f"  End: {end_date.date()}")
print(f"  Weeks: 27")
print()

print("Configuration:")
print("  Model: LightGBM")
print("  Strategy: Recursive")
print("  Horizons: W1-W8")
print()
print("Estimated time: ~30 minutes")
print()

# Run backtest
backtester = WalkForwardBacktester(config)

print("="*80)
print("STARTING BACKTEST...")
print("="*80)
print()

results = backtester.run_backtest(
    training_data=training_data,
    start_date=start_date,
    end_date=end_date,
    models=['lightgbm'],
    strategies=['recursive']
)

# Save results
output_dir = Path('artifacts/backtesting/simplified')
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_path = output_dir / f'lightgbm_recursive_{timestamp}.csv'
results.to_csv(results_path, index=False)

print()
print("="*80)
print("BACKTEST COMPLETE")
print("="*80)
print(f"✓ Results saved: {results_path}")
print(f"  Total forecasts: {len(results):,}")
print()

# Quick analysis
print("="*80)
print("QUICK ANALYSIS")
print("="*80)

# Filter clean data
results_clean = results[results['actual'].abs() >= 1000].copy()
print(f"Clean forecasts (|actual| >= €1,000): {len(results_clean):,}")
print()

# Portfolio WAPE by horizon
print("PORTFOLIO WAPE BY HORIZON:")
print("-"*80)

for horizon in range(1, 9):
    h_data = results_clean[results_clean['horizon'] == horizon]

    if len(h_data) > 0:
        total_actual = h_data['actual'].abs().sum()
        total_error = h_data['abs_error'].sum()
        wape = (total_error / total_actual * 100) if total_actual > 0 else 0

        target_map = {1: 5.0, 2: 7.5, 3: 10.0, 4: 12.5, 5: 15.0, 6: 17.5, 7: 20.0, 8: 22.5}
        target = target_map[horizon]

        status = "✅ PASS" if wape <= target else "❌ FAIL"
        gap = wape - target

        print(f"W{horizon}: {wape:6.2f}% (target ≤{target:5.1f}%, gap: {gap:+6.2f}%) {status}")

print()

# Directionality
directionality = results_clean['directionality'].mean() * 100
print(f"Directionality: {directionality:.1f}%")
print()

# Comparison to baseline
print("COMPARISON TO BASELINE:")
print("-"*80)
w1_data = results_clean[results_clean['horizon'] == 1]
w1_wape = (w1_data['abs_error'].sum() / w1_data['actual'].abs().sum() * 100)

print(f"Baseline (all 34 combinations, 152 features):")
print(f"  W1 WAPE: 21.4%")
print()
print(f"After Tier 2 exclusions (theoretical):")
print(f"  W1 WAPE: 18.1%")
print()
print(f"LightGBM + Enhanced Features (204 features, Tier 1 only):")
print(f"  W1 WAPE: {w1_wape:.2f}%")
print(f"  Improvement from baseline: {21.4 - w1_wape:.2f} pp")
print()

# Phase 1 target check
print("PHASE 1 TARGET:")
print("-"*80)
target_range = "14-15%"
if w1_wape <= 15.0:
    print(f"✅ TARGET MET: {w1_wape:.2f}% ≤ 15%")
    print(f"   Ready for Phase 2 (LSTM/SARIMAX)")
elif w1_wape <= 17.0:
    print(f"⚠️ CLOSE TO TARGET: {w1_wape:.2f}% vs 15% target")
    print(f"   Gap: {w1_wape - 15.0:.2f} pp")
    print(f"   Consider: XGBoost comparison or additional tuning")
else:
    print(f"❌ TARGET NOT MET: {w1_wape:.2f}% vs 15% target")
    print(f"   Gap: {w1_wape - 15.0:.2f} pp")
    print(f"   Next: Try XGBoost, Direct strategy, or entity-specific models")
print()

# Entity-level breakdown (top 5 best/worst)
print("ENTITY-LEVEL PERFORMANCE (W1):")
print("-"*80)

w1_by_entity = []
for (entity, liq), group in w1_data.groupby(['entity_id', 'liquidity_group']):
    total_actual = group['actual'].abs().sum()
    total_error = group['abs_error'].sum()
    wape = (total_error / total_actual * 100) if total_actual > 0 else 0

    w1_by_entity.append({
        'Entity-Liq': f'{entity}-{liq}',
        'WAPE': wape,
        'Forecasts': len(group)
    })

entity_df = pd.DataFrame(w1_by_entity).sort_values('WAPE')

print("\nTOP 5 PERFORMERS:")
for _, row in entity_df.head(5).iterrows():
    print(f"  {row['Entity-Liq']:15s}: {row['WAPE']:6.2f}% ({row['Forecasts']} forecasts)")

print("\nBOTTOM 5 PERFORMERS:")
for _, row in entity_df.tail(5).iterrows():
    print(f"  {row['Entity-Liq']:15s}: {row['WAPE']:6.2f}% ({row['Forecasts']} forecasts)")

print()
print("="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Review results above")
print("2. If approved, can proceed with:")
print("   - XGBoost + Recursive (for comparison)")
print("   - Direct strategy test")
print("   - DirRec strategy test")
print("   - LSTM/SARIMAX (Phase 2)")
print()
print(f"Results file: {results_path}")
print("="*80)
