#!/usr/bin/env python3
"""
Quick analysis of LightGBM Recursive results with clean dataset
"""
import pandas as pd
import numpy as np

# Load results
results = pd.read_csv('artifacts/backtesting/3month/lightgbm_recursive_3month_20251113_105113.csv')

# Filter near-zero actuals
results_clean = results[results['actual'].abs() >= 1000].copy()

print("=" * 80)
print("LIGHTGBM RECURSIVE - CLEAN DATASET RESULTS")
print("=" * 80)
print()

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

# Compare to previous baseline (enhanced dataset)
enhanced_w1_wape = 19.79  # LightGBM Recursive on enhanced dataset
clean_w1_wape = wapes[0]
improvement = enhanced_w1_wape - clean_w1_wape

print("Comparison to Enhanced Dataset Baseline:")
print("-" * 80)
print(f"Enhanced Dataset W1 WAPE:  {enhanced_w1_wape:.2f}%")
print(f"Clean Dataset W1 WAPE:     {clean_w1_wape:.2f}%")
print(f"Improvement:               {improvement:+.2f} pp {'(Clean better)' if improvement > 0 else '(Enhanced better)'}")
print()

# Compare to Treasury LP baseline
treasury_lp_wape = 10.0
difference_from_lp = clean_w1_wape - treasury_lp_wape
print("Comparison to Treasury LP Baseline:")
print("-" * 80)
print(f"Treasury LP W1 WAPE:       {treasury_lp_wape:.2f}%")
print(f"ML Clean Dataset W1 WAPE:  {clean_w1_wape:.2f}%")
print(f"Gap:                       {difference_from_lp:+.2f} pp")
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
print("ANALYSIS COMPLETE")
print("=" * 80)
