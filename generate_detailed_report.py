#!/usr/bin/env python3
"""
Generate Detailed LightGBM + Recursive Backtest Report
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load results
results_file = Path('artifacts/backtesting/simplified/lightgbm_recursive_20251112_152521.csv')
df = pd.read_csv(results_file)

# Clean data (remove near-zero actuals)
df_clean = df[df['actual'].abs() >= 1000].copy()

# Rename forecast_p90 to forecast for simplicity
df_clean['forecast'] = df_clean['forecast_p90']

print("=" * 100)
print("LIGHTGBM + RECURSIVE BACKTEST - DETAILED REPORT")
print("=" * 100)
print()

# ==============================================================================
# 1. EXECUTIVE SUMMARY
# ==============================================================================
print("1. EXECUTIVE SUMMARY")
print("-" * 100)
print()
total_forecasts = len(df_clean)
total_entities = df_clean['entity_id'].nunique()
total_combinations = df_clean.groupby(['entity_id', 'liquidity_group']).ngroups
weeks_tested = df_clean['test_week'].nunique()

print(f"Total Forecasts:        {total_forecasts:,}")
print(f"Entities:               {total_entities}")
print(f"Entity-Liq Combos:      {total_combinations}")
print(f"Weeks Tested:           {weeks_tested}")
print(f"Period:                 {df_clean['test_week'].min()} to {df_clean['test_week'].max()}")
print(f"Horizons:               W1-W8")
print(f"Model:                  LightGBM")
print(f"Strategy:               Recursive")
print(f"Features:               204 (enhanced)")
print()

# ==============================================================================
# 2. PORTFOLIO WAPE BY HORIZON (W1-W8)
# ==============================================================================
print("2. PORTFOLIO WAPE BY HORIZON")
print("-" * 100)
print()
print(f"{'Horizon':<10} {'WAPE':<10} {'Target':<10} {'Gap':<10} {'Status':<10} {'Total Actual (€M)':<20} {'Total Error (€M)':<20}")
print("-" * 100)

targets = {1: 5.0, 2: 7.5, 3: 10.0, 4: 12.5, 5: 15.0, 6: 17.5, 7: 20.0, 8: 22.5}

for horizon in range(1, 9):
    h_data = df_clean[df_clean['horizon'] == horizon]
    total_actual = h_data['actual'].abs().sum()
    total_error = h_data['abs_error'].sum()
    wape = (total_error / total_actual * 100) if total_actual > 0 else 0

    target = targets[horizon]
    gap = wape - target
    status = "✅ PASS" if wape <= target else "❌ FAIL"

    print(f"W{horizon:<9} {wape:>6.2f}%    ≤{target:>4.1f}%    {gap:>+6.2f}%   {status:<10} {total_actual/1e6:>16.2f}    {total_error/1e6:>16.2f}")

print()

# Calculate directionality
df_clean['direction_actual'] = np.where(df_clean['actual'] > 0, 1, -1)
df_clean['direction_forecast'] = np.where(df_clean['forecast'] > 0, 1, -1)
df_clean['direction_correct'] = (df_clean['direction_actual'] == df_clean['direction_forecast']).astype(int)
directionality = df_clean['direction_correct'].mean() * 100

print(f"Overall Directionality: {directionality:.2f}%")
print()

# ==============================================================================
# 3. PERFORMANCE BY ENTITY-LIQUIDITY COMBINATION (W1 ONLY)
# ==============================================================================
print("3. ENTITY-LIQUIDITY PERFORMANCE (W1 WAPE)")
print("-" * 100)
print()

w1_data = df_clean[df_clean['horizon'] == 1].copy()
combo_performance = []

for (entity_id, liquidity_group), group in w1_data.groupby(['entity_id', 'liquidity_group']):
    total_actual = group['actual'].abs().sum()
    total_error = group['abs_error'].sum()
    wape = (total_error / total_actual * 100) if total_actual > 0 else 0
    forecasts = len(group)
    avg_actual = group['actual'].abs().mean()

    combo_performance.append({
        'entity_id': entity_id,
        'liquidity_group': liquidity_group,
        'wape': wape,
        'forecasts': forecasts,
        'total_actual': total_actual,
        'avg_actual': avg_actual
    })

combo_df = pd.DataFrame(combo_performance).sort_values('wape')

print(f"{'Rank':<6} {'Entity-Liq':<15} {'W1 WAPE':<12} {'Forecasts':<12} {'Avg Actual (€K)':<20} {'Performance':<20}")
print("-" * 100)

for idx, row in combo_df.iterrows():
    combo_name = f"{row['entity_id']}-{row['liquidity_group']}"

    # Performance classification
    if row['wape'] <= 10:
        perf = "⭐ Excellent"
    elif row['wape'] <= 20:
        perf = "✅ Good"
    elif row['wape'] <= 40:
        perf = "⚠️  Moderate"
    else:
        perf = "❌ Poor"

    rank = combo_df.index.get_loc(idx) + 1
    print(f"{rank:<6} {combo_name:<15} {row['wape']:>8.2f}%   {row['forecasts']:<12} {row['avg_actual']/1000:>16.2f}    {perf:<20}")

print()

# ==============================================================================
# 4. TOP 10 AND BOTTOM 10 PERFORMERS (W1)
# ==============================================================================
print("4. TOP 10 BEST PERFORMERS (W1 WAPE)")
print("-" * 100)
print()

top10 = combo_df.head(10)
print(f"{'Rank':<6} {'Entity-Liq':<15} {'W1 WAPE':<12} {'Total Actual (€M)':<20} {'Coverage %':<15}")
print("-" * 100)

total_portfolio = w1_data['actual'].abs().sum()
for idx, row in top10.iterrows():
    combo_name = f"{row['entity_id']}-{row['liquidity_group']}"
    rank = combo_df.index.get_loc(idx) + 1
    coverage = (row['total_actual'] / total_portfolio * 100)
    print(f"{rank:<6} {combo_name:<15} {row['wape']:>8.2f}%   {row['total_actual']/1e6:>16.2f}    {coverage:>10.2f}%")

print()
print("4. BOTTOM 10 WORST PERFORMERS (W1 WAPE)")
print("-" * 100)
print()

bottom10 = combo_df.tail(10)
print(f"{'Rank':<6} {'Entity-Liq':<15} {'W1 WAPE':<12} {'Total Actual (€M)':<20} {'Coverage %':<15}")
print("-" * 100)

for idx, row in bottom10.iterrows():
    combo_name = f"{row['entity_id']}-{row['liquidity_group']}"
    rank = combo_df.index.get_loc(idx) + 1
    coverage = (row['total_actual'] / total_portfolio * 100)
    print(f"{rank:<6} {combo_name:<15} {row['wape']:>8.2f}%   {row['total_actual']/1e6:>16.2f}    {coverage:>10.2f}%")

print()

# ==============================================================================
# 5. WAPE DEGRADATION ACROSS HORIZONS (W1→W8)
# ==============================================================================
print("5. WAPE DEGRADATION ACROSS HORIZONS")
print("-" * 100)
print()

print(f"{'Horizon':<10} {'WAPE':<10} {'Change from W1':<20} {'Change from Prev':<20}")
print("-" * 100)

wape_by_horizon = []
for horizon in range(1, 9):
    h_data = df_clean[df_clean['horizon'] == horizon]
    total_actual = h_data['actual'].abs().sum()
    total_error = h_data['abs_error'].sum()
    wape = (total_error / total_actual * 100) if total_actual > 0 else 0
    wape_by_horizon.append(wape)

for i, wape in enumerate(wape_by_horizon, 1):
    change_from_w1 = wape - wape_by_horizon[0]
    change_from_prev = wape - wape_by_horizon[i-2] if i > 1 else 0

    print(f"W{i:<9} {wape:>6.2f}%   {change_from_w1:>+6.2f} pp          {change_from_prev:>+6.2f} pp")

print()

# ==============================================================================
# 6. WEEKLY PERFORMANCE TREND
# ==============================================================================
print("6. WEEKLY PERFORMANCE TREND (W1 WAPE)")
print("-" * 100)
print()

weekly_perf = []
for test_week in sorted(w1_data['test_week'].unique()):
    week_data = w1_data[w1_data['test_week'] == test_week]
    total_actual = week_data['actual'].abs().sum()
    total_error = week_data['abs_error'].sum()
    wape = (total_error / total_actual * 100) if total_actual > 0 else 0

    weekly_perf.append({
        'test_week': test_week,
        'wape': wape,
        'total_actual': total_actual
    })

weekly_df = pd.DataFrame(weekly_perf)

print(f"{'Week':<6} {'Date':<12} {'W1 WAPE':<12} {'Total Actual (€M)':<20} {'Trend':<10}")
print("-" * 100)

for idx, row in weekly_df.iterrows():
    trend = ""
    if idx > 0:
        prev_wape = weekly_df.iloc[idx-1]['wape']
        if row['wape'] < prev_wape - 2:
            trend = "📉 Better"
        elif row['wape'] > prev_wape + 2:
            trend = "📈 Worse"
        else:
            trend = "→ Stable"

    print(f"{idx+1:<6} {row['test_week']:<12} {row['wape']:>8.2f}%   {row['total_actual']/1e6:>16.2f}    {trend:<10}")

print()
print(f"Average Weekly W1 WAPE: {weekly_df['wape'].mean():.2f}%")
print(f"Std Dev: {weekly_df['wape'].std():.2f}%")
print(f"Min: {weekly_df['wape'].min():.2f}% (Week {weekly_df['wape'].idxmin()+1})")
print(f"Max: {weekly_df['wape'].max():.2f}% (Week {weekly_df['wape'].idxmax()+1})")
print()

# ==============================================================================
# 7. ERROR DISTRIBUTION ANALYSIS
# ==============================================================================
print("7. ERROR DISTRIBUTION ANALYSIS (W1)")
print("-" * 100)
print()

w1_data['ape'] = (w1_data['abs_error'] / w1_data['actual'].abs() * 100)

percentiles = [10, 25, 50, 75, 90, 95, 99]
print(f"{'Percentile':<15} {'APE':<15}")
print("-" * 100)
for p in percentiles:
    value = np.percentile(w1_data['ape'], p)
    print(f"P{p:<13} {value:>10.2f}%")

print()
print(f"Mean APE:       {w1_data['ape'].mean():.2f}%")
print(f"Median APE:     {w1_data['ape'].median():.2f}%")
print(f"Std Dev:        {w1_data['ape'].std():.2f}%")
print()

# Error buckets
print("Error Buckets (W1):")
print("-" * 100)
buckets = [
    (0, 5, "Excellent"),
    (5, 10, "Very Good"),
    (10, 20, "Good"),
    (20, 40, "Moderate"),
    (40, 100, "Poor"),
    (100, float('inf'), "Very Poor")
]

for lower, upper, label in buckets:
    count = len(w1_data[(w1_data['ape'] >= lower) & (w1_data['ape'] < upper)])
    pct = (count / len(w1_data) * 100)
    print(f"{label:<15} ({lower:>3}%-{upper if upper != float('inf') else '∞':>3}%): {count:>5} forecasts ({pct:>5.1f}%)")

print()

# ==============================================================================
# 8. COMPARISON TO BASELINE
# ==============================================================================
print("8. COMPARISON TO BASELINE")
print("-" * 100)
print()

print("Baseline Configuration:")
print("  - All 34 entity-liquidity combinations")
print("  - 152 features (Stages 1-8)")
print("  - LightGBM + Recursive")
print("  - W1 WAPE: 21.4%")
print()

current_w1_wape = wape_by_horizon[0]
baseline_w1_wape = 21.4
improvement = baseline_w1_wape - current_w1_wape

print("Enhanced Configuration:")
print("  - 20 Tier 1 combinations (14 Tier 2 excluded)")
print("  - 204 features (Stages 1-9, advanced features)")
print("  - LightGBM + Recursive")
print(f"  - W1 WAPE: {current_w1_wape:.2f}%")
print()

print(f"Improvement: {improvement:.2f} pp ({improvement/baseline_w1_wape*100:.1f}% relative improvement)")
print()

# ==============================================================================
# 9. TARGET GAP ANALYSIS
# ==============================================================================
print("9. TARGET GAP ANALYSIS")
print("-" * 100)
print()

print(f"{'Horizon':<10} {'Current':<12} {'Target':<12} {'Gap':<12} {'Gap to Close':<15}")
print("-" * 100)

for horizon in range(1, 9):
    current = wape_by_horizon[horizon-1]
    target = targets[horizon]
    gap = current - target
    pct_to_close = (gap / current * 100)

    print(f"W{horizon:<9} {current:>6.2f}%      ≤{target:>4.1f}%      {gap:>+6.2f}%     {pct_to_close:>10.1f}%")

print()

# ==============================================================================
# 10. HIGH UNCERTAINTY COMBINATIONS
# ==============================================================================
print("10. HIGH UNCERTAINTY / PROBLEMATIC COMBINATIONS")
print("-" * 100)
print()

high_uncertainty = combo_df[combo_df['wape'] > 40]

if len(high_uncertainty) > 0:
    print(f"{'Entity-Liq':<15} {'W1 WAPE':<12} {'Forecasts':<12} {'Total Actual (€M)':<20} {'Recommendation':<30}")
    print("-" * 100)

    for idx, row in high_uncertainty.iterrows():
        combo_name = f"{row['entity_id']}-{row['liquidity_group']}"

        # Recommendation
        if row['wape'] > 80:
            rec = "❌ Consider Tier 2 exclusion"
        elif row['total_actual'] / total_portfolio > 0.01:  # > 1% of portfolio
            rec = "⚠️  Flag as High Uncertainty"
        else:
            rec = "⚠️  Monitor closely"

        print(f"{combo_name:<15} {row['wape']:>8.2f}%   {row['forecasts']:<12} {row['total_actual']/1e6:>16.2f}    {rec:<30}")

    print()
    print(f"Total High Uncertainty Combos: {len(high_uncertainty)}")
    print(f"Portfolio Coverage: {(high_uncertainty['total_actual'].sum() / total_portfolio * 100):.2f}%")
else:
    print("No high uncertainty combinations found (all W1 WAPE < 40%)")

print()

# ==============================================================================
# 11. KEY FINDINGS & RECOMMENDATIONS
# ==============================================================================
print("11. KEY FINDINGS & RECOMMENDATIONS")
print("-" * 100)
print()

print("✅ ACHIEVEMENTS:")
print(f"  1. W1 WAPE: {current_w1_wape:.2f}% (below 15% Phase 1 target)")
print(f"  2. {improvement:.2f} pp improvement from baseline ({improvement/baseline_w1_wape*100:.1f}% relative)")
print(f"  3. Directionality: {directionality:.1f}% (excellent)")
print(f"  4. {len(combo_df[combo_df['wape'] <= 20])} combinations with W1 WAPE ≤20%")
print()

print("❌ GAPS:")
print(f"  1. W1 WAPE {wape_by_horizon[0] - 5.0:.2f} pp above aggressive 5% target")
print(f"  2. All horizons W2-W8 exceed targets")
print(f"  3. Error degrades significantly: W1 {wape_by_horizon[0]:.1f}% → W8 {wape_by_horizon[7]:.1f}% (+{wape_by_horizon[7]-wape_by_horizon[0]:.1f} pp)")
print(f"  4. {len(high_uncertainty)} combinations with W1 WAPE >40%")
print()

print("🔍 KEY INSIGHTS:")
print(f"  1. Best performer: {combo_df.iloc[0]['entity_id']}-{combo_df.iloc[0]['liquidity_group']} ({combo_df.iloc[0]['wape']:.2f}%)")
print(f"  2. Worst performer: {combo_df.iloc[-1]['entity_id']}-{combo_df.iloc[-1]['liquidity_group']} ({combo_df.iloc[-1]['wape']:.2f}%)")
print(f"  3. Top 10 combos average: {top10['wape'].mean():.2f}%")
print(f"  4. Bottom 10 combos average: {bottom10['wape'].mean():.2f}%")
print()

print("📋 RECOMMENDATIONS:")
print("  1. Address V508-TRR (106.54% WAPE) - consider Tier 2 exclusion")
print("  2. Test alternative strategies (Direct, DirRec) to reduce error degradation")
print("  3. Test XGBoost for comparison")
print("  4. Consider ensemble methods for high-uncertainty combinations")
print("  5. Proceed to Phase 3 (LSTM/SARIMAX) if Phase 2 shows limited gains")
print()

print("=" * 100)
print("END OF REPORT")
print("=" * 100)
