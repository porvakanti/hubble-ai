"""
Enhanced Backtest with Advanced Features

Tests:
- Models: LightGBM, XGBoost
- Strategies: Recursive, Direct, DirRec
- Data: Enhanced features (204 total) + Tier 1 only (20 combinations)
- Output: Granular W1-W8 WAPE by entity-liq-model-strategy

Expected improvements:
- Baseline (Tier 1 only): 18.1% W1 WAPE
- Target: 14-15% W1 WAPE after enhancements
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
print("ENHANCED WALK-FORWARD BACKTESTING")
print("="*80)
print()

# Create config
config = {
    'target_column': 'amount_eur',
    'entity_column': 'entity_id',
    'liquidity_group_column': 'liquidity_group',
    'week_column': 'week_start'
}

# Load enhanced training data (Tier 1 only + 204 features)
print("Loading enhanced training data...")
training_data = pd.read_csv('data/intermediate/training_data_enhanced.csv')
training_data['week_start'] = pd.to_datetime(training_data['week_start'])

# Convert all features to numeric
print("Converting features to numeric...")
metadata_cols = ['entity_id', 'liquidity_group', 'week_start']
feature_cols = [col for col in training_data.columns if col not in metadata_cols and col != 'amount_eur']

for col in feature_cols:
    training_data[col] = pd.to_numeric(training_data[col], errors='coerce')

training_data[feature_cols] = training_data[feature_cols].fillna(0)
training_data['amount_eur'] = pd.to_numeric(training_data['amount_eur'], errors='coerce')

print(f"  Rows: {len(training_data):,}")
print(f"  Features: {len(feature_cols)} (all numeric)")
print(f"  Entities: {training_data['entity_id'].nunique()}")
print(f"  Entity-liq combinations: {training_data.groupby(['entity_id', 'liquidity_group']).ngroups}")
print(f"  Date range: {training_data['week_start'].min().date()} to {training_data['week_start'].max().date()}")
print()

# Define backtesting period (last 6 months)
end_date = training_data['week_start'].max()
start_date = end_date - pd.DateOffset(months=6)

print(f"Backtesting period:")
print(f"  Start: {start_date.date()}")
print(f"  End: {end_date.date()}")
print(f"  Duration: ~6 months (27 weeks)")
print()

# Initialize backtester
backtester = WalkForwardBacktester(config)

# Run backtest matrix
print(f"Running walk-forward backtest matrix...")
print(f"  Models: LightGBM, XGBoost")
print(f"  Strategies: Recursive, Direct, DirRec")
print(f"  Horizons: W1-W8")
print(f"  Total configurations: 2 models × 3 strategies = 6")
print()
print(f"⚠️  This will take 2-3 hours depending on system...")
print()

# Strategy 1: Recursive
print("=" * 80)
print("RUNNING: RECURSIVE STRATEGY")
print("=" * 80)
results_recursive = backtester.run_backtest(
    training_data=training_data,
    start_date=start_date,
    end_date=end_date,
    models=['lightgbm', 'xgboost'],
    strategies=['recursive']
)

# Save recursive results
output_dir = Path('artifacts/backtesting/enhanced')
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
recursive_path = output_dir / f'backtest_recursive_{timestamp}.csv'
results_recursive.to_csv(recursive_path, index=False)
print(f"\n✓ Recursive results saved: {recursive_path}")
print()

# Strategy 2: Direct
print("=" * 80)
print("RUNNING: DIRECT STRATEGY")
print("=" * 80)
results_direct = backtester.run_backtest(
    training_data=training_data,
    start_date=start_date,
    end_date=end_date,
    models=['lightgbm', 'xgboost'],
    strategies=['direct']
)

direct_path = output_dir / f'backtest_direct_{timestamp}.csv'
results_direct.to_csv(direct_path, index=False)
print(f"\n✓ Direct results saved: {direct_path}")
print()

# Strategy 3: DirRec
print("=" * 80)
print("RUNNING: DIRREC STRATEGY")
print("=" * 80)
results_dirrec = backtester.run_backtest(
    training_data=training_data,
    start_date=start_date,
    end_date=end_date,
    models=['lightgbm', 'xgboost'],
    strategies=['dirrec']
)

dirrec_path = output_dir / f'backtest_dirrec_{timestamp}.csv'
results_dirrec.to_csv(dirrec_path, index=False)
print(f"\n✓ DirRec results saved: {dirrec_path}")
print()

# Combine all results
print("=" * 80)
print("COMBINING RESULTS")
print("=" * 80)

results_all = pd.concat([results_recursive, results_direct, results_dirrec], ignore_index=True)
all_path = output_dir / f'backtest_all_strategies_{timestamp}.csv'
results_all.to_csv(all_path, index=False)
print(f"✓ Combined results saved: {all_path}")
print(f"  Total forecasts: {len(results_all):,}")
print()

# Generate comprehensive report
print("=" * 80)
print("GENERATING COMPREHENSIVE REPORT")
print("=" * 80)

# Clean data (|actual| >= 1000)
results_clean = results_all[results_all['actual'].abs() >= 1000].copy()
print(f"Clean forecasts (|actual| >= €1,000): {len(results_clean):,}")
print()

# Portfolio WAPE by Model × Strategy × Horizon
print("PORTFOLIO WAPE BY MODEL × STRATEGY × HORIZON")
print("-" * 80)

summary_rows = []
for model in ['lightgbm', 'xgboost']:
    for strategy in ['recursive', 'direct', 'dirrec']:
        subset = results_clean[(results_clean['model'] == model) & (results_clean['strategy'] == strategy)]

        if len(subset) == 0:
            continue

        row = {
            'model': model,
            'strategy': strategy,
            'forecasts': len(subset)
        }

        # Calculate WAPE for each horizon
        for horizon in range(1, 9):
            h_data = subset[subset['horizon'] == horizon]
            if len(h_data) > 0:
                total_actual = h_data['actual'].abs().sum()
                total_error = h_data['abs_error'].sum()
                wape = (total_error / total_actual * 100) if total_actual > 0 else 0
                row[f'W{horizon}_WAPE'] = wape
            else:
                row[f'W{horizon}_WAPE'] = np.nan

        # Average across all horizons
        wape_values = [row[f'W{h}_WAPE'] for h in range(1, 9) if not pd.isna(row.get(f'W{h}_WAPE'))]
        row['avg_WAPE'] = np.mean(wape_values) if wape_values else np.nan

        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

# Format and print
for _, row in summary_df.iterrows():
    print(f"\n{row['model'].upper()} - {row['strategy'].upper()}")
    print(f"  Forecasts: {row['forecasts']:,}")
    wape_str = "  WAPE: "
    for h in range(1, 9):
        wape_val = row.get(f'W{h}_WAPE', np.nan)
        if not pd.isna(wape_val):
            wape_str += f"W{h}={wape_val:.1f}% "
    print(wape_str)
    print(f"  Average: {row['avg_WAPE']:.1f}%")

print()

# Save summary
summary_path = output_dir / f'backtest_summary_{timestamp}.csv'
summary_df.to_csv(summary_path, index=False)
print(f"✓ Summary saved: {summary_path}")
print()

# Entity-level granular report
print("GENERATING ENTITY-LEVEL GRANULAR REPORT")
print("-" * 80)

entity_rows = []
for (entity, liq, model, strategy), group in results_clean.groupby(['entity_id', 'liquidity_group', 'model', 'strategy']):
    row = {
        'entity_id': entity,
        'liquidity_group': liq,
        'model': model,
        'strategy': strategy,
        'forecasts': len(group)
    }

    # WAPE by horizon
    for horizon in range(1, 9):
        h_data = group[group['horizon'] == horizon]
        if len(h_data) > 0:
            total_actual = h_data['actual'].abs().sum()
            total_error = h_data['abs_error'].sum()
            wape = (total_error / total_actual * 100) if total_actual > 0 else 0
            row[f'W{horizon}_WAPE'] = round(wape, 2)
        else:
            row[f'W{horizon}_WAPE'] = np.nan

    # Average
    wape_values = [row[f'W{h}_WAPE'] for h in range(1, 9) if not pd.isna(row.get(f'W{h}_WAPE'))]
    row['avg_W1_W8_WAPE'] = round(np.mean(wape_values), 2) if wape_values else np.nan

    # Directionality
    row['directionality'] = group['directionality'].mean()

    entity_rows.append(row)

entity_df = pd.DataFrame(entity_rows)
entity_path = output_dir / f'backtest_entity_granular_{timestamp}.csv'
entity_df.to_csv(entity_path, index=False)

print(f"✓ Entity-level granular report saved: {entity_path}")
print(f"  Rows: {len(entity_df):,} (entity × liq_group × model × strategy combinations)")
print()

# Best configuration
print("=" * 80)
print("BEST CONFIGURATION")
print("=" * 80)

best_config = summary_df.loc[summary_df['avg_WAPE'].idxmin()]
print(f"Best Model-Strategy: {best_config['model'].upper()} - {best_config['strategy'].upper()}")
print(f"  Average WAPE: {best_config['avg_WAPE']:.1f}%")
print(f"  W1 WAPE: {best_config['W1_WAPE']:.1f}%")
print()

# Comparison to baseline
print("COMPARISON TO BASELINE")
print("-" * 80)
print("Baseline (Stage 8, all 34 combinations, recursive):")
print("  W1 WAPE: 21.4%")
print()
print("After Tier 2 exclusions (theoretical):")
print("  W1 WAPE: 18.1% (-3.3 pp)")
print()
print(f"Current (enhanced features + Tier 1 only):")
print(f"  W1 WAPE: {best_config['W1_WAPE']:.1f}%")
improvement = 21.4 - best_config['W1_WAPE']
print(f"  Improvement: {improvement:.1f} pp from baseline")
print()

# Check if target met
target_w1 = 14.0  # Top of Phase 1 target range (14-15%)
if best_config['W1_WAPE'] <= target_w1:
    print(f"✅ PHASE 1 TARGET MET: {best_config['W1_WAPE']:.1f}% ≤ {target_w1}%")
else:
    gap = best_config['W1_WAPE'] - target_w1
    print(f"⚠️ PHASE 1 TARGET NOT YET MET: {best_config['W1_WAPE']:.1f}% vs {target_w1}% target (gap: +{gap:.1f} pp)")
print()

print("=" * 80)
print("BACKTEST MATRIX COMPLETE")
print("=" * 80)
print()
print("Output files:")
print(f"  - {recursive_path.name}")
print(f"  - {direct_path.name}")
print(f"  - {dirrec_path.name}")
print(f"  - {all_path.name}")
print(f"  - {summary_path.name}")
print(f"  - {entity_path.name}")
print()
print("Next steps:")
print("  1. Review entity-level granular report")
print("  2. Compare strategies (Recursive vs Direct vs DirRec)")
print("  3. Proceed to Phase 2 if W1 WAPE > 15%")
print("  4. Proceed to Phase 3 (LSTM/SARIMAX) if W1 WAPE ≤ 15%")
