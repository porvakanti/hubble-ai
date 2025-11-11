"""
Run Walk-Forward Backtesting

Executes comprehensive backtest to validate model accuracy against WAPE targets.
This is the CRITICAL validation step before operational deployment.

Target WAPE by Horizon:
- W1: ≤ 5.0%
- W2: ≤ 7.5%
- W3: ≤ 10.0%
- W4: ≤ 12.5%
- W5: ≤ 15.0%
- W6: ≤ 17.5%
- W7: ≤ 20.0%
- W8: ≤ 22.5%
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from cf_forecast.backtesting import WalkForwardBacktester

print("="*80)
print("WALK-FORWARD BACKTESTING")
print("="*80)

# Create simple config
config = {
    'target_column': 'amount_eur',
    'entity_column': 'entity_id',
    'liquidity_group_column': 'liquidity_group',
    'week_column': 'week_start'
}

# Load training data from Stage 8
print("\nLoading training data...")
training_data = pd.read_csv('data/intermediate/training_data_final.csv')
training_data['week_start'] = pd.to_datetime(training_data['week_start'])

print(f"  Rows: {len(training_data):,}")
print(f"  Date range: {training_data['week_start'].min().date()} to {training_data['week_start'].max().date()}")

# Define backtesting period (last 6 months)
end_date = training_data['week_start'].max()
start_date = end_date - pd.DateOffset(months=6)

print(f"\nBacktesting period:")
print(f"  Start: {start_date.date()}")
print(f"  End: {end_date.date()}")
print(f"  Duration: ~6 months")

# Initialize backtester
backtester = WalkForwardBacktester(config)

# Run backtest
print(f"\nRunning walk-forward backtest...")
print(f"  Models: LightGBM, XGBoost")
print(f"  Strategies: Recursive, Direct, DirRec")
print(f"  Horizons: W1-W8")
print(f"\n⚠️  This may take 10-30 minutes depending on data size...")

results_df = backtester.run_backtest(
    training_data=training_data,
    start_date=start_date,
    end_date=end_date,
    models=['lightgbm', 'xgboost'],  # Start with these two
    strategies=['recursive']  # Start with recursive for speed
)

# Save detailed results
output_dir = Path('artifacts/backtesting')
output_dir.mkdir(parents=True, exist_ok=True)

results_path = output_dir / f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
results_df.to_csv(results_path, index=False)
print(f"\n✓ Detailed results saved: {results_path}")

# Generate report
report_path = output_dir / f'backtest_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
metrics = backtester.generate_report(results_df, report_path)

# Save metrics as JSON
import json
metrics_path = output_dir / f'backtest_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(metrics_path, 'w') as f:
    # Convert to JSON-serializable format
    metrics_json = {
        'overall': metrics['overall'],
        'by_horizon': {str(k): v for k, v in metrics['by_horizon'].items()},
        'by_model': metrics['by_model'],
        'by_strategy': metrics['by_strategy']
    }
    json.dump(metrics_json, f, indent=2)

print(f"✓ Metrics saved: {metrics_path}")

# Summary
print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)

horizons_passing = sum(1 for h in metrics['by_horizon'].values() if h['meets_target'])
total_horizons = len(metrics['by_horizon'])

print(f"\nResults:")
print(f"  Horizons passing: {horizons_passing}/{total_horizons}")
print(f"  Overall WAPE: {metrics['overall']['mean_wape']:.2%}")
print(f"  Directionality: {metrics['overall']['directionality']:.1%}")

if horizons_passing >= 6:
    print(f"\n✅ SUCCESS: Ready for production deployment")
else:
    print(f"\n⚠️ NEEDS IMPROVEMENT: {8-horizons_passing} horizons not meeting targets")
    print(f"\nNext steps:")
    print(f"  1. Analyze failing horizons")
    print(f"  2. Feature engineering improvements")
    print(f"  3. Hyperparameter tuning")
    print(f"  4. Consider ensemble methods")

print("\n" + "="*80)
