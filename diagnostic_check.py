"""
Phase 1 Diagnostic: Understanding Current Data Flow
Objective: Identify why actuals are showing as None for Aug 2025
"""
import pandas as pd
from datetime import datetime
from pathlib import Path

print("=" * 80)
print("PHASE 1 DIAGNOSTIC: DATA FLOW ANALYSIS")
print("=" * 80)

# Configuration
asof_date = datetime(2025, 8, 5)  # First week of August 2025
data_dir = Path("data/raw")

print(f"\nTarget forecast date: {asof_date}")
print(f"Expected to forecast: W1-W8 from {asof_date}")

# ============================================================================
# CHECK 1: Raw Actuals Data
# ============================================================================
print("\n" + "=" * 80)
print("CHECK 1: Raw Actuals Data")
print("=" * 80)

actuals_file = data_dir / "actuals_curated.csv"
if actuals_file.exists():
    actuals = pd.read_csv(actuals_file)
    actuals['Value Date'] = pd.to_datetime(actuals['Value Date'])

    print(f"\n✓ File exists: {actuals_file}")
    print(f"  Rows: {len(actuals):,}")
    print(f"  Columns: {list(actuals.columns)}")
    print(f"  Date range: {actuals['Value Date'].min()} to {actuals['Value Date'].max()}")

    # Check August 2025 data
    aug_2025 = actuals[
        (actuals['Value Date'] >= '2025-08-01') &
        (actuals['Value Date'] <= '2025-08-31')
    ]
    print(f"\n  August 2025 data:")
    print(f"    Rows: {len(aug_2025):,}")
    if len(aug_2025) > 0:
        print(f"    Sample:")
        print(aug_2025.head(3).to_string(index=False))
    else:
        print("    ⚠️  NO DATA FOR AUGUST 2025!")

    # Check up to Sep 29, 2025
    up_to_sep = actuals[actuals['Value Date'] <= '2025-09-29']
    print(f"\n  Data up to Sep 29, 2025:")
    print(f"    Rows: {len(up_to_sep):,}")
    print(f"    Latest date: {up_to_sep['Value Date'].max()}")
else:
    print(f"✗ File NOT found: {actuals_file}")

# ============================================================================
# CHECK 2: Weekly Aggregation
# ============================================================================
print("\n" + "=" * 80)
print("CHECK 2: Weekly Aggregation Output")
print("=" * 80)

if actuals_file.exists():
    # Simulate current preprocessing logic
    from src.cf_forecast.preprocessing import DataPreprocessor
    from src.cf_forecast.config import load_config

    config = load_config()
    preprocessor = DataPreprocessor(config)

    # Process actuals to weekly
    weekly = preprocessor.process_actuals_daily_to_weekly(actuals, asof_date=asof_date)

    print(f"\n✓ Weekly aggregation complete")
    print(f"  Rows: {len(weekly):,}")
    print(f"  Columns: {list(weekly.columns)}")
    print(f"  Date range: {weekly['week_start'].min()} to {weekly['week_start'].max()}")

    # Check August 2025 weeks
    aug_weeks = weekly[
        (weekly['week_start'] >= '2025-08-01') &
        (weekly['week_start'] <= '2025-08-31')
    ]
    print(f"\n  August 2025 weeks:")
    print(f"    Rows: {len(aug_weeks):,}")
    if len(aug_weeks) > 0:
        print(f"    Sample:")
        print(aug_weeks.head(5).to_string(index=False))
    else:
        print("    ⚠️  NO WEEKLY DATA FOR AUGUST 2025!")

# ============================================================================
# CHECK 3: Modeling DataFrame
# ============================================================================
print("\n" + "=" * 80)
print("CHECK 3: Modeling DataFrame (after join with LP)")
print("=" * 80)

if actuals_file.exists():
    lp_file = data_dir / "LP_17C7.csv"
    entity_map_file = Path("data/reference/Entity-Liquidity_Map.csv")

    if lp_file.exists() and entity_map_file.exists():
        lp = pd.read_csv(lp_file)
        entity_map = pd.read_csv(entity_map_file)

        # Run full preprocessing
        modeling_df = preprocessor.prepare_modeling_data(
            actuals, lp, entity_map, asof_date
        )

        print(f"\n✓ Modeling DataFrame created")
        print(f"  Rows: {len(modeling_df):,}")
        print(f"  Columns: {list(modeling_df.columns)}")
        if 'week_start' in modeling_df.columns:
            print(f"  Date range: {modeling_df['week_start'].min()} to {modeling_df['week_start'].max()}")

        # Check August 2025 in modeling_df
        if 'week_start' in modeling_df.columns:
            aug_modeling = modeling_df[
                (modeling_df['week_start'] >= '2025-08-01') &
                (modeling_df['week_start'] <= '2025-08-31')
            ]
            print(f"\n  August 2025 in modeling_df:")
            print(f"    Rows: {len(aug_modeling):,}")
            if len(aug_modeling) > 0:
                print(f"    Sample:")
                print(aug_modeling.head(5).to_string(index=False))

                # Check if amount_eur has values
                if 'amount_eur' in aug_modeling.columns:
                    print(f"\n    amount_eur stats:")
                    print(f"      Non-null: {aug_modeling['amount_eur'].notna().sum()}")
                    print(f"      Sum: {aug_modeling['amount_eur'].sum():,.2f}")
                    print(f"      Mean: {aug_modeling['amount_eur'].mean():,.2f}")
            else:
                print("    ⚠️  NO MODELING DATA FOR AUGUST 2025!")
    else:
        print(f"✗ Required files missing:")
        if not lp_file.exists():
            print(f"  - {lp_file}")
        if not entity_map_file.exists():
            print(f"  - {entity_map_file}")

# ============================================================================
# CHECK 4: Forecast Output Format
# ============================================================================
print("\n" + "=" * 80)
print("CHECK 4: Forecast Output Structure")
print("=" * 80)

print("\nExpected forecast structure:")
print("  - week_date column should match modeling_df['week_start']")
print("  - Join keys: ['entity_id', 'liquidity_group', 'week_date']")
print("\nCurrent join logic (from pipeline.py:159-166):")
print("  actuals_map = modeling_df[['entity_id', 'liquidity_group', 'week_start', 'amount_eur']]")
print("  actuals_map.rename(columns={'week_start': 'week_date'})")
print("  forecasts_df.merge(actuals_map, on=['entity_id', 'liquidity_group', 'week_date'])")

print("\n⚠️  KEY QUESTION: Does orchestrator output 'week_date' column that matches?")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print("\nPotential Issues to Investigate:")
print("1. Date range: Does actuals_curated.csv actually have data up to Sep 29, 2025?")
print("2. Weekly aggregation: Are August 2025 weeks being created correctly?")
print("3. Week format: Does orchestrator's 'week_date' match modeling_df's 'week_start'?")
print("4. Join keys: Are entity_id and liquidity_group formats matching?")

print("\n" + "=" * 80)
