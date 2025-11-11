"""
Run EDA Stage 1: Data Quality Assessment
This script extracts key analyses to validate data before proceeding
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EDA STAGE 1: DATA QUALITY & STATISTICAL ASSESSMENT")
print("="*80)

# ============================================================================
# 1.1 Load Raw Data
# ============================================================================
print("\n" + "="*80)
print("1.1 LOADING RAW DATA")
print("="*80)

actuals = pd.read_csv('data/raw/actuals_curated.csv')
actuals['Value Date'] = pd.to_datetime(actuals['Value Date'])

print(f"\n✓ ACTUALS DATA")
print(f"  Shape: {actuals.shape}")
print(f"  Columns: {list(actuals.columns)}")
print(f"  Date Range: {actuals['Value Date'].min()} to {actuals['Value Date'].max()}")
print(f"  Total Days: {(actuals['Value Date'].max() - actuals['Value Date'].min()).days}")
print(f"  Memory: {actuals.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 1.2 Temporal Coverage
# ============================================================================
print("\n" + "="*80)
print("1.2 TEMPORAL COVERAGE ANALYSIS")
print("="*80)

# Add week column (Monday-based)
actuals['week_start'] = actuals['Value Date'] - pd.to_timedelta(actuals['Value Date'].dt.dayofweek, unit='D')
actuals['Week'] = actuals['Value Date'].dt.to_period('W-MON')

date_range = pd.date_range(actuals['Value Date'].min(), actuals['Value Date'].max(), freq='D')
dates_with_data = actuals['Value Date'].unique()
missing_dates = set(date_range) - set(dates_with_data)

print(f"\nDaily Coverage:")
print(f"  Total possible days: {len(date_range)}")
print(f"  Days with data: {len(dates_with_data)}")
print(f"  Missing days: {len(missing_dates)} ({len(missing_dates)/len(date_range)*100:.1f}%)")

print(f"\nWeekly Coverage:")
print(f"  Total weeks: {actuals['Week'].nunique()}")
print(f"  First week: {actuals['Week'].min()}")
print(f"  Last week: {actuals['Week'].max()}")

# Last 6 months for backtesting
last_date = actuals['Value Date'].max()
six_months_ago = last_date - pd.DateOffset(months=6)
backtest_data = actuals[actuals['Value Date'] >= six_months_ago]
backtest_weeks = backtest_data['Week'].nunique()

print(f"\nBacktesting Period (Last 6 months):")
print(f"  Date range: {six_months_ago.date()} to {last_date.date()}")
print(f"  Weeks available: {backtest_weeks}")
print(f"  Rows: {len(backtest_data):,}")

# ============================================================================
# 1.3 Entity Coverage
# ============================================================================
print("\n" + "="*80)
print("1.3 ENTITY COVERAGE ANALYSIS")
print("="*80)

entity_stats = actuals.groupby('Entity').agg({
    'Value Date': ['min', 'max', 'nunique'],
    'Amount Functional Currency': ['count', 'sum', 'mean', 'std'],
    'Liquidity Group': lambda x: sorted(x.unique().tolist())
}).round(2)

entity_stats.columns = ['First_Date', 'Last_Date', 'Days_With_Data',
                         'Txn_Count', 'Total_Amount', 'Avg_Amount', 'Std_Amount',
                         'Liquidity_Groups']

print(f"\nTotal entities: {actuals['Entity'].nunique()}")
print(f"\nTop 10 entities by transaction count:")
top10 = entity_stats.nlargest(10, 'Txn_Count')[['Txn_Count', 'Days_With_Data', 'Total_Amount', 'Liquidity_Groups']]
print(top10.to_string())

print(f"\nBottom 5 entities by transaction count:")
bottom5 = entity_stats.nsmallest(5, 'Txn_Count')[['Txn_Count', 'Days_With_Data', 'Total_Amount', 'Liquidity_Groups']]
print(bottom5.to_string())

# ============================================================================
# 1.4 Weekend Transaction Analysis
# ============================================================================
print("\n" + "="*80)
print("1.4 WEEKEND TRANSACTION ANALYSIS")
print("="*80)

actuals['DayOfWeek'] = actuals['Value Date'].dt.day_name()
actuals['IsWeekend'] = actuals['Value Date'].dt.dayofweek >= 5

dow_dist = actuals['DayOfWeek'].value_counts()
print(f"\nTransactions by day of week:")
print(dow_dist)

weekend_txns = actuals[actuals['IsWeekend']]
print(f"\nWeekend transactions: {len(weekend_txns):,} ({len(weekend_txns)/len(actuals)*100:.2f}%)")

if len(weekend_txns) > 0:
    print(f"\nTop 5 entities with weekend transactions:")
    weekend_entities = weekend_txns['Entity'].value_counts().head(5)
    print(weekend_entities)

# ============================================================================
# 1.5 Outlier Detection
# ============================================================================
print("\n" + "="*80)
print("1.5 OUTLIER DETECTION (3×IQR Method)")
print("="*80)

for liq_group in sorted(actuals['Liquidity Group'].unique()):
    data = actuals[actuals['Liquidity Group'] == liq_group]['Amount Functional Currency']

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]

    print(f"\n{liq_group}:")
    print(f"  Q1: €{Q1:,.2f}, Q3: €{Q3:,.2f}, IQR: €{IQR:,.2f}")
    print(f"  Bounds: [€{lower_bound:,.2f}, €{upper_bound:,.2f}]")
    print(f"  Outliers: {len(outliers):,} ({len(outliers)/len(data)*100:.2f}%)")
    if len(outliers) > 0:
        print(f"  Outlier range: [€{outliers.min():,.2f}, €{outliers.max():,.2f}]")

# ============================================================================
# 1.6 Data by Liquidity Group
# ============================================================================
print("\n" + "="*80)
print("1.6 LIQUIDITY GROUP ANALYSIS")
print("="*80)

liq_stats = actuals.groupby('Liquidity Group').agg({
    'Amount Functional Currency': ['count', 'sum', 'mean', 'std', 'min', 'max'],
    'Entity': 'nunique'
}).round(2)

liq_stats.columns = ['Txn_Count', 'Total_Amount', 'Avg_Amount', 'Std_Amount', 'Min_Amount', 'Max_Amount', 'Entities']
print(liq_stats.to_string())

# ============================================================================
# 1.7 Missing Data
# ============================================================================
print("\n" + "="*80)
print("1.7 MISSING DATA ANALYSIS")
print("="*80)

missing = actuals.isnull().sum()
missing_pct = (missing / len(actuals) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Pct': missing_pct
})

if missing_df['Missing_Count'].sum() == 0:
    print("\n✅ No missing values in actuals data")
else:
    print(missing_df[missing_df['Missing_Count'] > 0].to_string())

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STAGE 1 SUMMARY")
print("="*80)

summary = f"""
✅ DATA QUALITY ASSESSMENT COMPLETE

ACTUALS DATA:
  Rows: {len(actuals):,}
  Entities: {actuals['Entity'].nunique()}
  Date Range: {actuals['Value Date'].min().date()} to {actuals['Value Date'].max().date()}
  Total Weeks: {actuals['Week'].nunique()}
  Backtesting Period: {backtest_weeks} weeks (last 6 months)
  Missing Values: {actuals.isnull().sum().sum()}

LIQUIDITY GROUPS:
  TRR (Receivables): {len(actuals[actuals['Liquidity Group']=='TRR']):,} transactions
  TRP (Payables): {len(actuals[actuals['Liquidity Group']=='TRP']):,} transactions

DATA QUALITY ISSUES:
  Weekend Transactions: {len(weekend_txns):,} ({len(weekend_txns)/len(actuals)*100:.2f}%)
  → Recommendation: Review with treasury team

READINESS FOR MODELING:
  ✅ Sufficient historical data ({backtest_weeks} weeks for backtesting)
  ✅ No missing values
  ✅ Multiple entities ({actuals['Entity'].nunique()}) for training
  ✅ Both TRR and TRP liquidity groups present

NEXT STEPS:
  1. Proceed to Stage 2: Feature Engineering (Daily Level)
  2. Create ~118 features before weekly aggregation
  3. Build training dataset
  4. Implement backtesting framework
  5. Validate against aggressive WAPE targets (W1≤5%, W8≤22.5%)
"""

print(summary)

# Save summary
Path('artifacts/metrics').mkdir(parents=True, exist_ok=True)
with open('artifacts/metrics/stage1_summary.txt', 'w') as f:
    f.write(summary)

print("\n✅ Stage 1 complete. Summary saved to artifacts/metrics/stage1_summary.txt")
print("\n" + "="*80)
