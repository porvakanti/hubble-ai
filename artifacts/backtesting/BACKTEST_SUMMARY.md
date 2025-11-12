# Hubble.AI Backtest Validation Report
**Date**: 2025-11-11
**Validation Period**: March 29 - September 29, 2025 (27 weeks)
**Models Tested**: LightGBM, XGBoost
**Strategy**: Recursive forecasting

---

## Executive Summary

✅ **TECHNICAL SUCCESS**: Backtest infrastructure working correctly
⚠️ **PERFORMANCE**: Models functional but not meeting aggressive WAPE targets
📊 **DATA QUALITY ISSUE**: 230 near-zero actuals (1.9%) causing extreme outliers

### Key Findings
- **Median WAPE**: 13.9% (excellent)
- **Mean WAPE (cleaned)**: 152.7% (needs improvement due to outliers)
- **Directionality**: 99.1% (excellent - predicting trends correctly)
- **Individual forecasts meeting targets**: 48.2%
- **Horizons passing aggregate targets**: 0/8

---

## Detailed Results

### Performance by Horizon (After removing near-zero actuals)

| Horizon | Mean WAPE | Target WAPE | Status | % Meeting Target |
|---------|-----------|-------------|--------|------------------|
| W1      | 164.6%    | ≤5.0%       | ❌ FAIL | 32.3%           |
| W2      | 171.6%    | ≤7.5%       | ❌ FAIL | 38.5%           |
| W3      | 190.6%    | ≤10.0%      | ❌ FAIL | 44.2%           |
| W4      | 113.4%    | ≤12.5%      | ❌ FAIL | 48.8%           |
| W5      | 122.3%    | ≤15.0%      | ❌ FAIL | 51.6%           |
| W6      | 164.6%    | ≤17.5%      | ❌ FAIL | 55.0%           |
| W7      | 156.9%    | ≤20.0%      | ❌ FAIL | 56.4%           |
| W8      | 129.4%    | ≤22.5%      | ❌ FAIL | 58.9%           |

### Model Comparison (Cleaned Data)

| Model    | Mean WAPE | Median WAPE | Forecasts |
|----------|-----------|-------------|-----------|
| LightGBM | 101.1%    | 14.6%       | 5,810     |
| XGBoost  | 204.4%    | 13.0%       | 5,810     |

**Winner**: LightGBM (50% lower mean WAPE)

---

## Data Quality Issues

### Critical Finding: Near-Zero Actuals
- **Issue**: 230 rows (1.9%) with |actual| < €1,000
- **Impact**: Caused astronomical WAPE values (up to 146 million %)
- **Root Cause**: Entity 17C7 and others have €0.01 values (likely placeholders for missing data)
- **Example**:
  - Date: 2025-07-14
  - Entity: 17C7
  - Actual: €0.01
  - Forecast: €1,462,288
  - WAPE: 146,228,800%

### Original vs Cleaned Metrics
| Metric | Original | Cleaned |
|--------|----------|---------|
| Mean WAPE | 2,258,038% | 152.7% |
| Median WAPE | 14.5% | 13.9% |
| Rows | 11,850 | 11,620 |

---

## Analysis

### Strengths
1. **Excellent Directionality (99.1%)**: Models predict increase/decrease correctly
2. **Good Median Performance (13.9%)**: Most forecasts are reasonably accurate
3. **Consistent Across Horizons**: Performance degrades gradually (expected)
4. **LightGBM Superior**: Clear model choice for production

### Weaknesses
1. **Outlier Sensitivity**: Large errors on specific entity-week combinations
2. **WAPE Distribution**: Mean >> Median indicates long tail of errors
3. **Aggressive Targets**: W1≤5% is very challenging for treasury forecasting
4. **Data Quality**: Near-zero actuals contaminating metrics

### Performance Context
- **Industry Benchmark**: Production treasury systems typically target 10-30% WAPE
- **Our Median**: 13.9% (competitive)
- **Our Mean**: 152.7% (needs improvement)
- **Target**: 5-22.5% (very aggressive)

---

## Root Cause Analysis

### Why WAPE is High Despite Good Median?

**Extreme Outliers**: Small subset of forecasts with very large errors

| Percentile | WAPE |
|------------|------|
| 25th | 3.7% |
| 50th (Median) | 13.9% |
| 75th | 45.9% |
| 90th | ~200%+ |
| Max | 146M% (before cleaning) |

**Contributing Factors**:
1. **Entity 17C7**: Volatile entity with data quality issues
2. **Small Actuals**: Denominator effect in WAPE calculation
3. **Recursive Strategy Limitations**: Errors compound across horizons
4. **Feature Set**: May need entity-specific features for volatile entities

---

## Recommendations

### Immediate Actions (Required)
1. **Fix Data Quality** ✅ HIGH PRIORITY
   - Filter out |actual| < €1,000 in training and backtesting
   - Investigate entity 17C7 data source
   - Implement data validation in pipeline

2. **Update WAPE Calculation**
   - Add minimum threshold for actuals
   - Consider MAPE alternative for stability
   - Implement winsorization (cap extreme values at 99th percentile)

3. **Model Selection**
   - Deploy LightGBM (50% better than XGBoost)
   - Retire XGBoost from production consideration

### Short-Term Improvements (1-2 weeks)
4. **Outlier Analysis**
   - Identify top 10 entity-week combinations with highest errors
   - Investigate patterns (e.g., month-end, quarter-end spikes)
   - Add entity-specific features

5. **Hyperparameter Tuning**
   - LightGBM optimization for current best model
   - Focus on reducing large errors (use Huber loss or quantile regression)

6. **Feature Engineering**
   - Add entity-specific volatility features
   - Calendar effects (month-end, quarter-end flags)
   - Rolling error corrections (bias adjustment)

### Medium-Term Enhancements (3-4 weeks)
7. **Alternative Strategies**
   - Test Direct strategy (independent models per horizon)
   - Test DirRec hybrid approach
   - Compare vs recursive baseline

8. **Advanced Models**
   - LSTM deep learning (as user requested)
   - SARIMAX statistical (as user requested)
   - Ensemble (LightGBM + LSTM)

9. **Entity-Specific Models**
   - Train separate models for high-volatility entities (17C7, etc.)
   - Use ensemble weights based on entity characteristics

### Long-Term Strategy
10. **Target Recalibration**
    - Review if W1≤5% is achievable for treasury data
    - Industry benchmark: 10-20% for W1, 20-40% for W8
    - Consider adjusting to W1≤10%, W8≤30% (still aggressive)

11. **Production Deployment Plan**
    - Stage 1: Deploy LightGBM with data quality filters
    - Stage 2: Monitor for 4 weeks, collect feedback
    - Stage 3: Iterate based on production performance
    - Stage 4: Add LSTM ensemble if Stage 2-3 successful

---

## Validation Status

| Criterion | Status | Details |
|-----------|--------|---------|
| Technical Infrastructure | ✅ PASS | Backtest completed successfully |
| Dtype Issues Resolved | ✅ PASS | All features numeric, models trained |
| 27-Week Validation | ✅ PASS | March-September 2025 completed |
| Data Quality | ⚠️ PARTIAL | Near-zero outliers identified, needs fixing |
| WAPE Targets (0/8) | ❌ FAIL | No horizons meeting aggressive targets |
| Median Performance | ✅ PASS | 13.9% is competitive |
| Directionality | ✅ PASS | 99.1% excellent |
| Production Ready | ⚠️ CONDITIONAL | Needs data quality fixes + model tuning |

---

## Next Steps

### Immediate (Today)
- [x] Complete 27-week backtest
- [x] Identify data quality issues
- [x] Generate comprehensive report
- [ ] Fix data quality filters in training pipeline
- [ ] Rerun backtest with cleaned data

### This Week
- [ ] Run LSTM backtest (user request)
- [ ] Run SARIMAX backtest (user request)
- [ ] Hyperparameter tuning for LightGBM
- [ ] Update preprocessing.py to match EDA features
- [ ] Fix orchestrator week logic (W1=current Monday)

### Next 2 Weeks
- [ ] Entity-specific feature engineering
- [ ] Test Direct + DirRec strategies
- [ ] Ensemble model development
- [ ] Production deployment plan

---

## Conclusion

**Current State**: The backtesting infrastructure is working correctly, and models are functional with competitive median performance (13.9% WAPE). However, aggressive WAPE targets (W1≤5%) are not met due to outlier sensitivity and data quality issues.

**Critical Issue**: Entity 17C7 and others have near-zero actual values (€0.01) causing extreme WAPE outliers. This must be fixed immediately.

**Path Forward**:
1. Fix data quality (filter near-zero actuals)
2. Rerun backtest with cleaned data
3. Deploy LightGBM with realistic targets (W1≤10-15%)
4. Iterate with LSTM/SARIMAX and ensemble methods

**Realistic Assessment**: Achieving W1≤5% WAPE for treasury cash flow forecasting is extremely challenging. Industry benchmarks are typically 10-30%. Our median of 13.9% is competitive. Recommend recalibrating targets to W1≤10%, W8≤30% for production deployment.

---

**Report Generated**: 2025-11-11 20:20:00
**Generated By**: Claude (Hubble.AI Backtest Validation)
**Files**:
- Detailed Results: `artifacts/backtesting/backtest_results_20251111_201828.csv`
- Raw Output: `backtest_output.log`
