# Treasury Cash Flow Forecasting - WAPE Accuracy Targets

**Project Goal**: Beat existing LP forecast accuracy with machine learning models

## Target WAPE by Horizon

| Horizon | Target WAPE | Notes |
|---------|-------------|-------|
| W1      | ≤ 5.0%     | Most critical - current week forecast |
| W2      | ≤ 7.5%     | |
| W3      | ≤ 10.0%    | |
| W4      | ≤ 12.5%    | End of typical LP 4-week horizon |
| W5      | ≤ 15.0%    | |
| W6      | ≤ 17.5%    | |
| W7      | ≤ 20.0%    | |
| W8      | ≤ 22.5%    | Longest horizon |

## Context

These are **aggressive targets** requiring:
- ~118 carefully engineered features
- Multiple model types (LightGBM, XGBoost, LSTM, SARIMAX)
- Ensemble methods for robustness
- Hierarchical reconciliation for coherence
- Probabilistic forecasts (p85, p90, p95, p99)

## Baseline Comparison

**LP (Liquidity Plan) Forecasts**: Current treasury forecasts to beat
- Typical WAPE: ~15-25% (to be measured during backtesting)
- Limited horizon: Usually 4 weeks
- Manual adjustments by treasury team

**Our Target**: Consistently beat LP across all 8 weeks while maintaining coherence

## Validation Approach

**Backtesting Period**: Last 6 months of historical data
**Method**: Walk-forward validation
- For each week, train on data ≤ that week
- Forecast W1-W8
- Compare to actuals
- Compute WAPE per horizon
- Track performance vs targets

## Success Criteria

✅ **Phase 1 Success**: Model validation shows WAPE ≤ targets for majority of test weeks
✅ **Phase 2 Success**: Operational forecasts maintain target accuracy over time
✅ **Phase 3 Success**: Consistently beat LP baseline by ≥20% improvement

## Model Selection Criteria

If ensemble doesn't hit targets:
1. Identify which horizons are struggling (W1-W4 vs W5-W8)
2. Consider horizon-specific models (Direct strategy)
3. Feature engineering refinement
4. Consider external features (economic indicators, seasonality)
5. Iterative improvement until targets met

**Note**: These targets are NON-NEGOTIABLE for production deployment.
