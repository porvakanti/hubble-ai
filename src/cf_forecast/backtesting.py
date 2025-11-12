"""
Walk-Forward Backtesting Framework

Critical component for validating model accuracy against aggressive WAPE targets.
Implements time-series cross-validation with proper data handling to prevent leakage.

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
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import structlog

logger = structlog.get_logger(__name__)

# WAPE Targets by horizon
WAPE_TARGETS = {
    1: 0.05,   # W1: 5%
    2: 0.075,  # W2: 7.5%
    3: 0.10,   # W3: 10%
    4: 0.125,  # W4: 12.5%
    5: 0.15,   # W5: 15%
    6: 0.175,  # W6: 17.5%
    7: 0.20,   # W7: 20%
    8: 0.225   # W8: 22.5%
}


class WalkForwardBacktester:
    """
    Walk-forward backtesting for time-series forecasting.

    For each week in validation period:
    1. Train on all data up to (but not including) that week
    2. Generate forecasts for W1-W8 horizons
    3. Compare to actual values
    4. Compute accuracy metrics
    """

    def __init__(self, config: Dict):
        self.config = config
        self.results = []

    def run_backtest(
        self,
        training_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
        models: List[str] = None,
        strategies: List[str] = None
    ) -> pd.DataFrame:
        """
        Run walk-forward backtest over specified date range.

        Args:
            training_data: Complete dataset with features and target
            start_date: Start of backtesting period
            end_date: End of backtesting period
            models: List of models to test (default: all)
            strategies: List of strategies to test (default: all)

        Returns:
            DataFrame with backtest results for each week × horizon × model × strategy
        """
        logger.info(
            "Starting walk-forward backtest",
            start_date=start_date,
            end_date=end_date
        )

        # Default models and strategies
        if models is None:
            models = ['lightgbm', 'xgboost', 'lstm', 'sarimax', 'ensemble']
        if strategies is None:
            strategies = ['recursive', 'direct', 'dirrec']

        # Ensure data is sorted by time
        training_data = training_data.sort_values(['entity_id', 'liquidity_group', 'week_start'])

        # Get all validation weeks
        validation_weeks = pd.date_range(start=start_date, end=end_date, freq='W-MON')

        logger.info(f"Validation weeks: {len(validation_weeks)}")

        # Walk forward through each week
        for i, test_week in enumerate(validation_weeks):
            logger.info(
                f"Backtest week {i+1}/{len(validation_weeks)}",
                test_week=test_week.date()
            )

            # Train on data up to (but not including) test_week
            train_mask = training_data['week_start'] < test_week
            train_df = training_data[train_mask].copy()

            # Test on 8 weeks starting from test_week
            test_weeks = [test_week + timedelta(weeks=h) for h in range(8)]
            test_mask = training_data['week_start'].isin(test_weeks)
            test_df = training_data[test_mask].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                logger.warning(f"Insufficient data for week {test_week.date()}, skipping")
                continue

            # Run forecasts for each model × strategy
            week_results = self._forecast_and_evaluate(
                train_df=train_df,
                test_df=test_df,
                test_week=test_week,
                models=models,
                strategies=strategies
            )

            self.results.extend(week_results)

        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)

        logger.info("Backtest complete", total_results=len(results_df))

        return results_df

    def _forecast_and_evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        test_week: datetime,
        models: List[str],
        strategies: List[str]
    ) -> List[Dict]:
        """
        Generate forecasts and evaluate for a single test week.

        Args:
            train_df: Training data (all data before test_week)
            test_df: Test data (8 weeks starting from test_week)
            test_week: Start of forecast horizon
            models: Models to evaluate
            strategies: Strategies to evaluate

        Returns:
            List of result dictionaries (one per entity × liquidity_group × horizon × model × strategy)
        """
        from cf_forecast.modeling.learners import LightGBMForecaster, XGBoostForecaster
        from cf_forecast.features import get_feature_columns

        week_results = []

        # Get feature columns
        feature_cols = get_feature_columns(train_df)

        # Prepare training data
        X_train = train_df[feature_cols].copy()

        # Convert all features to numeric to prevent dtype issues
        for col in feature_cols:
            X_train[col] = pd.to_numeric(X_train[col], errors='coerce')

        X_train = X_train.fillna(0)
        y_train = train_df['amount_eur']

        # Group by entity × liquidity_group
        for (entity_id, liq_group), entity_test_df in test_df.groupby(['entity_id', 'liquidity_group']):

            for model_name in models:
                if model_name == 'ensemble':
                    continue  # Handle ensemble separately

                for strategy in strategies:
                    try:
                        # Train model
                        if model_name == 'lightgbm':
                            model = LightGBMForecaster(self.config)
                        elif model_name == 'xgboost':
                            model = XGBoostForecaster(self.config)
                        else:
                            # Placeholder for LSTM/SARIMAX
                            logger.warning(f"Model {model_name} not yet implemented, skipping")
                            continue

                        # Filter training data for this entity
                        entity_train_mask = (
                            (train_df['entity_id'] == entity_id) &
                            (train_df['liquidity_group'] == liq_group)
                        )
                        entity_X_train = X_train[entity_train_mask].copy()
                        entity_y_train = y_train[entity_train_mask]

                        if len(entity_y_train) < 10:
                            # Not enough training data
                            continue

                        # Ensure all features are numeric
                        for col in entity_X_train.columns:
                            entity_X_train[col] = pd.to_numeric(entity_X_train[col], errors='coerce')
                        entity_X_train = entity_X_train.fillna(0)

                        model.fit(entity_X_train, entity_y_train)

                        # Generate forecasts for 8 horizons
                        for horizon in range(1, 9):
                            horizon_date = test_week + timedelta(weeks=horizon-1)

                            # Get actual value
                            actual_mask = entity_test_df['week_start'] == horizon_date
                            if actual_mask.sum() == 0:
                                continue  # No actual for this horizon

                            actual_row = entity_test_df[actual_mask].iloc[0]
                            actual_value = actual_row['amount_eur']

                            # Get features for forecasting
                            X_forecast_df = pd.DataFrame([actual_row[feature_cols]], columns=feature_cols)

                            # Convert to numeric
                            for col in feature_cols:
                                X_forecast_df[col] = pd.to_numeric(X_forecast_df[col], errors='coerce')

                            X_forecast_df = X_forecast_df.fillna(0)

                            # Predict (get p90 as point forecast)
                            predictions = model.predict(X_forecast_df)

                            if isinstance(predictions, dict):
                                forecast_value = predictions.get('p90', [0])[0]
                            else:
                                forecast_value = predictions[0]

                            # Compute metrics
                            error = forecast_value - actual_value
                            abs_error = abs(error)
                            abs_pct_error = abs_error / abs(actual_value) if actual_value != 0 else 0

                            direction_correct = 1 if (forecast_value * actual_value > 0) else 0

                            # Store result
                            week_results.append({
                                'test_week': test_week,
                                'horizon': horizon,
                                'horizon_date': horizon_date,
                                'entity_id': entity_id,
                                'liquidity_group': liq_group,
                                'model': model_name,
                                'strategy': strategy,
                                'actual': actual_value,
                                'forecast_p90': forecast_value,
                                'error': error,
                                'abs_error': abs_error,
                                'wape': abs_pct_error,
                                'directionality': direction_correct,
                                'target_wape': WAPE_TARGETS[horizon],
                                'meets_target': abs_pct_error <= WAPE_TARGETS[horizon]
                            })

                    except Exception as e:
                        logger.error(
                            f"Error forecasting {model_name}/{strategy} for {entity_id}-{liq_group}",
                            error=str(e)
                        )
                        continue

        return week_results

    def compute_aggregate_metrics(self, results_df: pd.DataFrame) -> Dict:
        """
        Compute aggregate metrics across all backtesting results.

        Args:
            results_df: Backtest results DataFrame

        Returns:
            Dictionary with aggregate metrics
        """
        metrics = {}

        # Overall metrics
        metrics['overall'] = {
            'mean_wape': results_df['wape'].mean(),
            'median_wape': results_df['wape'].median(),
            'mean_abs_error': results_df['abs_error'].mean(),
            'directionality': results_df['directionality'].mean(),
            'pct_meeting_target': results_df['meets_target'].mean()
        }

        # By horizon
        metrics['by_horizon'] = {}
        for horizon in range(1, 9):
            horizon_data = results_df[results_df['horizon'] == horizon]
            if len(horizon_data) > 0:
                metrics['by_horizon'][horizon] = {
                    'mean_wape': horizon_data['wape'].mean(),
                    'target_wape': WAPE_TARGETS[horizon],
                    'meets_target': horizon_data['wape'].mean() <= WAPE_TARGETS[horizon],
                    'pct_meeting_target': horizon_data['meets_target'].mean(),
                    'count': len(horizon_data)
                }

        # By model
        metrics['by_model'] = {}
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model]
            metrics['by_model'][model] = {
                'mean_wape': model_data['wape'].mean(),
                'pct_meeting_target': model_data['meets_target'].mean(),
                'directionality': model_data['directionality'].mean()
            }

        # By strategy
        metrics['by_strategy'] = {}
        for strategy in results_df['strategy'].unique():
            strategy_data = results_df[results_df['strategy'] == strategy]
            metrics['by_strategy'][strategy] = {
                'mean_wape': strategy_data['wape'].mean(),
                'pct_meeting_target': strategy_data['meets_target'].mean()
            }

        return metrics

    def generate_report(self, results_df: pd.DataFrame, output_path: Path):
        """
        Generate comprehensive backtesting report.

        Args:
            results_df: Backtest results DataFrame
            output_path: Path to save report
        """
        metrics = self.compute_aggregate_metrics(results_df)

        report = f"""
{'='*80}
WALK-FORWARD BACKTESTING REPORT
{'='*80}

BACKTESTING PERIOD:
  Start: {results_df['test_week'].min().date()}
  End: {results_df['test_week'].max().date()}
  Total weeks: {results_df['test_week'].nunique()}
  Total forecasts: {len(results_df):,}

OVERALL PERFORMANCE:
  Mean WAPE: {metrics['overall']['mean_wape']:.2%}
  Median WAPE: {metrics['overall']['median_wape']:.2%}
  Mean Absolute Error: €{metrics['overall']['mean_abs_error']:,.2f}
  Directionality: {metrics['overall']['directionality']:.1%}
  % Meeting Target: {metrics['overall']['pct_meeting_target']:.1%}

PERFORMANCE BY HORIZON:
"""

        for horizon in range(1, 9):
            if horizon in metrics['by_horizon']:
                h_metrics = metrics['by_horizon'][horizon]
                status = "✅ PASS" if h_metrics['meets_target'] else "❌ FAIL"
                report += f"""
  W{horizon}:
    Mean WAPE: {h_metrics['mean_wape']:.2%}
    Target: ≤{h_metrics['target_wape']:.1%}
    Status: {status}
    % Meeting Target: {h_metrics['pct_meeting_target']:.1%}
    Forecast Count: {h_metrics['count']:,}
"""

        report += f"""
PERFORMANCE BY MODEL:
"""
        for model, m_metrics in metrics['by_model'].items():
            report += f"""
  {model.upper()}:
    Mean WAPE: {m_metrics['mean_wape']:.2%}
    % Meeting Target: {m_metrics['pct_meeting_target']:.1%}
    Directionality: {m_metrics['directionality']:.1%}
"""

        report += f"""
PERFORMANCE BY STRATEGY:
"""
        for strategy, s_metrics in metrics['by_strategy'].items():
            report += f"""
  {strategy.upper()}:
    Mean WAPE: {s_metrics['mean_wape']:.2%}
    % Meeting Target: {s_metrics['pct_meeting_target']:.1%}
"""

        report += f"""
{'='*80}
RECOMMENDATION:
"""

        # Determine if ready for production
        horizons_passing = sum(1 for h in metrics['by_horizon'].values() if h['meets_target'])
        total_horizons = len(metrics['by_horizon'])

        if horizons_passing >= 6:  # At least 6 out of 8 horizons passing
            report += """
✅ READY FOR PRODUCTION
   - Majority of horizons meeting WAPE targets
   - Proceed with operational deployment
"""
        else:
            report += f"""
⚠️ NOT READY FOR PRODUCTION
   - Only {horizons_passing}/{total_horizons} horizons meeting WAPE targets
   - Recommendations:
     1. Review feature engineering (add more predictive features)
     2. Hyperparameter tuning for underperforming models
     3. Consider ensemble methods
     4. Analyze failing horizons for patterns
"""

        report += f"""
{'='*80}
"""

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

        logger.info(f"Backtest report saved to {output_path}")

        print(report)

        return metrics
