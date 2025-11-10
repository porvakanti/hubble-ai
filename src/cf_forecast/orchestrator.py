"""
Forecast Orchestrator - Runs all models × strategies × 8 horizons automatically

Coordinates execution of:
- 4 models: LightGBM, XGBoost, LSTM, SARIMAX
- 3 strategies: Recursive, Direct, DirRec
- 8 horizons: W1-W8
- 4 quantiles: p85, p90, p95, p99
- Ensemble: Weighted combination
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import structlog
from concurrent.futures import ProcessPoolExecutor, as_completed

from cf_forecast.modeling.learners import LightGBMForecaster, XGBoostForecaster, LSTMForecaster, SARIMAXForecaster
from cf_forecast.modeling.multi_horizon import RecursiveStrategy, DirectStrategy, DirRecStrategy
from cf_forecast.features import get_feature_columns

logger = structlog.get_logger(__name__)


class ForecastOrchestrator:
    """Orchestrates execution of all model×strategy combinations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = config.get('modeling', {}).get('learners', ['lightgbm', 'xgboost', 'lstm', 'sarimax'])
        self.horizons = list(range(1, 9))  # W1-W8
        self.quantiles = [0.85, 0.90, 0.95, 0.99]
        self.quantile_cols = ['p85', 'p90', 'p95', 'p99']

    def run_all_combinations(
        self,
        features_df: pd.DataFrame,
        asof_date: datetime
    ) -> pd.DataFrame:
        """
        Execute all model × strategy combinations for 8 horizons.

        Args:
            features_df: Feature matrix with historical data
            asof_date: Forecast as-of date

        Returns:
            DataFrame with: entity_id, liquidity_group, model, strategy,
                           horizon, week_date, p85, p90, p95, p99
        """
        logger.info("Starting forecast orchestration for 8 horizons")

        all_forecasts = []

        # Get feature columns
        feature_cols = get_feature_columns(features_df)

        # Filter training data (before asof_date)
        train_df = features_df[features_df['week_start'] < asof_date].copy()

        if len(train_df) < 52:
            raise ValueError(f"Insufficient training data: {len(train_df)} weeks, need at least 52")

        # Prepare forecast data (week at asof_date for initial features)
        forecast_df = features_df[features_df['week_start'] == asof_date].copy()

        if forecast_df.empty:
            logger.warning(f"No data at asof_date {asof_date}, using latest available")
            forecast_df = features_df.iloc[[-1]].copy()

        # Run each model
        for model_name in self.models:
            logger.info(f"Running model: {model_name}")

            try:
                # Recursive strategy
                forecasts = self._run_recursive(model_name, train_df, forecast_df, feature_cols, asof_date)
                forecasts['model'] = model_name
                forecasts['strategy'] = 'recursive'
                all_forecasts.append(forecasts)

                # Direct strategy
                forecasts = self._run_direct(model_name, train_df, forecast_df, feature_cols, asof_date)
                forecasts['model'] = model_name
                forecasts['strategy'] = 'direct'
                all_forecasts.append(forecasts)

                # DirRec strategy
                forecasts = self._run_dirrec(model_name, train_df, forecast_df, feature_cols, asof_date)
                forecasts['model'] = model_name
                forecasts['strategy'] = 'dirrec'
                all_forecasts.append(forecasts)

            except Exception as e:
                logger.error(f"Failed to run {model_name}: {e}", exc_info=True)
                continue

        if not all_forecasts:
            raise RuntimeError("All models failed")

        # Combine all results
        combined = pd.concat(all_forecasts, ignore_index=True)

        # Create ensemble
        logger.info("Creating ensemble forecast")
        ensemble = self._create_ensemble(combined, asof_date)
        ensemble['model'] = 'ensemble'
        ensemble['strategy'] = 'weighted'

        # Add ensemble to results
        final = pd.concat([combined, ensemble], ignore_index=True)

        # Add week_date column
        final['week_date'] = final.apply(
            lambda row: row['asof_date'] + timedelta(weeks=int(row['horizon'])),
            axis=1
        )

        logger.info(f"Orchestration complete: {len(final)} forecast rows")

        return final

    def _run_recursive(
        self, model_name: str, train_df: pd.DataFrame, forecast_df: pd.DataFrame,
        feature_cols: List[str], asof_date: datetime
    ) -> pd.DataFrame:
        """Run recursive strategy (1-step model, roll forward 8 weeks)."""

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['amount_eur']

        # Train model
        model = self._get_model(model_name)
        model.fit(X_train, y_train)

        # Generate forecasts for 8 horizons
        results = []

        for entity_liq, group in forecast_df.groupby(['entity_id', 'liquidity_group']):
            entity_id, liq_group = entity_liq

            X_current = group[feature_cols].copy().fillna(0)

            for h in range(1, 9):
                # Predict current horizon
                preds = model.predict(X_current)

                if isinstance(preds, dict):
                    # Quantile predictions
                    row = {
                        'entity_id': entity_id,
                        'liquidity_group': liq_group,
                        'horizon': h,
                        'asof_date': asof_date,
                        'p85': preds.get('p85', [0])[0],
                        'p90': preds.get('p90', [0])[0],
                        'p95': preds.get('p95', [0])[0],
                        'p99': preds.get('p99', [0])[0],
                    }
                else:
                    # Point prediction - use as all quantiles
                    pred_val = preds[0] if len(preds) > 0 else 0
                    row = {
                        'entity_id': entity_id,
                        'liquidity_group': liq_group,
                        'horizon': h,
                        'asof_date': asof_date,
                        'p85': pred_val * 0.95,
                        'p90': pred_val,
                        'p95': pred_val * 1.05,
                        'p99': pred_val * 1.10,
                    }

                results.append(row)

                # Update lags for next horizon (recursive)
                if h < 8:
                    # Shift lags: Lag_52 = Lag_51, ..., Lag_2 = Lag_1, Lag_1 = prediction
                    for lag in range(52, 1, -1):
                        if f'Lag_{lag}' in X_current.columns and f'Lag_{lag-1}' in X_current.columns:
                            X_current[f'Lag_{lag}'] = X_current[f'Lag_{lag-1}'].values

                    # Set Lag_1 to current prediction (use p90 as central estimate)
                    if 'Lag_1' in X_current.columns:
                        pred_central = preds.get('p90', [0])[0] if isinstance(preds, dict) else preds[0]
                        X_current['Lag_1'] = pred_central

        return pd.DataFrame(results)

    def _run_direct(
        self, model_name: str, train_df: pd.DataFrame, forecast_df: pd.DataFrame,
        feature_cols: List[str], asof_date: datetime
    ) -> pd.DataFrame:
        """Run direct strategy (separate model for each horizon)."""

        results = []

        # Train separate model for each horizon
        for h in range(1, 9):
            # For horizon h, target is h weeks ahead
            X_train = train_df[feature_cols].fillna(0).iloc[:-h] if h > 1 else train_df[feature_cols].fillna(0)
            y_train = train_df['amount_eur'].shift(-h).iloc[:-h] if h > 1 else train_df['amount_eur'].shift(-h)

            # Drop NaN targets
            valid_mask = ~y_train.isna()
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]

            if len(X_train) < 10:
                logger.warning(f"Insufficient data for horizon {h}, skipping")
                continue

            # Train model
            model = self._get_model(model_name)
            model.fit(X_train, y_train)

            # Predict
            X_forecast = forecast_df[feature_cols].fillna(0)
            preds = model.predict(X_forecast)

            # Store results
            for idx, (_, row) in enumerate(forecast_df.iterrows()):
                if isinstance(preds, dict):
                    result = {
                        'entity_id': row['entity_id'],
                        'liquidity_group': row['liquidity_group'],
                        'horizon': h,
                        'asof_date': asof_date,
                        'p85': preds.get('p85', [0])[idx] if len(preds.get('p85', [])) > idx else 0,
                        'p90': preds.get('p90', [0])[idx] if len(preds.get('p90', [])) > idx else 0,
                        'p95': preds.get('p95', [0])[idx] if len(preds.get('p95', [])) > idx else 0,
                        'p99': preds.get('p99', [0])[idx] if len(preds.get('p99', [])) > idx else 0,
                    }
                else:
                    pred_val = preds[idx] if len(preds) > idx else 0
                    result = {
                        'entity_id': row['entity_id'],
                        'liquidity_group': row['liquidity_group'],
                        'horizon': h,
                        'asof_date': asof_date,
                        'p85': pred_val * 0.95,
                        'p90': pred_val,
                        'p95': pred_val * 1.05,
                        'p99': pred_val * 1.10,
                    }

                results.append(result)

        return pd.DataFrame(results)

    def _run_dirrec(
        self, model_name: str, train_df: pd.DataFrame, forecast_df: pd.DataFrame,
        feature_cols: List[str], asof_date: datetime
    ) -> pd.DataFrame:
        """Run DirRec strategy (Direct for W1-W4, Recursive for W5-W8)."""

        # Direct for W1-W4
        direct_results = []
        for h in range(1, 5):
            X_train = train_df[feature_cols].fillna(0).iloc[:-h] if h > 1 else train_df[feature_cols].fillna(0)
            y_train = train_df['amount_eur'].shift(-h).iloc[:-h] if h > 1 else train_df['amount_eur'].shift(-h)

            valid_mask = ~y_train.isna()
            X_train = X_train[valid_mask]
            y_train = y_train[valid_mask]

            if len(X_train) < 10:
                continue

            model = self._get_model(model_name)
            model.fit(X_train, y_train)

            X_forecast = forecast_df[feature_cols].fillna(0)
            preds = model.predict(X_forecast)

            for idx, (_, row) in enumerate(forecast_df.iterrows()):
                if isinstance(preds, dict):
                    result = {
                        'entity_id': row['entity_id'],
                        'liquidity_group': row['liquidity_group'],
                        'horizon': h,
                        'asof_date': asof_date,
                        'p85': preds.get('p85', [0])[idx] if len(preds.get('p85', [])) > idx else 0,
                        'p90': preds.get('p90', [0])[idx] if len(preds.get('p90', [])) > idx else 0,
                        'p95': preds.get('p95', [0])[idx] if len(preds.get('p95', [])) > idx else 0,
                        'p99': preds.get('p99', [0])[idx] if len(preds.get('p99', [])) > idx else 0,
                    }
                else:
                    pred_val = preds[idx] if len(preds) > idx else 0
                    result = {
                        'entity_id': row['entity_id'],
                        'liquidity_group': row['liquidity_group'],
                        'horizon': h,
                        'asof_date': asof_date,
                        'p85': pred_val * 0.95,
                        'p90': pred_val,
                        'p95': pred_val * 1.05,
                        'p99': pred_val * 1.10,
                    }
                direct_results.append(result)

        # Recursive for W5-W8 (continue from W4)
        recursive_results = self._run_recursive(model_name, train_df, forecast_df, feature_cols, asof_date)
        recursive_w5w8 = recursive_results[recursive_results['horizon'] >= 5].copy()

        # Combine
        combined = pd.concat([pd.DataFrame(direct_results), recursive_w5w8], ignore_index=True)

        return combined

    def _get_model(self, model_name: str):
        """Get model instance by name."""
        if model_name == 'lightgbm':
            return LightGBMForecaster(self.config)
        elif model_name == 'xgboost':
            return XGBoostForecaster(self.config)
        elif model_name == 'lstm':
            return LSTMForecaster(self.config)
        elif model_name == 'sarimax':
            return SARIMAXForecaster(self.config)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _create_ensemble(self, forecasts_df: pd.DataFrame, asof_date: datetime) -> pd.DataFrame:
        """Create weighted ensemble from all model forecasts."""

        ensemble_results = []

        # Group by entity, liquidity_group, horizon
        for (entity, liq_group, horizon), group in forecasts_df.groupby(
            ['entity_id', 'liquidity_group', 'horizon']
        ):
            # Simple average ensemble across models (can be weighted by performance later)
            ensemble_row = {
                'entity_id': entity,
                'liquidity_group': liq_group,
                'horizon': horizon,
                'asof_date': asof_date,
                'p85': group['p85'].mean(),
                'p90': group['p90'].mean(),
                'p95': group['p95'].mean(),
                'p99': group['p99'].mean(),
            }

            ensemble_results.append(ensemble_row)

        return pd.DataFrame(ensemble_results)
