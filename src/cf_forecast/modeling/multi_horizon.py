"""Multi-Horizon Forecasting Strategies"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List
import structlog

logger = structlog.get_logger(__name__)


class RecursiveStrategy:
    """Recursive: 1-step model, predictions replace lags for W2-W8."""

    def __init__(self, base_model: Any, horizons: List[int] = list(range(1, 9))):
        self.base_model = base_model
        self.horizons = horizons

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RecursiveStrategy':
        self.base_model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Predict all horizons recursively."""
        predictions = {}
        X_current = X.copy()

        for h in self.horizons:
            # Predict current horizon
            y_pred = self.base_model.predict(X_current)
            predictions[h] = y_pred

            # Update lags for next horizon (shift predictions into lag features)
            if h < max(self.horizons):
                # Shift all lags: Lag_1 becomes Lag_2, etc.
                for lag in range(52, 1, -1):
                    if f'Lag_{lag}' in X_current.columns and f'Lag_{lag-1}' in X_current.columns:
                        X_current[f'Lag_{lag}'] = X_current[f'Lag_{lag-1}']

                # Set Lag_1 to current prediction
                if 'Lag_1' in X_current.columns:
                    X_current['Lag_1'] = y_pred

                # Shift LP features
                for lp_h in range(4, 1, -1):
                    if f'LP_W{lp_h}' in X_current.columns and f'LP_W{lp_h-1}' in X_current.columns:
                        X_current[f'LP_W{lp_h}'] = X_current[f'LP_W{lp_h-1}']

        return predictions


class DirectStrategy:
    """Direct: Separate model for each horizon."""

    def __init__(self, horizons: List[int] = list(range(1, 9))):
        self.horizons = horizons
        self.models = {}

    def fit(self, X_by_horizon: Dict[int, pd.DataFrame], y_by_horizon: Dict[int, pd.Series], model_class: Any) -> 'DirectStrategy':
        """Train separate model for each horizon."""
        for h in self.horizons:
            logger.info(f"Training direct model for horizon {h}")
            self.models[h] = model_class()
            self.models[h].fit(X_by_horizon[h], y_by_horizon[h])
        return self

    def predict(self, X_by_horizon: Dict[int, pd.DataFrame]) -> Dict[int, np.ndarray]:
        predictions = {}
        for h in self.horizons:
            predictions[h] = self.models[h].predict(X_by_horizon[h])
        return predictions


class DirRecStrategy:
    """DirRec: Direct for W1-W4, Recursive for W5-W8."""

    def __init__(self, direct_horizons: List[int], recursive_horizons: List[int]):
        self.direct_horizons = direct_horizons
        self.recursive_horizons = recursive_horizons
        self.direct_models = {}
        self.recursive_model = None

    def fit(self, X_by_horizon: Dict[int, pd.DataFrame], y_by_horizon: Dict[int, pd.Series], model_class: Any) -> 'DirRecStrategy':
        # Train direct models
        for h in self.direct_horizons:
            logger.info(f"Training direct model for horizon {h}")
            self.direct_models[h] = model_class()
            self.direct_models[h].fit(X_by_horizon[h], y_by_horizon[h])

        # Train recursive model (using W1 data)
        if 1 in X_by_horizon and 1 in y_by_horizon:
            self.recursive_model = RecursiveStrategy(model_class(), horizons=self.recursive_horizons)
            self.recursive_model.fit(X_by_horizon[1], y_by_horizon[1])

        return self

    def predict(self, X_by_horizon: Dict[int, pd.DataFrame]) -> Dict[int, np.ndarray]:
        predictions = {}

        # Direct predictions
        for h in self.direct_horizons:
            predictions[h] = self.direct_models[h].predict(X_by_horizon[h])

        # Recursive predictions
        if self.recursive_model and 1 in X_by_horizon:
            recursive_preds = self.recursive_model.predict(X_by_horizon[1])
            for h in self.recursive_horizons:
                predictions[h] = recursive_preds[h]

        return predictions
