"""ML Learners: LightGBM, XGBoost, LSTM, SARIMAX"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import structlog

logger = structlog.get_logger(__name__)


class LightGBMForecaster:
    """LightGBM with quantile prediction."""

    def __init__(self, config: Dict[str, Any]):
        import lightgbm as lgb
        self.config = config.get("lightgbm", {})
        self.quantiles = config.get("modeling", {}).get("quantiles", [0.85, 0.90, 0.95, 0.99])
        self.models = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LightGBMForecaster':
        import lightgbm as lgb

        # Point prediction (p50)
        params = {**self.config, "objective": "regression", "metric": "mae"}
        self.models["p50"] = lgb.LGBMRegressor(**params)
        self.models["p50"].fit(X, y)

        # Quantile predictions
        for q in self.quantiles:
            params_q = {**self.config, "objective": "quantile", "alpha": q}
            self.models[f"p{int(q*100)}"] = lgb.LGBMRegressor(**params_q)
            self.models[f"p{int(q*100)}"].fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        return predictions


class XGBoostForecaster:
    """XGBoost with quantile prediction."""

    def __init__(self, config: Dict[str, Any]):
        import xgboost as xgb
        self.config = config.get("xgboost", {})
        self.quantiles = config.get("modeling", {}).get("quantiles", [0.85, 0.90, 0.95, 0.99])
        self.models = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostForecaster':
        import xgboost as xgb

        # Point prediction
        params = {**self.config, "objective": "reg:squarederror"}
        self.models["p50"] = xgb.XGBRegressor(**params)
        self.models["p50"].fit(X, y)

        # Quantile predictions
        for q in self.quantiles:
            params_q = {**self.config, "objective": f"reg:quantileerror", "quantile_alpha": q}
            self.models[f"p{int(q*100)}"] = xgb.XGBRegressor(**params_q)
            self.models[f"p{int(q*100)}"].fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        return predictions


class LSTMForecaster:
    """LSTM for time series forecasting."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("lstm", {})
        self.sequence_length = self.config.get("sequence_length", 52)
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LSTMForecaster':
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            logger.warning("TensorFlow not available, LSTM disabled")
            return self

        # Build simple LSTM model
        model = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True, input_shape=(None, X.shape[1])),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64),
            keras.layers.Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")

        # Reshape for LSTM (samples, timesteps, features)
        X_arr = X.values.reshape((len(X), 1, -1))

        model.fit(X_arr, y.values, epochs=50, batch_size=32, verbose=0, validation_split=0.1)

        self.model = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X))
        X_arr = X.values.reshape((len(X), 1, -1))
        return self.model.predict(X_arr, verbose=0).flatten()


class SARIMAXForecaster:
    """SARIMAX with exogenous variables."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("sarimax", {})
        self.order = self.config.get("order", (1, 0, 1))
        self.seasonal_order = self.config.get("seasonal_order", (1, 0, 1, 52))
        self.model_fit = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SARIMAXForecaster':
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            logger.warning("statsmodels not available, SARIMAX disabled")
            return self

        try:
            model = SARIMAX(y, exog=X, order=self.order, seasonal_order=self.seasonal_order)
            self.model_fit = model.fit(disp=False)
        except Exception as e:
            logger.warning(f"SARIMAX fit failed: {e}")
            self.model_fit = None

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_fit is None:
            return np.zeros(len(X))
        try:
            return self.model_fit.forecast(steps=len(X), exog=X)
        except:
            return np.zeros(len(X))
