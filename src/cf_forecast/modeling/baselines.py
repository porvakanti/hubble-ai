"""Baseline Forecasters"""
import numpy as np
import pandas as pd
from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class NaiveForecaster:
    """Naïve forecast: last observed value."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NaiveForecaster':
        self.last_value_ = y.iloc[-1] if len(y) > 0 else 0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.last_value_)


class LPAsIsForecaster:
    """LP as-is: use liquidity plan values directly."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LPAsIsForecaster':
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Use LP_W1 if available, else 0
        if 'LP_W1' in X.columns:
            return X['LP_W1'].fillna(0).values
        return np.zeros(len(X))


class SESForecaster:
    """Simple Exponential Smoothing."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SESForecaster':
        if len(y) == 0:
            self.level_ = 0
        else:
            level = y.iloc[0]
            for val in y[1:]:
                level = self.alpha * val + (1 - self.alpha) * level
            self.level_ = level
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.level_)
