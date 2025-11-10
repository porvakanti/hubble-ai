"""
Modeling Package

Contains all forecasting models and strategies:
- Baselines (Naïve, LP-as-is, SES)
- Learners (LightGBM, XGBoost, LSTM, SARIMAX)
- Multi-horizon strategies (Recursive, Direct, DirRec)
- Ensemble (Stacking)
"""

from cf_forecast.modeling.baselines import NaiveForecaster, LPAsIsForecaster, SESForecaster
from cf_forecast.modeling.learners import LightGBMForecaster, XGBoostForecaster, LSTMForecaster, SARIMAXForecaster
from cf_forecast.modeling.multi_horizon import RecursiveStrategy, DirectStrategy, DirRecStrategy
from cf_forecast.modeling.ensemble import StackedEnsemble

__all__ = [
    "NaiveForecaster",
    "LPAsIsForecaster",
    "SESForecaster",
    "LightGBMForecaster",
    "XGBoostForecaster",
    "LSTMForecaster",
    "SARIMAXForecaster",
    "RecursiveStrategy",
    "DirectStrategy",
    "DirRecStrategy",
    "StackedEnsemble",
]
