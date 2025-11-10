"""Tests for feature engineering module"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cf_forecast.features import FeatureEngineer, get_feature_columns


@pytest.fixture
def config():
    return {
        "features": {
            "daily_patterns": {"enabled": True},
            "lags": {"enabled": True, "max_lag": 52},
            "rolling_stats": {"enabled": True, "windows": [4, 12, 26]},
            "calendar": {"enabled": True},
            "trend": {"enabled": True},
            "liquidity_plans": {"enabled": True, "horizons": [1, 2, 3, 4]},
            "entity_encoding": {"enabled": True, "top_n": 15},
            "liquidity_group": {"enabled": True}
        }
    }


@pytest.fixture
def feature_engineer(config):
    return FeatureEngineer(config)


@pytest.fixture
def daily_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "entity_id": ["057"] * 100,
        "liquidity_group": ["TRR"] * 100,
        "posting_date": dates,
        "amount_eur": np.random.randn(100) * 1000 + 5000
    })


@pytest.fixture
def weekly_data():
    weeks = pd.date_range("2024-01-01", periods=52, freq="W-MON")
    return pd.DataFrame({
        "entity_id": ["057"] * 52,
        "liquidity_group": ["TRR"] * 52,
        "week_start": weeks,
        "amount_eur": np.random.randn(52) * 1000 + 5000,
        "lp_amount": np.random.randn(52) * 1000 + 5000
    })


def test_compute_daily_patterns(feature_engineer, daily_data):
    """Test daily pattern feature computation."""
    features = feature_engineer.compute_daily_patterns(daily_data)
    assert len(features) > 0
    assert "Pct_Txn_Friday" in features.columns
    assert "Daily_Std_Within_Week" in features.columns


def test_compute_lag_features(feature_engineer, weekly_data):
    """Test lag feature computation."""
    features = feature_engineer.compute_lag_features(weekly_data, max_lag=52)
    assert "Lag_1" in features.columns
    assert "Lag_52" in features.columns
    # Check no leakage: Lag_1 should be NaN for first row
    assert pd.isna(features.loc[0, "Lag_1"])


def test_compute_calendar_features(feature_engineer, weekly_data):
    """Test calendar feature computation."""
    features = feature_engineer.compute_calendar_features(weekly_data)
    assert "Week_of_Year" in features.columns
    assert "Month" in features.columns
    assert "Quarter" in features.columns
    assert "Is_Month_End" in features.columns


def test_build_features_count(feature_engineer, weekly_data, daily_data):
    """Test total feature count."""
    features = feature_engineer.build_features(weekly_data, daily_data)
    feature_cols = get_feature_columns(features)
    # Should have ~118 features
    assert len(feature_cols) >= 100  # Allow some flexibility
    assert len(feature_cols) <= 130


def test_no_future_data_in_lags(feature_engineer, weekly_data):
    """Test that lag features don't contain future data."""
    features = feature_engineer.compute_lag_features(weekly_data)
    # For row i, Lag_1 should equal amount_eur from row i-1
    for i in range(1, min(5, len(features))):
        expected = features.loc[i-1, "amount_eur"]
        actual = features.loc[i, "Lag_1"]
        if not pd.isna(actual):
            assert np.isclose(expected, actual, rtol=0.01)
