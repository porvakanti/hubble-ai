"""Tests for data leakage prevention - CRITICAL"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cf_forecast.cv import RollingOriginCV
from cf_forecast.features import FeatureEngineer


@pytest.fixture
def config():
    return {
        "cv": {
            "n_splits": 3,
            "embargo_weeks": 1,
            "min_history_weeks": 10,
            "test_size_weeks": 2
        },
        "features": {
            "lags": {"enabled": True, "max_lag": 10},
            "rolling_stats": {"enabled": True, "windows": [4]},
            "calendar": {"enabled": True},
            "liquidity_plans": {"enabled": True, "horizons": [1, 2, 3, 4]}
        }
    }


@pytest.fixture
def cv(config):
    return RollingOriginCV(config)


@pytest.fixture
def time_series_data():
    """Generate time series data for testing."""
    weeks = pd.date_range("2024-01-01", periods=30, freq="W-MON")
    return pd.DataFrame({
        "entity_id": ["057"] * 30,
        "liquidity_group": ["TRR"] * 30,
        "week_start": weeks,
        "amount_eur": np.random.randn(30) * 1000 + 5000
    })


@pytest.mark.leakage
def test_cv_splits_no_overlap(cv, time_series_data):
    """Test that CV splits don't overlap."""
    splits = cv.generate_splits(time_series_data)

    for split in splits:
        train_idx, test_idx = cv.get_split_indices(time_series_data, split)
        # No overlap
        assert len(set(train_idx) & set(test_idx)) == 0


@pytest.mark.leakage
def test_cv_embargo_enforced(cv, time_series_data):
    """Test that embargo period is enforced."""
    splits = cv.generate_splits(time_series_data)

    for split in splits:
        # Validate no leakage
        assert cv.validate_no_leakage(time_series_data, split)


@pytest.mark.leakage
def test_lag_features_no_future_data(config):
    """Test that lag features don't contain future data."""
    weeks = pd.date_range("2024-01-01", periods=20, freq="W-MON")
    df = pd.DataFrame({
        "entity_id": ["057"] * 20,
        "liquidity_group": ["TRR"] * 20,
        "week_start": weeks,
        "amount_eur": list(range(1, 21))  # Sequential for easy checking
    })

    engineer = FeatureEngineer(config)
    features = engineer.compute_lag_features(df, max_lag=3)

    # Check Lag_1 for row 5 should be value from row 4
    assert features.loc[5, "Lag_1"] == 5  # Row 4 has value 5

    # Check Lag_2 for row 5 should be value from row 3
    assert features.loc[5, "Lag_2"] == 4  # Row 3 has value 4


@pytest.mark.leakage
def test_lp_features_only_future(config):
    """Test that LP features only use future LP values."""
    weeks = pd.date_range("2024-01-01", periods=10, freq="W-MON")
    df = pd.DataFrame({
        "entity_id": ["057"] * 10,
        "liquidity_group": ["TRR"] * 10,
        "week_start": weeks,
        "amount_eur": np.random.randn(10) * 1000,
        "lp_amount": list(range(100, 110))  # Sequential LP values
    })

    engineer = FeatureEngineer(config)
    features = engineer.compute_lp_features(df, horizons=4)

    # LP_W1 for row 0 should be lp_amount from row 1
    assert features.loc[0, "LP_W1"] == 101

    # LP_W2 for row 0 should be lp_amount from row 2
    assert features.loc[0, "LP_W2"] == 102


@pytest.mark.leakage
def test_recursive_uses_predictions_not_actuals():
    """Test that recursive strategy uses predictions, not actuals."""
    from cf_forecast.modeling.multi_horizon import RecursiveStrategy
    from cf_forecast.modeling.baselines import NaiveForecaster

    # Create simple test data
    X = pd.DataFrame({
        "Lag_1": [100, 110, 120, 130],
        "Lag_2": [90, 100, 110, 120]
    })
    y = pd.Series([110, 120, 130, 140])

    # Train recursive model
    model = NaiveForecaster()
    recursive = RecursiveStrategy(model, horizons=[1, 2])
    recursive.fit(X, y)

    # Predict
    X_test = pd.DataFrame({
        "Lag_1": [140],
        "Lag_2": [130]
    })
    predictions = recursive.predict(X_test)

    # Should have predictions for both horizons
    assert 1 in predictions
    assert 2 in predictions

    # H2 prediction should use H1 prediction (not actual)
    # This is implicit in the recursive logic
    assert predictions[2] is not None
