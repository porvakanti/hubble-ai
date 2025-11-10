"""Tests for metrics module"""
import pytest
import numpy as np
from cf_forecast.metrics import wape, mae, rmse, directionality, picp


def test_wape():
    """Test WAPE calculation."""
    actual = np.array([100, 200, 300])
    forecast = np.array([90, 210, 310])
    # WAPE = (10 + 10 + 10) / (100 + 200 + 300) = 30/600 = 0.05
    assert np.isclose(wape(actual, forecast), 0.05, rtol=0.01)


def test_mae():
    """Test MAE calculation."""
    actual = np.array([100, 200, 300])
    forecast = np.array([90, 210, 310])
    # MAE = (10 + 10 + 10) / 3 = 10
    assert np.isclose(mae(actual, forecast), 10.0)


def test_directionality():
    """Test directionality calculation."""
    actual = np.array([100, -200, 300, -400])
    forecast = np.array([50, -150, 250, 100])  # 3 correct signs, 1 wrong
    # Directionality = 3/4 = 0.75
    assert np.isclose(directionality(actual, forecast), 0.75)


def test_picp():
    """Test PICP calculation."""
    actual = np.array([100, 200, 300, 400])
    lower = np.array([90, 180, 290, 350])
    upper = np.array([110, 220, 310, 450])
    # All 4 within bounds = 1.0
    assert np.isclose(picp(actual, lower, upper), 1.0)


def test_wape_with_zeros():
    """Test WAPE handles zero actuals."""
    actual = np.array([0, 100, 200])
    forecast = np.array([10, 110, 190])
    result = wape(actual, forecast)
    # Should not fail, return valid number
    assert not np.isnan(result)
