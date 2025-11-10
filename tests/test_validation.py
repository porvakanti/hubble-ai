"""Tests for data validation module"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from cf_forecast.validation import DataValidator, ValidationResult


@pytest.fixture
def config():
    return {
        "validation": {
            "sign_enforcement": True,
            "quality": {"max_date_gaps_pct": 0.01}
        }
    }


@pytest.fixture
def validator(config):
    return DataValidator(config)


@pytest.fixture
def valid_actuals():
    return pd.DataFrame({
        "Entity": ["057", "057", "057"],
        "Value Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Amount Functional Currency": [1000.0, 2000.0, 1500.0],
        "Liquidity Group": ["TRR", "TRR", "TRP"]
    })


def test_validate_actuals_valid(validator, valid_actuals):
    """Test validation of valid actuals data."""
    result = validator.validate_actuals(valid_actuals)
    assert result.passed
    assert len(result.errors) == 0


def test_validate_actuals_missing_columns(validator):
    """Test validation fails with missing columns."""
    df = pd.DataFrame({"Entity": ["057"], "Value Date": ["2024-01-01"]})
    result = validator.validate_actuals(df)
    assert not result.passed
    assert len(result.errors) > 0


def test_sign_policy_trr_negative(validator):
    """Test sign policy: TRR should be >= 0."""
    df = pd.DataFrame({
        "Entity": ["057"],
        "Value Date": ["2024-01-01"],
        "Amount Functional Currency": [-1000.0],
        "Liquidity Group": ["TRR"]
    })
    result = validator.validate_actuals(df)
    assert len(result.warnings) > 0


def test_validate_entity_mapping(validator):
    """Test entity mapping validation."""
    df = pd.DataFrame({
        "Entity": ["057", "11G5"],
        "TRR_active": [True, False],
        "TRP_active": [True, True]
    })
    result = validator.validate_entity_mapping(df)
    assert result.passed
    assert result.metadata["total_entities"] == 2
