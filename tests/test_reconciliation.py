"""Tests for hierarchical reconciliation"""
import pytest
import numpy as np
import pandas as pd
from cf_forecast.reconciliation import HierarchicalReconciler


@pytest.fixture
def config():
    return {"reconciliation": "mint_ols"}


@pytest.fixture
def reconciler(config):
    return HierarchicalReconciler(config)


def test_summing_matrix_shape(reconciler):
    """Test summing matrix has correct shape."""
    entities = ["057", "11G5", "17C7"]
    S = reconciler.build_summing_matrix(entities)

    n = len(entities)
    n_bottom = 2 * n  # TRR + TRP per entity
    m = 3 * n + 3  # Total rows

    assert S.shape == (m, n_bottom)


def test_summing_matrix_bottom_identity(reconciler):
    """Test bottom level is identity."""
    entities = ["057", "11G5"]
    S = reconciler.build_summing_matrix(entities)

    n_bottom = 4  # 2 entities × 2 groups
    # Bottom rows should be identity
    assert np.allclose(S[:n_bottom, :n_bottom], np.eye(n_bottom))


def test_entity_net_aggregation(reconciler):
    """Test entity net aggregates TRR + TRP."""
    entities = ["057"]
    S = reconciler.build_summing_matrix(entities)

    # Bottom: [TRR_057, TRP_057]
    # Entity Net row should sum both
    entity_net_row = S[2, :]  # Row 2 is Entity Net for 057
    expected = np.array([1, 1])  # Sums both TRR and TRP

    assert np.allclose(entity_net_row, expected)


def test_reconciliation_preserves_total(reconciler):
    """Test reconciliation preserves grand total."""
    entities = ["057", "11G5"]
    S = reconciler.build_summing_matrix(entities)

    # Base forecasts (random)
    np.random.seed(42)
    n_bottom = 4
    bottom_forecasts = np.random.randn(n_bottom) * 1000

    # Build full hierarchy
    base_forecasts = S @ bottom_forecasts

    # Reconcile
    reconciled = reconciler.reconcile(base_forecasts)

    # Grand total should be preserved (last row)
    assert np.isclose(reconciled[-1], base_forecasts[-1])


def test_quantile_monotonicity(reconciler):
    """Test quantile monotonicity enforcement."""
    df = pd.DataFrame({
        "entity_id": ["057", "057"],
        "liquidity_group": ["TRR", "TRP"],
        "level": ["bottom", "bottom"],
        "p85": [100, 200],
        "p90": [95, 210],  # Violation: p90 < p85 for first row
        "p95": [105, 220],
    })

    result = reconciler.enforce_monotonicity(df, ["p85", "p90", "p95"])

    # p90 should be >= p85
    assert result.loc[0, "p90"] >= result.loc[0, "p85"]
    assert result.loc[1, "p90"] >= result.loc[1, "p85"]
