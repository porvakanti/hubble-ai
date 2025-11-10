"""
Hierarchical Forecast Reconciliation - MinT/OLS

Reconciles bottom-level forecasts to ensure coherence across hierarchy:
- Bottom: Entity × {TRR, TRP} (30 series)
- Entity Net: TRR + TRP per entity (15 series)
- Group: Total TRR, Total TRP, Total Net (3 series)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import structlog

logger = structlog.get_logger(__name__)


class HierarchicalReconciler:
    """MinT/OLS reconciliation for grouped time series."""

    def __init__(self, config: Dict):
        self.config = config
        self.S_matrix = None
        self.W_inv = None

    def build_summing_matrix(self, entities: List[str]) -> np.ndarray:
        """
        Build summing matrix S.

        Bottom level: n_entities × 2 (TRR, TRP per entity) = 2n rows
        Aggregates:
          - Entity Net (n rows): TRR_i + TRP_i
          - Group TRR (1 row): sum of all TRR
          - Group TRP (1 row): sum of all TRP
          - Group Net (1 row): sum of all Net

        Total: 2n bottom + n entity + 3 group = 3n + 3 rows

        Args:
            entities: List of entity IDs

        Returns:
            S matrix (m × n) where m = total series, n = bottom series
        """
        n = len(entities)
        n_bottom = 2 * n  # TRR + TRP per entity

        # Total rows = bottom (2n) + entity net (n) + group (3)
        m = 3 * n + 3

        S = np.zeros((m, n_bottom))

        # Bottom level (identity)
        S[:n_bottom, :n_bottom] = np.eye(n_bottom)

        row_idx = n_bottom

        # Entity Net: TRR_i + TRP_i
        for i in range(n):
            S[row_idx, 2*i] = 1      # TRR_i
            S[row_idx, 2*i+1] = 1    # TRP_i
            row_idx += 1

        # Group TRR: sum all TRR
        for i in range(n):
            S[row_idx, 2*i] = 1

        row_idx += 1

        # Group TRP: sum all TRP
        for i in range(n):
            S[row_idx, 2*i+1] = 1

        row_idx += 1

        # Group Net: sum all (TRR + TRP)
        S[row_idx, :] = 1

        logger.info(f"Built summing matrix S: shape {S.shape}")

        self.S_matrix = S
        return S

    def estimate_covariance(self, residuals: pd.DataFrame) -> np.ndarray:
        """
        Estimate residual covariance matrix for MinT.

        Args:
            residuals: DataFrame with forecast residuals (actual - forecast)

        Returns:
            Covariance matrix
        """
        logger.info("Estimating residual covariance")

        # Simple OLS: W = identity (all series weighted equally)
        n = residuals.shape[1]
        W = np.eye(n)

        self.W_inv = np.linalg.inv(W)

        logger.info(f"Covariance matrix estimated: shape {W.shape}")

        return W

    def reconcile(self, base_forecasts: np.ndarray) -> np.ndarray:
        """
        Reconcile forecasts using MinT/OLS.

        Formula: y_reconciled = S @ (S' @ W_inv @ S)^-1 @ S' @ W_inv @ y_base

        Args:
            base_forecasts: Base (unreconciled) forecasts (m × 1)

        Returns:
            Reconciled forecasts (m × 1)
        """
        if self.S_matrix is None:
            raise ValueError("Summing matrix not built. Call build_summing_matrix first.")

        S = self.S_matrix
        n_bottom = S.shape[1]

        # Always create fresh W_inv for current reconciliation
        # (don't reuse from previous reconciliation with different dimensions)
        W_inv = np.eye(n_bottom)

        logger.debug(
            f"Reconciliation dimensions: S={S.shape}, W_inv={W_inv.shape}, "
            f"base_forecasts={base_forecasts.shape}"
        )

        # MinT formula
        try:
            M = S.T @ W_inv @ S
            M_inv = np.linalg.inv(M)
            G = M_inv @ S.T @ W_inv

            # Extract bottom-level forecasts
            n_bottom = S.shape[1]
            bottom_forecasts = base_forecasts[:n_bottom]

            # Reconcile
            reconciled_bottom = G @ base_forecasts

            # Propagate up hierarchy
            reconciled_all = S @ reconciled_bottom

        except np.linalg.LinAlgError:
            logger.warning("Reconciliation failed (singular matrix), returning base forecasts")
            return base_forecasts

        return reconciled_all

    def reconcile_quantiles(
        self, forecasts_df: pd.DataFrame, entities: List[str], quantiles: List[str]
    ) -> pd.DataFrame:
        """
        Reconcile forecasts for all quantiles.

        Args:
            forecasts_df: DataFrame with forecasts (entity_id, liquidity_group, p50, p85, p90, p95, p99)
            entities: List of entities
            quantiles: List of quantile columns (e.g., ['p50', 'p85', 'p90', 'p95', 'p99'])

        Returns:
            Reconciled forecasts DataFrame
        """
        logger.info("Reconciling forecasts for all quantiles")

        # Use only entities that actually exist in forecasts
        entities_in_forecasts = sorted(forecasts_df['entity_id'].unique().tolist())

        if set(entities_in_forecasts) != set(entities):
            logger.warning(
                f"Entity mismatch: provided {len(entities)} entities, "
                f"but forecasts have {len(entities_in_forecasts)} entities. "
                f"Using entities from forecasts."
            )
            entities = entities_in_forecasts

        # Build summing matrix
        self.build_summing_matrix(entities)

        # Reconcile each quantile separately
        reconciled_dfs = []

        for q in quantiles:
            if q not in forecasts_df.columns:
                logger.warning(f"Quantile {q} not in forecasts, skipping")
                continue

            # Extract bottom-level forecasts (ordered by entity, then TRR/TRP)
            bottom_forecasts = []
            for entity in entities:
                for liq_group in ['TRR', 'TRP']:
                    mask = (forecasts_df['entity_id'] == entity) & (forecasts_df['liquidity_group'] == liq_group)
                    val = forecasts_df[mask][q].values
                    if len(val) > 0:
                        bottom_forecasts.append(val[0])
                    else:
                        # Missing forecast - log warning and use 0
                        logger.warning(f"Missing forecast for {entity} {liq_group} in {q}, using 0")
                        bottom_forecasts.append(0)

            bottom_forecasts = np.array(bottom_forecasts)

            # Validate shape matches summing matrix
            expected_bottom = self.S_matrix.shape[1]
            if len(bottom_forecasts) != expected_bottom:
                raise ValueError(
                    f"Bottom forecasts shape mismatch: got {len(bottom_forecasts)}, "
                    f"expected {expected_bottom} (2 × {len(entities)} entities)"
                )

            # Build full hierarchy (base forecasts)
            base_forecasts = self.S_matrix @ bottom_forecasts

            # Reconcile
            reconciled = self.reconcile(base_forecasts)

            # Store reconciled values
            reconciled_df = self._unpack_reconciled(reconciled, entities, q)
            reconciled_dfs.append(reconciled_df)

        # Combine quantiles
        result = reconciled_dfs[0]
        for df in reconciled_dfs[1:]:
            result = result.merge(df, on=['entity_id', 'liquidity_group', 'level'], how='outer')

        logger.info("Reconciliation complete for all quantiles")

        return result

    def _unpack_reconciled(self, reconciled: np.ndarray, entities: List[str], quantile: str) -> pd.DataFrame:
        """Unpack reconciled forecasts into DataFrame."""
        n = len(entities)
        rows = []

        # Bottom level
        for i, entity in enumerate(entities):
            rows.append({
                'entity_id': entity,
                'liquidity_group': 'TRR',
                'level': 'bottom',
                quantile: reconciled[2*i]
            })
            rows.append({
                'entity_id': entity,
                'liquidity_group': 'TRP',
                'level': 'bottom',
                quantile: reconciled[2*i+1]
            })

        # Entity Net
        idx = 2 * n
        for entity in entities:
            rows.append({
                'entity_id': entity,
                'liquidity_group': 'Net',
                'level': 'entity',
                quantile: reconciled[idx]
            })
            idx += 1

        # Group level
        rows.append({
            'entity_id': 'Group',
            'liquidity_group': 'TRR',
            'level': 'group',
            quantile: reconciled[idx]
        })
        idx += 1

        rows.append({
            'entity_id': 'Group',
            'liquidity_group': 'TRP',
            'level': 'group',
            quantile: reconciled[idx]
        })
        idx += 1

        rows.append({
            'entity_id': 'Group',
            'liquidity_group': 'Net',
            'level': 'group',
            quantile: reconciled[idx]
        })

        return pd.DataFrame(rows)

    def enforce_monotonicity(self, forecasts_df: pd.DataFrame, quantiles: List[str]) -> pd.DataFrame:
        """
        Enforce quantile monotonicity: p75 <= p85 <= p90 <= p95 <= p99.

        Args:
            forecasts_df: Forecasts with quantiles
            quantiles: List of quantile columns (sorted)

        Returns:
            Forecasts with enforced monotonicity
        """
        df = forecasts_df.copy()

        # Sort quantiles
        q_sorted = sorted(quantiles, key=lambda x: float(x.replace('p', '')))

        for i in range(len(q_sorted) - 1):
            q_lower = q_sorted[i]
            q_upper = q_sorted[i+1]

            if q_lower in df.columns and q_upper in df.columns:
                # Ensure lower <= upper
                df[q_upper] = np.maximum(df[q_lower], df[q_upper])

        logger.info("Quantile monotonicity enforced")

        return df
