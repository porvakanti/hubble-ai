"""
Metrics Module

Implements performance metrics for treasury cash flow forecasting:
- WAPE (Weighted Absolute Percentage Error) - primary metric
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Directionality (% correct sign predictions)
- PICP (Prediction Interval Coverage Probability)
- Quantile Score (Pinball loss)

All metrics are computed per-horizon (W1-W8) and per entity-liquidity group.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Compute WAPE (Weighted Absolute Percentage Error).

    WAPE = sum(|actual - forecast|) / sum(|actual|)

    Args:
        actual: Actual values
        forecast: Forecasted values

    Returns:
        WAPE score (0 = perfect, higher = worse)
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    # Handle NaN
    mask = ~(np.isnan(actual) | np.isnan(forecast))
    if mask.sum() == 0:
        return np.nan

    actual_clean = actual[mask]
    forecast_clean = forecast[mask]

    numerator = np.sum(np.abs(actual_clean - forecast_clean))
    denominator = np.sum(np.abs(actual_clean))

    if denominator == 0:
        return np.nan

    return numerator / denominator


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Compute MAE (Mean Absolute Error).

    Args:
        actual: Actual values
        forecast: Forecasted values

    Returns:
        MAE score
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    mask = ~(np.isnan(actual) | np.isnan(forecast))
    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(actual[mask] - forecast[mask]))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Compute RMSE (Root Mean Squared Error).

    Args:
        actual: Actual values
        forecast: Forecasted values

    Returns:
        RMSE score
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    mask = ~(np.isnan(actual) | np.isnan(forecast))
    if mask.sum() == 0:
        return np.nan

    return np.sqrt(np.mean((actual[mask] - forecast[mask]) ** 2))


def directionality(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Compute directionality score (% correct sign predictions).

    Measures how often the forecast correctly predicts positive vs negative cash flow.

    Args:
        actual: Actual values
        forecast: Forecasted values

    Returns:
        Directionality score (0-1, 1 = perfect)
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    mask = ~(np.isnan(actual) | np.isnan(forecast))
    if mask.sum() == 0:
        return np.nan

    actual_clean = actual[mask]
    forecast_clean = forecast[mask]

    # Same sign = correct direction
    correct = np.sign(actual_clean) == np.sign(forecast_clean)

    return np.mean(correct)


def picp(
    actual: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray
) -> float:
    """
    Compute PICP (Prediction Interval Coverage Probability).

    Measures how often actual values fall within predicted intervals.

    Args:
        actual: Actual values
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval

    Returns:
        PICP score (0-1, target typically 0.80 for 80% intervals)
    """
    actual = np.asarray(actual)
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)

    mask = ~(np.isnan(actual) | np.isnan(lower_bound) | np.isnan(upper_bound))
    if mask.sum() == 0:
        return np.nan

    actual_clean = actual[mask]
    lower_clean = lower_bound[mask]
    upper_clean = upper_bound[mask]

    # Check if actual falls within bounds
    within_bounds = (actual_clean >= lower_clean) & (actual_clean <= upper_clean)

    return np.mean(within_bounds)


def quantile_score(actual: np.ndarray, forecast: np.ndarray, quantile: float) -> float:
    """
    Compute quantile score (pinball loss).

    Args:
        actual: Actual values
        forecast: Forecasted quantile values
        quantile: Target quantile (e.g., 0.85, 0.90)

    Returns:
        Quantile score (lower = better)
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    mask = ~(np.isnan(actual) | np.isnan(forecast))
    if mask.sum() == 0:
        return np.nan

    actual_clean = actual[mask]
    forecast_clean = forecast[mask]

    error = actual_clean - forecast_clean
    loss = np.where(error >= 0, quantile * error, (quantile - 1) * error)

    return np.mean(loss)


class MetricsCalculator:
    """
    Metrics calculator for forecast evaluation.

    Computes metrics per-horizon and per entity-liquidity group.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics calculator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.monitoring_config = config.get("monitoring", {})
        logger.info("Initialized MetricsCalculator")

    def compute_metrics_per_horizon(
        self,
        actuals: pd.DataFrame,
        forecasts: pd.DataFrame,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Compute metrics for a single forecast horizon.

        Args:
            actuals: DataFrame with actual values (entity_id, liquidity_group, week_start, amount_eur)
            forecasts: DataFrame with forecasts (entity_id, liquidity_group, week_start, p50, p85, p90, p95, p99)
            horizon: Forecast horizon (1-8)

        Returns:
            DataFrame with metrics per entity-liquidity group
        """
        logger.info(f"Computing metrics for horizon {horizon}")

        # Merge actuals with forecasts
        merged = actuals.merge(
            forecasts,
            on=["entity_id", "liquidity_group", "week_start"],
            how="inner",
            suffixes=("_actual", "_forecast"),
        )

        if merged.empty:
            logger.warning(f"No matching data for horizon {horizon}")
            return pd.DataFrame()

        # Compute metrics per entity-liquidity group
        results = []

        for (entity, liq_group), group in merged.groupby(["entity_id", "liquidity_group"]):
            actual_vals = group["amount_eur"].values
            forecast_p50 = group.get("p50", group.get("forecast", np.nan)).values

            metrics_dict = {
                "entity_id": entity,
                "liquidity_group": liq_group,
                "horizon": horizon,
                "n_observations": len(group),
                "wape": wape(actual_vals, forecast_p50),
                "mae": mae(actual_vals, forecast_p50),
                "rmse": rmse(actual_vals, forecast_p50),
                "directionality": directionality(actual_vals, forecast_p50),
            }

            # PICP (if quantiles available)
            if "p85" in group.columns and "p99" in group.columns:
                metrics_dict["picp_85_99"] = picp(actual_vals, group["p85"].values, group["p99"].values)

            if "p90" in group.columns and "p95" in group.columns:
                metrics_dict["picp_90_95"] = picp(actual_vals, group["p90"].values, group["p95"].values)

            # Quantile scores
            for q in [0.85, 0.90, 0.95, 0.99]:
                q_col = f"p{int(q*100)}"
                if q_col in group.columns:
                    metrics_dict[f"quantile_score_{q_col}"] = quantile_score(
                        actual_vals, group[q_col].values, q
                    )

            results.append(metrics_dict)

        metrics_df = pd.DataFrame(results)

        logger.info(
            f"Metrics computed for horizon {horizon}",
            entities=metrics_df["entity_id"].nunique(),
            rows=len(metrics_df),
        )

        return metrics_df

    def compute_aggregate_metrics(
        self, metrics_per_entity: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute aggregate metrics across all entities.

        Args:
            metrics_per_entity: DataFrame with per-entity metrics

        Returns:
            Dictionary of aggregate metrics
        """
        agg_metrics = {
            "wape_mean": metrics_per_entity["wape"].mean(),
            "wape_median": metrics_per_entity["wape"].median(),
            "wape_p95": metrics_per_entity["wape"].quantile(0.95),
            "mae_mean": metrics_per_entity["mae"].mean(),
            "rmse_mean": metrics_per_entity["rmse"].mean(),
            "directionality_mean": metrics_per_entity["directionality"].mean(),
            "directionality_min": metrics_per_entity["directionality"].min(),
        }

        if "picp_85_99" in metrics_per_entity.columns:
            agg_metrics["picp_85_99_mean"] = metrics_per_entity["picp_85_99"].mean()

        return agg_metrics

    def check_against_targets(
        self, metrics: pd.DataFrame, horizon: int
    ) -> Dict[str, Any]:
        """
        Check metrics against configured targets.

        Args:
            metrics: Metrics DataFrame
            horizon: Forecast horizon

        Returns:
            Dictionary with pass/fail status and violations
        """
        wape_targets = self.monitoring_config.get("wape_targets", {})
        dir_targets = self.monitoring_config.get("directionality_min", {})
        picp_target = self.monitoring_config.get("picp_target", 0.80)

        horizon_str = str(horizon)
        wape_target = wape_targets.get(horizon_str, 0.10)
        dir_target = dir_targets.get(horizon_str, 0.85)

        # Aggregate metrics
        wape_mean = metrics["wape"].mean()
        dir_mean = metrics["directionality"].mean()

        # Violations
        wape_violations = metrics[metrics["wape"] > wape_target]
        dir_violations = metrics[metrics["directionality"] < dir_target]

        results = {
            "horizon": horizon,
            "wape_target": wape_target,
            "wape_actual": wape_mean,
            "wape_pass": wape_mean <= wape_target,
            "wape_violations_count": len(wape_violations),
            "directionality_target": dir_target,
            "directionality_actual": dir_mean,
            "directionality_pass": dir_mean >= dir_target,
            "directionality_violations_count": len(dir_violations),
        }

        if "picp_85_99" in metrics.columns:
            picp_mean = metrics["picp_85_99"].mean()
            results["picp_target"] = picp_target
            results["picp_actual"] = picp_mean
            results["picp_pass"] = picp_mean >= picp_target

        logger.info(
            f"Target check for horizon {horizon}",
            wape_pass=results["wape_pass"],
            dir_pass=results["directionality_pass"],
        )

        return results

    def compute_all_horizons(
        self,
        actuals: pd.DataFrame,
        forecasts_by_horizon: Dict[int, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Compute metrics for all forecast horizons.

        Args:
            actuals: Actuals DataFrame
            forecasts_by_horizon: Dictionary mapping horizon -> forecasts DataFrame

        Returns:
            Combined metrics DataFrame for all horizons
        """
        logger.info("Computing metrics for all horizons")

        all_metrics = []

        for horizon in sorted(forecasts_by_horizon.keys()):
            forecasts = forecasts_by_horizon[horizon]
            metrics_h = self.compute_metrics_per_horizon(actuals, forecasts, horizon)
            all_metrics.append(metrics_h)

        if not all_metrics:
            return pd.DataFrame()

        combined = pd.concat(all_metrics, ignore_index=True)

        logger.info(
            "All horizons metrics complete",
            horizons=combined["horizon"].nunique(),
            total_rows=len(combined),
        )

        return combined


def format_metrics_for_display(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Format metrics DataFrame for readable display.

    Args:
        metrics: Raw metrics DataFrame

    Returns:
        Formatted DataFrame with percentages and rounded values
    """
    display = metrics.copy()

    # Convert WAPE to percentage
    if "wape" in display.columns:
        display["wape_pct"] = (display["wape"] * 100).round(2)

    # Convert directionality to percentage
    if "directionality" in display.columns:
        display["directionality_pct"] = (display["directionality"] * 100).round(2)

    # Round numeric columns
    numeric_cols = display.select_dtypes(include=[np.number]).columns
    display[numeric_cols] = display[numeric_cols].round(2)

    return display
