"""Performance Monitoring and Alerting"""
import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class PerformanceMonitor:
    """Monitor forecast performance and trigger alerts."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_config = config.get("monitoring", {})

    def check_thresholds(self, metrics: pd.DataFrame) -> Dict[str, Any]:
        """Check metrics against configured thresholds."""
        alerts = []
        wape_targets = self.monitoring_config.get("wape_targets", {})
        dir_targets = self.monitoring_config.get("directionality_min", {})

        for _, row in metrics.iterrows():
            horizon = str(row["horizon"])
            entity = row["entity_id"]
            liq_group = row["liquidity_group"]

            # WAPE check
            if horizon in wape_targets:
                target = wape_targets[horizon]
                actual = row["wape"]
                if actual > target:
                    alerts.append({
                        "type": "wape_threshold",
                        "horizon": horizon,
                        "entity": entity,
                        "liquidity_group": liq_group,
                        "target": target,
                        "actual": actual,
                        "severity": "high" if actual > target * 1.5 else "medium"
                    })

            # Directionality check
            if horizon in dir_targets and "directionality" in row:
                target = dir_targets[horizon]
                actual = row["directionality"]
                if actual < target:
                    alerts.append({
                        "type": "directionality_low",
                        "horizon": horizon,
                        "entity": entity,
                        "liquidity_group": liq_group,
                        "target": target,
                        "actual": actual,
                        "severity": "medium"
                    })

        logger.info(f"Threshold checks complete", alerts=len(alerts))
        return {"alerts": alerts, "total_alerts": len(alerts)}

    def check_drift(self, current_features: pd.DataFrame, reference_features: pd.DataFrame) -> Dict[str, Any]:
        """Check for data drift using PSI."""
        psi_threshold = self.monitoring_config.get("drift_psi_threshold", 0.20)

        drift_results = {}

        feature_cols = [col for col in current_features.columns if col not in ["entity_id", "liquidity_group", "week_start"]]

        for col in feature_cols[:10]:  # Check top 10 features
            if col in reference_features.columns:
                psi = self._calculate_psi(reference_features[col].dropna(), current_features[col].dropna())
                if psi > psi_threshold:
                    drift_results[col] = {"psi": psi, "drifted": True}

        logger.info(f"Drift check complete", drifted_features=sum(1 for v in drift_results.values() if v["drifted"]))
        return drift_results

    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        try:
            ref_hist, bin_edges = np.histogram(reference, bins=bins)
            cur_hist, _ = np.histogram(current, bins=bin_edges)

            ref_pct = (ref_hist + 1) / (len(reference) + bins)
            cur_pct = (cur_hist + 1) / (len(current) + bins)

            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            return psi
        except:
            return 0.0

    def generate_report(self, metrics: pd.DataFrame, alerts: Dict[str, Any], asof_date: datetime) -> Dict[str, Any]:
        """Generate monitoring report."""
        report = {
            "asof_date": asof_date.isoformat(),
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_entities": metrics["entity_id"].nunique(),
                "total_horizons": metrics["horizon"].nunique(),
                "total_alerts": alerts["total_alerts"],
            },
            "metrics_summary": {
                "wape_mean": metrics["wape"].mean(),
                "wape_median": metrics["wape"].median(),
                "directionality_mean": metrics.get("directionality", pd.Series([0])).mean(),
            },
            "alerts": alerts["alerts"],
        }

        logger.info("Monitoring report generated")
        return report

    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save monitoring report to JSON."""
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")
