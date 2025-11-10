"""
Pipeline Orchestration

End-to-end pipeline: ingest → validate → preprocess → features → train → forecast → reconcile → monitor → publish
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import structlog

from cf_forecast.io import get_backend
from cf_forecast.validation import DataValidator
from cf_forecast.preprocessing import DataPreprocessor
from cf_forecast.features import FeatureEngineer, get_feature_columns
from cf_forecast.modeling.learners import LightGBMForecaster
from cf_forecast.reconciliation import HierarchicalReconciler
from cf_forecast.monitor import PerformanceMonitor
from cf_forecast.metrics import MetricsCalculator

logger = structlog.get_logger(__name__)


class ForecastPipeline:
    """Main forecast pipeline orchestrator."""

    def __init__(self, config: Dict[str, Any], credentials: Optional[Dict[str, Any]] = None):
        self.config = config
        self.credentials = credentials or {}

        # Initialize components
        self.backend = get_backend(
            config["data_backend"]["profile"],
            config["data_backend"]["options"],
            credentials
        )
        self.validator = DataValidator(config)
        self.preprocessor = DataPreprocessor(config)
        self.feature_engineer = FeatureEngineer(config)
        self.reconciler = HierarchicalReconciler(config)
        self.monitor = PerformanceMonitor(config)
        self.metrics_calc = MetricsCalculator(config)

        logger.info("Initialized ForecastPipeline")

    def run_full_pipeline(self, asof_date: datetime) -> Dict[str, Any]:
        """Run complete forecast pipeline."""
        logger.info(f"Starting full pipeline for {asof_date}")

        results = {}

        # Step 1: Ingest
        logger.info("Step 1: Ingesting data")
        data = self.ingest_data()
        results["data_loaded"] = True

        # Step 2: Validate
        logger.info("Step 2: Validating data")
        validation_results = self.validate_data(data)
        results["validation"] = validation_results

        # Step 3: Preprocess
        logger.info("Step 3: Preprocessing data")

        # Preprocess daily actuals (standardize column names but don't aggregate yet)
        daily_actuals_processed = self.preprocessor.standardize_actuals_columns(data["actuals"])
        # Filter to asof_date for daily patterns
        daily_actuals_processed = daily_actuals_processed[
            daily_actuals_processed["posting_date"] <= asof_date
        ].copy()

        # Prepare weekly modeling data
        modeling_df = self.preprocessor.prepare_modeling_data(
            data["actuals"],
            data["lp"],
            data["entity_mapping"],
            asof_date
        )
        results["preprocessing_complete"] = True

        # Step 4: Feature engineering
        logger.info("Step 4: Engineering features")
        features_df = self.feature_engineer.build_features(modeling_df, daily_actuals_processed)
        results["features_engineered"] = len(get_feature_columns(features_df))

        # Step 5: Train and forecast
        logger.info("Step 5: Training and forecasting")
        forecasts = self.train_and_forecast(features_df, asof_date)
        results["forecasts_generated"] = True

        # Step 6: Reconcile
        logger.info("Step 6: Reconciling forecasts")
        entities = features_df["entity_id"].unique().tolist()
        reconciled = self.reconciler.reconcile_quantiles(
            forecasts,
            entities,
            ["p50", "p85", "p90", "p95", "p99"]
        )
        results["reconciliation_complete"] = True

        # Step 7: Save outputs
        logger.info("Step 7: Saving outputs")
        self.save_outputs(reconciled, asof_date)
        results["outputs_saved"] = True

        logger.info("Full pipeline complete")
        return results

    def ingest_data(self) -> Dict[str, pd.DataFrame]:
        """Ingest all required data."""
        paths = self.config["paths"]

        actuals = self.backend.read_csv(f"{paths['raw']}actuals_curated.csv")
        lp = self.backend.read_csv(f"{paths['raw']}LP_17C7.csv")
        entity_mapping = self.backend.read_csv(f"{paths['reference']}Entity-Liquidity_Map.csv")

        logger.info("Data ingested", actuals=len(actuals), lp=len(lp), entities=len(entity_mapping))

        return {"actuals": actuals, "lp": lp, "entity_mapping": entity_mapping}

    def validate_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate all data sources."""
        results = {}

        results["actuals"] = self.validator.validate_actuals(data["actuals"])
        results["lp"] = self.validator.validate_liquidity_plan(data["lp"])
        results["entity_mapping"] = self.validator.validate_entity_mapping(data["entity_mapping"])

        all_passed = all(r.passed for r in results.values())
        logger.info("Validation complete", all_passed=all_passed)

        return results

    def train_and_forecast(self, features_df: pd.DataFrame, asof_date: datetime) -> pd.DataFrame:
        """Train model and generate forecasts."""
        # Simple implementation: use LightGBM for W1 forecast
        feature_cols = get_feature_columns(features_df)

        # Filter to training data (before asof_date)
        train_df = features_df[features_df["week_start"] < asof_date].copy()

        # Prepare training data
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["amount_eur"]

        # Train model
        model = LightGBMForecaster(self.config)
        model.fit(X_train, y_train)

        # Prepare forecast data (week after asof_date)
        forecast_df = features_df[features_df["week_start"] == asof_date].copy()
        X_forecast = forecast_df[feature_cols].fillna(0)

        # Generate predictions
        predictions = model.predict(X_forecast)

        # Build results DataFrame
        results = forecast_df[["entity_id", "liquidity_group", "week_start"]].copy()
        for q, vals in predictions.items():
            results[q] = vals

        logger.info("Forecasts generated", rows=len(results))

        return results

    def save_outputs(self, forecasts: pd.DataFrame, asof_date: datetime):
        """Save forecast outputs."""
        output_dir = Path(self.config["paths"]["forecasts"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parquet
        parquet_path = output_dir / f"forecasts_{asof_date.strftime('%Y%m%d')}.parquet"
        forecasts.to_parquet(parquet_path, index=False)

        # Excel
        if self.config.get("publishing", {}).get("write_excel", True):
            excel_path = output_dir / f"forecasts_{asof_date.strftime('%Y%m%d')}.xlsx"
            with pd.ExcelWriter(excel_path) as writer:
                forecasts[forecasts["level"] == "bottom"].to_excel(writer, sheet_name="Bottom", index=False)
                forecasts[forecasts["level"] == "entity"].to_excel(writer, sheet_name="Entity_Net", index=False)
                forecasts[forecasts["level"] == "group"].to_excel(writer, sheet_name="Group", index=False)

        logger.info(f"Outputs saved to {output_dir}")
