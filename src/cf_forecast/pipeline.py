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
from cf_forecast.metrics import MetricsCalculator, wape, mae, directionality
from cf_forecast.orchestrator import ForecastOrchestrator

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

        # Step 5: Train and forecast (all models × strategies × 8 horizons)
        logger.info("Step 5: Training and forecasting")
        forecasts = self.train_and_forecast(features_df, asof_date)
        results["forecasts_generated"] = True

        # Step 6: Add actuals and metrics for backtesting
        logger.info("Step 6: Adding actuals and computing metrics")
        forecasts_with_actuals = self.add_actuals_and_metrics(forecasts, modeling_df, asof_date)
        results["metrics_computed"] = True

        # Step 7: Format output to required structure
        logger.info("Step 7: Formatting output")
        final_output = self.format_output(forecasts_with_actuals)
        results["output_formatted"] = True

        # Step 8: Save outputs
        logger.info("Step 8: Saving outputs")
        self.save_outputs(final_output, asof_date)
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
        """Train all models and generate forecasts for 8 horizons."""
        # Use orchestrator to run all model×strategy×horizon combinations
        orchestrator = ForecastOrchestrator(self.config)
        forecasts = orchestrator.run_all_combinations(features_df, asof_date)

        logger.info(
            "Forecasts generated",
            rows=len(forecasts),
            models=forecasts["model"].nunique(),
            strategies=forecasts["strategy"].nunique(),
            horizons=forecasts["horizon"].nunique()
        )

        return forecasts

    def add_actuals_and_metrics(
        self,
        forecasts_df: pd.DataFrame,
        modeling_df: pd.DataFrame,
        asof_date: datetime
    ) -> pd.DataFrame:
        """Add actuals and compute metrics for backtesting."""
        # Add actuals by joining on entity_id, liquidity_group, and week_date
        # modeling_df has historical actuals with week_start column
        actuals_map = modeling_df[["entity_id", "liquidity_group", "week_start", "amount_eur"]].copy()
        actuals_map = actuals_map.rename(columns={"week_start": "week_date", "amount_eur": "actual"})

        # Merge forecasts with actuals
        result = forecasts_df.merge(
            actuals_map,
            on=["entity_id", "liquidity_group", "week_date"],
            how="left"
        )

        # Compute metrics for each row where actual exists
        result["error"] = result["actual"] - result["p90"]  # Use p90 as point forecast

        # Compute WAPE, MAE, Directionality per row
        # These are more meaningful when aggregated, but we can compute per-row errors
        result["abs_error"] = result["error"].abs()
        result["abs_pct_error"] = (result["abs_error"] / result["actual"].abs()).fillna(0)

        # Directionality: 1 if signs match, 0 otherwise
        result["direction_correct"] = (
            (result["p90"] * result["actual"] > 0) |
            ((result["p90"] == 0) & (result["actual"] == 0))
        ).astype(int)

        logger.info(
            "Added actuals and metrics",
            rows_with_actuals=result["actual"].notna().sum(),
            total_rows=len(result)
        )

        return result

    def format_output(self, forecasts_df: pd.DataFrame) -> pd.DataFrame:
        """Format output to required structure."""
        # Required columns: Entity, Liquidity_Group, Model, Strategy, Week_Date,
        # Horizon, P85, P90, P95, P99, Actual, Error, WAPE, MAE, Directionality

        output = forecasts_df[[
            "entity_id", "liquidity_group", "model", "strategy",
            "week_date", "horizon", "p85", "p90", "p95", "p99"
        ]].copy()

        # Add optional columns if they exist
        if "actual" in forecasts_df.columns:
            output["actual"] = forecasts_df["actual"]
        else:
            output["actual"] = None

        if "error" in forecasts_df.columns:
            output["error"] = forecasts_df["error"]
        else:
            output["error"] = None

        if "abs_pct_error" in forecasts_df.columns:
            output["wape"] = forecasts_df["abs_pct_error"]
        else:
            output["wape"] = None

        if "abs_error" in forecasts_df.columns:
            output["mae"] = forecasts_df["abs_error"]
        else:
            output["mae"] = None

        if "direction_correct" in forecasts_df.columns:
            output["directionality"] = forecasts_df["direction_correct"]
        else:
            output["directionality"] = None

        # Rename columns to match required format
        output = output.rename(columns={
            "entity_id": "Entity",
            "liquidity_group": "Liquidity_Group",
            "model": "Model",
            "strategy": "Strategy",
            "week_date": "Week_Date",
            "horizon": "Horizon",
            "p85": "P85",
            "p90": "P90",
            "p95": "P95",
            "p99": "P99",
            "actual": "Actual",
            "error": "Error",
            "wape": "WAPE",
            "mae": "MAE",
            "directionality": "Directionality"
        })

        logger.info("Output formatted", rows=len(output), columns=len(output.columns))

        return output

    def save_outputs(self, forecasts: pd.DataFrame, asof_date: datetime):
        """Save forecast outputs."""
        output_dir = Path(self.config["paths"]["forecasts"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parquet
        parquet_path = output_dir / f"forecasts_{asof_date.strftime('%Y%m%d')}.parquet"
        forecasts.to_parquet(parquet_path, index=False)

        # Excel - organize by model
        if self.config.get("publishing", {}).get("write_excel", True):
            excel_path = output_dir / f"forecasts_{asof_date.strftime('%Y%m%d')}.xlsx"
            with pd.ExcelWriter(excel_path) as writer:
                # Summary sheet with all forecasts
                forecasts.to_excel(writer, sheet_name="All_Forecasts", index=False)

                # Separate sheets by model if Model column exists
                if "Model" in forecasts.columns:
                    for model in forecasts["Model"].unique():
                        model_data = forecasts[forecasts["Model"] == model]
                        sheet_name = model[:31]  # Excel sheet name limit
                        model_data.to_excel(writer, sheet_name=sheet_name, index=False)

        # CSV for easy viewing
        csv_path = output_dir / f"forecasts_{asof_date.strftime('%Y%m%d')}.csv"
        forecasts.to_csv(csv_path, index=False)

        logger.info(f"Outputs saved to {output_dir}")
