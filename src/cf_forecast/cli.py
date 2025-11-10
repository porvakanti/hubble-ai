"""
Command Line Interface

Provides CLI commands for running the forecast pipeline:
- run: Full pipeline
- ingest: Data ingestion only
- features: Feature engineering only
- train: Model training only
- forecast: Generate forecasts
- reconcile: Reconcile forecasts
- monitor: Performance monitoring
- publish: Publish outputs
"""
import sys
from datetime import datetime
from pathlib import Path
import click
import yaml
import structlog

from cf_forecast import __version__
from cf_forecast.pipeline import ForecastPipeline

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger(__name__)


@click.group()
@click.version_option(version=__version__)
def main():
    """Hubble.AI - Treasury Cash-Flow Forecasting Platform"""
    pass


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to config.yml")
@click.option("--asof", required=True, type=str, help="As-of date (YYYY-MM-DD)")
@click.option("--credentials", type=click.Path(exists=True), help="Path to credentials.yml")
def run(config: str, asof: str, credentials: str = None):
    """Run full forecast pipeline."""
    logger.info("Starting full forecast pipeline", asof=asof)

    # Load config
    with open(config) as f:
        config_dict = yaml.safe_load(f)

    # Load credentials if provided
    creds = {}
    if credentials:
        with open(credentials) as f:
            creds = yaml.safe_load(f)

    # Parse asof date
    asof_date = datetime.strptime(asof, "%Y-%m-%d")

    # Run pipeline
    pipeline = ForecastPipeline(config_dict, creds)

    try:
        results = pipeline.run_full_pipeline(asof_date)
        logger.info("Pipeline completed successfully", results=results)
        click.echo("✓ Pipeline completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        click.echo(f"✗ Pipeline failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to config.yml")
@click.option("--asof", required=True, type=str, help="As-of date (YYYY-MM-DD)")
def ingest(config: str, asof: str):
    """Ingest and validate data."""
    logger.info("Ingesting data", asof=asof)

    with open(config) as f:
        config_dict = yaml.safe_load(f)

    asof_date = datetime.strptime(asof, "%Y-%m-%d")

    pipeline = ForecastPipeline(config_dict)
    data = pipeline.ingest_data()
    validation = pipeline.validate_data(data)

    all_passed = all(r.passed for r in validation.values())

    if all_passed:
        click.echo("✓ Data validation passed")
        sys.exit(0)
    else:
        click.echo("✗ Data validation failed", err=True)
        for name, result in validation.items():
            if not result.passed:
                click.echo(f"  {name}: {result.errors}", err=True)
        sys.exit(1)


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to config.yml")
@click.option("--asof", required=True, type=str, help="As-of date (YYYY-MM-DD)")
def features(config: str, asof: str):
    """Generate feature set."""
    logger.info("Generating features", asof=asof)

    with open(config) as f:
        config_dict = yaml.safe_load(f)

    asof_date = datetime.strptime(asof, "%Y-%m-%d")

    pipeline = ForecastPipeline(config_dict)
    data = pipeline.ingest_data()

    modeling_df = pipeline.preprocessor.prepare_modeling_data(
        data["actuals"],
        data["lp"],
        data["entity_mapping"],
        asof_date
    )

    features_df = pipeline.feature_engineer.build_features(modeling_df, data["actuals"])

    # Save features
    output_path = Path(config_dict["paths"]["features"]) / f"features_{asof_date.strftime('%Y%m%d')}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    click.echo(f"✓ Features generated: {len(features_df)} rows, saved to {output_path}")
    sys.exit(0)


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to config.yml")
@click.option("--asof", required=True, type=str, help="As-of date (YYYY-MM-DD)")
def forecast(config: str, asof: str):
    """Generate forecasts."""
    logger.info("Generating forecasts", asof=asof)

    with open(config) as f:
        config_dict = yaml.safe_load(f)

    asof_date = datetime.strptime(asof, "%Y-%m-%d")

    pipeline = ForecastPipeline(config_dict)

    try:
        results = pipeline.run_full_pipeline(asof_date)
        click.echo("✓ Forecasts generated successfully")
        sys.exit(0)
    except Exception as e:
        click.echo(f"✗ Forecast generation failed: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="Path to config.yml")
def version(config: str):
    """Show version and configuration."""
    click.echo(f"Hubble.AI v{__version__}")
    click.echo(f"Config: {config}")

    with open(config) as f:
        config_dict = yaml.safe_load(f)

    click.echo(f"Backend: {config_dict['data_backend']['profile']}")
    click.echo(f"Learners: {', '.join(config_dict['modeling']['learners'])}")
    click.echo(f"Horizons: {config_dict['modeling']['horizons']}")


if __name__ == "__main__":
    main()
