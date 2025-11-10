"""
Streamlit UI for Hubble.AI Treasury Forecasting

Features:
1. Run Forecast Wizard
2. Data Preview
3. Feature Controls
4. Model Options
5. Metrics Dashboard
6. Downloads
"""
import streamlit as st
import pandas as pd
import yaml
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cf_forecast import __version__, __tagline__
from cf_forecast.pipeline import ForecastPipeline

# Page config
st.set_page_config(
    page_title="Hubble.AI Treasury Forecasting",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🔭 Hubble.AI - Treasury Cash-Flow Forecasting")
st.caption(__tagline__)
st.caption(f"Version {__version__}")

# Sidebar
st.sidebar.header("Configuration")

# Load config
config_path = st.sidebar.text_input("Config Path", value="configs/config.yml")

if Path(config_path).exists():
    with open(config_path) as f:
        config = yaml.safe_load(f)
    st.sidebar.success("✓ Config loaded")
else:
    st.sidebar.error("✗ Config not found")
    st.stop()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Run Forecast", "📊 Data Preview", "📈 Metrics", "📥 Downloads"])

# Tab 1: Run Forecast
with tab1:
    st.header("Run Forecast Wizard")

    col1, col2 = st.columns(2)

    with col1:
        # As-of date selection
        default_date = datetime.now() - timedelta(days=(datetime.now().weekday() + 5) % 7)
        asof_date = st.date_input(
            "Forecast As-of Date (Tuesday)",
            value=default_date,
            help="Select the Tuesday as-of date for the forecast. The system will forecast 8 weeks ahead (W1-W8) starting from this date."
        )

    with col2:
        # Backend selection
        backend = st.selectbox(
            "Data Backend",
            options=["local", "s3", "azure_blob", "databricks_delta", "denodo"],
            index=0,
            help="Select the data backend to use for loading actuals and liquidity plan data."
        )

    # Info box
    st.info(
        "🤖 **Automatic Execution:** The system will automatically run all 4 models "
        "(LightGBM, XGBoost, LSTM, SARIMAX) × 3 strategies (Recursive, Direct, DirRec) "
        "× 8 horizons (W1-W8) and generate an ensemble forecast. "
        "Quantile forecasts will be generated at p85, p90, p95, p99 confidence levels."
    )

    # Run button
    if st.button("🚀 Run Full Pipeline", type="primary"):
        with st.spinner("Running forecast pipeline..."):
            try:
                # Update config
                config["data_backend"]["profile"] = backend

                # Initialize pipeline
                pipeline = ForecastPipeline(config)

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Ingest
                status_text.text("Step 1/8: Ingesting data...")
                progress_bar.progress(1/8)
                data = pipeline.ingest_data()

                # Step 2: Validate
                status_text.text("Step 2/8: Validating data...")
                progress_bar.progress(2/8)
                validation = pipeline.validate_data(data)

                if not all(r.passed for r in validation.values()):
                    st.error("❌ Data validation failed")
                    for name, result in validation.items():
                        if not result.passed:
                            st.error(f"{name}: {result.errors}")
                    st.stop()

                # Step 3: Preprocess
                status_text.text("Step 3/8: Preprocessing...")
                progress_bar.progress(3/8)

                asof_datetime = datetime.combine(asof_date, datetime.min.time())

                # Preprocess daily actuals (standardize columns but don't aggregate)
                daily_actuals_processed = pipeline.preprocessor.standardize_actuals_columns(data["actuals"])
                daily_actuals_processed = daily_actuals_processed[
                    daily_actuals_processed["posting_date"] <= asof_datetime
                ].copy()

                # Prepare weekly modeling data
                modeling_df = pipeline.preprocessor.prepare_modeling_data(
                    data["actuals"],
                    data["lp"],
                    data["entity_mapping"],
                    asof_datetime
                )

                # Step 4: Features
                status_text.text("Step 4/8: Engineering features...")
                progress_bar.progress(4/8)
                features_df = pipeline.feature_engineer.build_features(modeling_df, daily_actuals_processed)

                # Step 5: Train & Forecast (all models × strategies × horizons)
                status_text.text("Step 5/8: Training all models and forecasting (this may take a few minutes)...")
                progress_bar.progress(5/8)
                forecasts = pipeline.train_and_forecast(
                    features_df,
                    asof_datetime
                )

                # Step 6: Add actuals and metrics
                status_text.text("Step 6/8: Computing metrics for backtesting...")
                progress_bar.progress(6/8)
                forecasts_with_metrics = pipeline.add_actuals_and_metrics(forecasts, modeling_df, asof_datetime)

                # Step 7: Format output
                status_text.text("Step 7/8: Formatting output...")
                progress_bar.progress(7/8)
                final_output = pipeline.format_output(forecasts_with_metrics)

                # Step 8: Save
                status_text.text("Step 8/8: Saving outputs...")
                progress_bar.progress(8/8)
                pipeline.save_outputs(final_output, asof_datetime)

                # Complete
                progress_bar.progress(1.0)
                status_text.text("✅ Pipeline complete!")

                st.success(f"✅ Forecast completed successfully for {asof_date}")

                # Show summary
                st.subheader("Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Entities", final_output["Entity"].nunique())
                col2.metric("Models", final_output["Model"].nunique())
                col3.metric("Total Forecasts", len(final_output))

                # Count backtesting rows
                backtest_count = final_output["Actual"].notna().sum()
                col4.metric("Backtesting Rows", backtest_count)

                # Store in session state for other tabs
                st.session_state["forecasts"] = final_output
                st.session_state["asof_date"] = asof_date

            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
                st.exception(e)

# Tab 2: Data Preview
with tab2:
    st.header("Data Preview")

    try:
        pipeline = ForecastPipeline(config)
        data = pipeline.ingest_data()

        st.subheader("Actuals")
        st.dataframe(data["actuals"].head(100))
        st.caption(f"Total rows: {len(data['actuals'])}")

        st.subheader("Liquidity Plan")
        st.dataframe(data["lp"].head(100))
        st.caption(f"Total rows: {len(data['lp'])}")

        st.subheader("Entity Mapping")
        st.dataframe(data["entity_mapping"])

    except Exception as e:
        st.error(f"Failed to load data: {e}")

# Tab 3: Metrics
with tab3:
    st.header("Metrics Dashboard")

    if "forecasts" in st.session_state:
        forecasts_df = st.session_state["forecasts"]

        # Display full table
        st.subheader("All Forecasts")
        st.dataframe(forecasts_df, height=400)

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_entity = st.selectbox("Select Entity", ["All"] + sorted(forecasts_df["Entity"].unique().tolist()))
        with col2:
            selected_model = st.selectbox("Select Model", ["All"] + sorted(forecasts_df["Model"].unique().tolist()))
        with col3:
            selected_liquidity_group = st.selectbox("Select Liquidity Group", ["All"] + sorted(forecasts_df["Liquidity_Group"].unique().tolist()))

        # Filter data
        filtered_df = forecasts_df.copy()
        if selected_entity != "All":
            filtered_df = filtered_df[filtered_df["Entity"] == selected_entity]
        if selected_model != "All":
            filtered_df = filtered_df[filtered_df["Model"] == selected_model]
        if selected_liquidity_group != "All":
            filtered_df = filtered_df[filtered_df["Liquidity_Group"] == selected_liquidity_group]

        # Visualizations
        st.subheader("Forecast Quantiles Over Horizons")

        if not filtered_df.empty:
            import plotly.graph_objects as go

            fig = go.Figure()

            # Plot quantiles
            for q in ["P85", "P90", "P95", "P99"]:
                if q in filtered_df.columns:
                    # Aggregate by horizon
                    horizon_data = filtered_df.groupby("Horizon")[q].mean().reset_index()
                    fig.add_trace(go.Scatter(
                        x=horizon_data["Horizon"],
                        y=horizon_data[q],
                        mode="lines+markers",
                        name=q
                    ))

            # Add actuals if available
            if "Actual" in filtered_df.columns and filtered_df["Actual"].notna().any():
                actual_data = filtered_df[filtered_df["Actual"].notna()].groupby("Horizon")["Actual"].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=actual_data["Horizon"],
                    y=actual_data["Actual"],
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color="black", width=3, dash="dash")
                ))

            fig.update_layout(
                title="Average Forecast Quantiles by Horizon",
                xaxis_title="Horizon (Weeks)",
                yaxis_title="Amount (EUR)",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Metrics summary if backtesting
            if "WAPE" in filtered_df.columns and filtered_df["WAPE"].notna().any():
                st.subheader("Backtesting Metrics Summary")

                metrics_df = filtered_df[filtered_df["WAPE"].notna()]

                col1, col2, col3 = st.columns(3)
                col1.metric("Average WAPE", f"{metrics_df['WAPE'].mean():.2%}")
                col2.metric("Average MAE", f"€{metrics_df['MAE'].mean():,.2f}")
                col3.metric("Directionality", f"{metrics_df['Directionality'].mean():.1%}")

                # Metrics by model
                st.subheader("Metrics by Model")
                model_metrics = metrics_df.groupby("Model").agg({
                    "WAPE": "mean",
                    "MAE": "mean",
                    "Directionality": "mean"
                }).reset_index()
                model_metrics.columns = ["Model", "Avg WAPE", "Avg MAE", "Avg Directionality"]
                st.dataframe(model_metrics, hide_index=True)

    else:
        st.info("Run a forecast first to see metrics")

# Tab 4: Downloads
with tab4:
    st.header("Download Outputs")

    if "forecasts" in st.session_state and "asof_date" in st.session_state:
        forecasts_df = st.session_state["forecasts"]
        asof = st.session_state["asof_date"]

        # Download as CSV
        csv = forecasts_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecasts (CSV)",
            data=csv,
            file_name=f"forecasts_{asof.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

        # Excel download
        st.caption("Excel file available in artifacts/forecasts/ directory")

    else:
        st.info("Run a forecast first to download outputs")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Aperam S.A. Treasury Analytics")
st.sidebar.caption("Bringing vision and intelligence to your cash horizon.")
