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
            "As-of Date (Tuesday)",
            value=default_date,
            help="Select the Tuesday as-of date for the forecast"
        )

        # Backend selection
        backend = st.selectbox(
            "Data Backend",
            options=["local", "s3", "azure_blob", "databricks_delta", "denodo"],
            index=0
        )

    with col2:
        # Learners
        learners = st.multiselect(
            "Select Learners",
            options=["lightgbm", "xgboost", "lstm", "sarimax"],
            default=["lightgbm"]
        )

        # Quantiles
        quantiles = st.multiselect(
            "Quantiles",
            options=["p75", "p85", "p90", "p95", "p99"],
            default=["p85", "p90", "p95", "p99"]
        )

    # Run button
    if st.button("🚀 Run Full Pipeline", type="primary"):
        with st.spinner("Running forecast pipeline..."):
            try:
                # Update config
                config["data_backend"]["profile"] = backend
                config["modeling"]["learners"] = learners

                # Initialize pipeline
                pipeline = ForecastPipeline(config)

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Ingest
                status_text.text("Step 1/7: Ingesting data...")
                progress_bar.progress(1/7)
                data = pipeline.ingest_data()

                # Step 2: Validate
                status_text.text("Step 2/7: Validating data...")
                progress_bar.progress(2/7)
                validation = pipeline.validate_data(data)

                if not all(r.passed for r in validation.values()):
                    st.error("❌ Data validation failed")
                    for name, result in validation.items():
                        if not result.passed:
                            st.error(f"{name}: {result.errors}")
                    st.stop()

                # Step 3: Preprocess
                status_text.text("Step 3/7: Preprocessing...")
                progress_bar.progress(3/7)
                modeling_df = pipeline.preprocessor.prepare_modeling_data(
                    data["actuals"],
                    data["lp"],
                    data["entity_mapping"],
                    datetime.combine(asof_date, datetime.min.time())
                )

                # Step 4: Features
                status_text.text("Step 4/7: Engineering features...")
                progress_bar.progress(4/7)
                features_df = pipeline.feature_engineer.build_features(modeling_df, data["actuals"])

                # Step 5: Train & Forecast
                status_text.text("Step 5/7: Training and forecasting...")
                progress_bar.progress(5/7)
                forecasts = pipeline.train_and_forecast(
                    features_df,
                    datetime.combine(asof_date, datetime.min.time())
                )

                # Step 6: Reconcile
                status_text.text("Step 6/7: Reconciling forecasts...")
                progress_bar.progress(6/7)
                entities = features_df["entity_id"].unique().tolist()
                reconciled = pipeline.reconciler.reconcile_quantiles(
                    forecasts,
                    entities,
                    [q for q in ["p50", "p75", "p85", "p90", "p95", "p99"] if q in quantiles or q == "p50"]
                )

                # Step 7: Save
                status_text.text("Step 7/7: Saving outputs...")
                progress_bar.progress(7/7)
                pipeline.save_outputs(reconciled, datetime.combine(asof_date, datetime.min.time()))

                # Complete
                progress_bar.progress(1.0)
                status_text.text("✅ Pipeline complete!")

                st.success(f"✅ Forecast completed successfully for {asof_date}")

                # Show summary
                st.subheader("Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Entities", len(entities))
                col2.metric("Forecasts", len(forecasts))
                col3.metric("Quantiles", len(quantiles))

                # Store in session state for other tabs
                st.session_state["forecasts"] = reconciled
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
        st.dataframe(forecasts_df)

        # Visualizations
        st.subheader("Forecast Quantiles")

        # Filter to bottom level
        bottom = forecasts_df[forecasts_df["level"] == "bottom"]

        if not bottom.empty:
            # Select entity
            entity = st.selectbox("Select Entity", bottom["entity_id"].unique())

            entity_data = bottom[bottom["entity_id"] == entity]

            # Plot quantiles
            import plotly.graph_objects as go

            fig = go.Figure()

            for q in ["p85", "p90", "p95", "p99"]:
                if q in entity_data.columns:
                    fig.add_trace(go.Scatter(
                        x=entity_data["liquidity_group"],
                        y=entity_data[q],
                        mode="lines+markers",
                        name=q
                    ))

            fig.update_layout(
                title=f"Forecast Quantiles - {entity}",
                xaxis_title="Liquidity Group",
                yaxis_title="Amount (EUR)"
            )

            st.plotly_chart(fig, use_container_width=True)

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
