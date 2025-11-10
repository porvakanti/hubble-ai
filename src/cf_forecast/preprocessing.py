"""
Preprocessing Module - Daily→Weekly Aggregation

Transforms daily transaction-level actuals to weekly ISO format,
joins with liquidity plans and FX rates, and prepares data for feature engineering.

Key operations:
1. Daily actuals → Weekly aggregation (ISO weeks, Monday start)
2. Join with Liquidity Plan forecasts
3. Currency conversion using FX rates
4. Entity-Liquidity group filtering based on mapping
5. Data quality checks and missing value handling
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class DataPreprocessor:
    """
    Preprocessor for treasury cash flow data.

    Handles transformation from daily transactions to weekly aggregates
    with LP integration and FX conversion.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preproc_config = config.get("preprocessing", {})
        self.currency = config.get("project", {}).get("currency", "EUR")
        logger.info("Initialized DataPreprocessor", currency=self.currency)

    def process_actuals_daily_to_weekly(
        self, df: pd.DataFrame, asof_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Aggregate daily actuals to weekly level (ISO weeks).

        Args:
            df: Daily actuals with columns: Entity, Value Date, Amount Functional Currency, Liquidity Group
            asof_date: Cut-off date for filtering (only use data up to this date)

        Returns:
            Weekly aggregated DataFrame with ISO week_start
        """
        logger.info("Aggregating daily actuals to weekly", rows=len(df))

        # Standardize column names
        df_proc = df.copy()
        df_proc = df_proc.rename(
            columns={
                "Entity": "entity_id",
                "Value Date": "posting_date",
                "Amount Functional Currency": "amount_eur",
                "Liquidity Group": "liquidity_group",
            }
        )

        # Ensure proper types
        df_proc["posting_date"] = pd.to_datetime(df_proc["posting_date"])
        df_proc["amount_eur"] = pd.to_numeric(df_proc["amount_eur"])

        # Filter by asof_date if provided (only use historical data)
        if asof_date is not None:
            df_proc = df_proc[df_proc["posting_date"] <= asof_date].copy()
            logger.info(f"Filtered actuals up to {asof_date}", rows=len(df_proc))

        # Add ISO week information
        df_proc["week_year"] = df_proc["posting_date"].dt.isocalendar().year
        df_proc["week_number"] = df_proc["posting_date"].dt.isocalendar().week

        # Calculate week_start (Monday of ISO week)
        df_proc["week_start"] = df_proc["posting_date"] - pd.to_timedelta(
            df_proc["posting_date"].dt.weekday, unit="D"
        )

        # Aggregate to weekly level
        agg_dict = {"amount_eur": "sum"}

        weekly = (
            df_proc.groupby(["entity_id", "liquidity_group", "week_start", "week_year", "week_number"])
            .agg(agg_dict)
            .reset_index()
        )

        # Sort by entity, liquidity group, and week
        weekly = weekly.sort_values(["entity_id", "liquidity_group", "week_start"])

        logger.info(
            "Weekly aggregation complete",
            weekly_rows=len(weekly),
            date_range=(weekly["week_start"].min(), weekly["week_start"].max()),
        )

        return weekly

    def process_liquidity_plan(
        self, df: pd.DataFrame, asof_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Process and standardize liquidity plan data.

        Args:
            df: LP data with flexible schema
            asof_date: Reference date for filtering relevant LP forecasts

        Returns:
            Standardized LP DataFrame with weekly forecasts
        """
        logger.info("Processing liquidity plan data", rows=len(df))

        # Identify columns (flexible naming)
        entity_col = self._find_column(df, ["Entity", "Entity ID", "entity_id", "Entity Short Name"])
        date_col = self._find_column(df, ["Item's Date", "Date", "week_start", "Item Date"])
        amount_col = self._find_column(
            df, ["Amount", "Amount in plan currency", "lp_amount", "Amount Functional Currency"]
        )
        liq_group_col = self._find_column(
            df,
            [
                "Liquidity Group",
                "Liquidity Group/Super Liquidity Group",
                "liquidity_group",
            ],
        )

        # Create standardized view
        df_proc = df[[entity_col, date_col, amount_col, liq_group_col]].copy()
        df_proc.columns = ["entity_id", "plan_date", "lp_amount", "liquidity_group"]

        # Parse dates
        df_proc["plan_date"] = pd.to_datetime(df_proc["plan_date"])
        df_proc["lp_amount"] = pd.to_numeric(df_proc["lp_amount"], errors="coerce")

        # Clean entity IDs (remove ITS prefix if present, keep short form)
        df_proc["entity_id"] = df_proc["entity_id"].astype(str).str.strip()

        # Filter valid liquidity groups
        df_proc = df_proc[df_proc["liquidity_group"].isin(["TRR", "TRP"])].copy()

        # Calculate ISO week_start
        df_proc["week_start"] = df_proc["plan_date"] - pd.to_timedelta(
            df_proc["plan_date"].dt.weekday, unit="D"
        )

        # Aggregate by entity, liquidity group, and week
        lp_weekly = (
            df_proc.groupby(["entity_id", "liquidity_group", "week_start"])
            .agg({"lp_amount": "sum"})
            .reset_index()
        )

        # Filter by asof_date if provided (only future forecasts matter)
        if asof_date is not None:
            # Keep LP forecasts that are in the future relative to asof_date
            lp_weekly = lp_weekly[lp_weekly["week_start"] > asof_date].copy()
            logger.info(f"Filtered LP forecasts after {asof_date}", rows=len(lp_weekly))

        logger.info(
            "LP processing complete",
            rows=len(lp_weekly),
            entities=lp_weekly["entity_id"].nunique(),
        )

        return lp_weekly

    def process_fx_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process FX rates data (ECB format).

        Args:
            df: FX rates with Date column and currency columns

        Returns:
            Processed FX rates DataFrame
        """
        logger.info("Processing FX rates", rows=len(df))

        df_proc = df.copy()
        df_proc["Date"] = pd.to_datetime(df_proc["Date"])
        df_proc = df_proc.sort_values("Date")

        # Forward fill missing rates (weekends/holidays)
        currency_cols = [col for col in df_proc.columns if col != "Date"]
        df_proc[currency_cols] = df_proc[currency_cols].fillna(method="ffill")

        logger.info("FX rates processing complete", rows=len(df_proc))

        return df_proc

    def load_entity_liquidity_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load and process entity-liquidity group mapping.

        Args:
            df: Entity mapping with Entity, TRR_active, TRP_active

        Returns:
            Processed mapping DataFrame
        """
        logger.info("Loading entity-liquidity mapping", rows=len(df))

        df_proc = df.copy()

        # Standardize column names
        if "Entity" in df_proc.columns:
            df_proc = df_proc.rename(columns={"Entity": "entity_id"})

        # Convert boolean flags
        df_proc["TRR_active"] = df_proc["TRR_active"].astype(str).str.upper() == "TRUE"
        df_proc["TRP_active"] = df_proc["TRP_active"].astype(str).str.upper() == "TRUE"

        # Explode to (entity, liquidity_group) pairs
        rows = []
        for _, row in df_proc.iterrows():
            entity = row["entity_id"]
            if row["TRR_active"]:
                rows.append({"entity_id": entity, "liquidity_group": "TRR"})
            if row["TRP_active"]:
                rows.append({"entity_id": entity, "liquidity_group": "TRP"})

        mapping_exploded = pd.DataFrame(rows)

        logger.info(
            "Entity-liquidity mapping loaded",
            unique_pairs=len(mapping_exploded),
            entities=mapping_exploded["entity_id"].nunique(),
        )

        return mapping_exploded

    def join_actuals_with_lp(
        self,
        actuals_weekly: pd.DataFrame,
        lp_weekly: pd.DataFrame,
        entity_mapping: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join weekly actuals with LP forecasts and filter by entity mapping.

        Args:
            actuals_weekly: Weekly aggregated actuals
            lp_weekly: Weekly LP forecasts
            entity_mapping: Valid (entity, liquidity_group) pairs

        Returns:
            Combined DataFrame ready for feature engineering
        """
        logger.info("Joining actuals with LP and entity mapping")

        # Filter actuals to only valid entity-liquidity pairs
        df = actuals_weekly.merge(
            entity_mapping, on=["entity_id", "liquidity_group"], how="inner"
        )

        logger.info(
            "After entity mapping filter",
            rows=len(df),
            entities=df["entity_id"].nunique(),
        )

        # Create full time series skeleton (all weeks for all entity-liquidity pairs)
        # This ensures we have rows even for weeks with no transactions
        min_week = df["week_start"].min()
        max_week = df["week_start"].max()

        # Generate all weeks
        all_weeks = pd.date_range(start=min_week, end=max_week, freq="W-MON")

        # Create cartesian product of entity-liquidity pairs × weeks
        entity_liq_pairs = df[["entity_id", "liquidity_group"]].drop_duplicates()
        skeleton = entity_liq_pairs.merge(
            pd.DataFrame({"week_start": all_weeks}), how="cross"
        )

        # Merge actuals onto skeleton
        df_full = skeleton.merge(
            df, on=["entity_id", "liquidity_group", "week_start"], how="left"
        )

        # Fill missing amounts with 0 (no transactions that week)
        df_full["amount_eur"] = df_full["amount_eur"].fillna(0)

        # Add LP forecasts (left join - LP may not exist for all weeks)
        df_full = df_full.merge(
            lp_weekly, on=["entity_id", "liquidity_group", "week_start"], how="left"
        )

        # LP amount remains NaN if not available (will be handled in features)

        # Sort
        df_full = df_full.sort_values(["entity_id", "liquidity_group", "week_start"])

        logger.info(
            "Join complete",
            total_rows=len(df_full),
            entities=df_full["entity_id"].nunique(),
            weeks=df_full["week_start"].nunique(),
        )

        return df_full

    def prepare_modeling_data(
        self,
        actuals_df: pd.DataFrame,
        lp_df: pd.DataFrame,
        entity_mapping_df: pd.DataFrame,
        asof_date: datetime,
    ) -> pd.DataFrame:
        """
        End-to-end preprocessing: actuals + LP → modeling-ready weekly data.

        Args:
            actuals_df: Raw daily actuals
            lp_df: Raw LP data
            entity_mapping_df: Entity-liquidity mapping
            asof_date: As-of date for forecast (cut-off for historical data)

        Returns:
            Modeling-ready DataFrame with weekly actuals + LP
        """
        logger.info("Starting end-to-end preprocessing", asof_date=asof_date)

        # Step 1: Aggregate actuals to weekly (up to asof_date)
        actuals_weekly = self.process_actuals_daily_to_weekly(actuals_df, asof_date=asof_date)

        # Step 2: Process LP (future forecasts only)
        lp_weekly = self.process_liquidity_plan(lp_df, asof_date=asof_date)

        # Step 3: Load entity mapping
        entity_mapping = self.load_entity_liquidity_mapping(entity_mapping_df)

        # Step 4: Join everything
        modeling_df = self.join_actuals_with_lp(actuals_weekly, lp_weekly, entity_mapping)

        # Step 5: Add metadata
        modeling_df["asof_date"] = asof_date

        logger.info(
            "Preprocessing complete",
            final_rows=len(modeling_df),
            entities=modeling_df["entity_id"].nunique(),
            date_range=(modeling_df["week_start"].min(), modeling_df["week_start"].max()),
        )

        return modeling_df

    def _find_column(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        """Find first matching column name from candidates."""
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None


def create_weekly_skeleton(
    entities: list[str],
    liquidity_groups: list[str],
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """
    Create a complete weekly time series skeleton.

    Args:
        entities: List of entity IDs
        liquidity_groups: List of liquidity groups (TRR, TRP)
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with all combinations of (entity, liquidity_group, week_start)
    """
    # Generate all Monday dates (ISO week starts)
    all_weeks = pd.date_range(start=start_date, end=end_date, freq="W-MON")

    # Create cartesian product
    from itertools import product

    rows = []
    for entity, liq_group, week_start in product(entities, liquidity_groups, all_weeks):
        rows.append(
            {"entity_id": entity, "liquidity_group": liq_group, "week_start": week_start}
        )

    return pd.DataFrame(rows)
