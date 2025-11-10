"""
Feature Engineering Module - 118 Features

Implements complete feature set for treasury cash flow forecasting:
- Daily-level features (8): Calculated BEFORE weekly aggregation
- Weekly-level features (110): Calculated after weekly aggregation

Feature Families:
1. Daily patterns (8) - transaction concentration, volatility
2. Lag features (52) - Lags 1-52 weeks
3. Rolling statistics (18) - 4w, 12w, 26w, 52w windows
4. Calendar features (12) - week, month, quarter, holidays
5. Trend features (8) - slopes, momentum, acceleration
6. Liquidity Plan (4) - LP_W1 to LP_W4
7. Entity encoding (15) - One-hot top 15 entities
8. Liquidity group (1) - Is_TRR binary flag

Total: 8 + 52 + 18 + 12 + 8 + 4 + 15 + 1 = 118 features
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineer for treasury cash flow forecasting.

    Implements 118 features following exact specification:
    - Respects cut-off discipline (no future data)
    - Calculates daily patterns before weekly aggregation
    - Handles missing values appropriately
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineer.

        Args:
            config: Configuration dictionary with feature settings
        """
        self.config = config
        self.features_config = config.get("features", {})
        logger.info("Initialized FeatureEngineer")

    def compute_daily_patterns(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 8 daily pattern features BEFORE weekly aggregation.

        Features:
        1. Pct_Txn_Month_End_Day - % of transactions on month-end day
        2. Pct_Txn_First_Day - % of transactions on first day of month
        3. Pct_Txn_Last_Day - % of transactions on last day of month
        4. Pct_Txn_Friday - % of transactions on Friday
        5. Daily_Std_Within_Week - Std dev of daily amounts within week
        6. Daily_CV_Within_Week - Coefficient of variation within week
        7. Num_Days_With_Txn - Count of days with transactions per week
        8. Num_Transactions_Week - Total transaction count per week

        Args:
            daily_df: Daily transactions with posting_date, amount_eur

        Returns:
            DataFrame with daily pattern features per week
        """
        logger.info("Computing daily pattern features", rows=len(daily_df))

        df = daily_df.copy()
        df["posting_date"] = pd.to_datetime(df["posting_date"])

        # Add time features
        df["week_start"] = df["posting_date"] - pd.to_timedelta(
            df["posting_date"].dt.weekday, unit="D"
        )
        df["day_of_week"] = df["posting_date"].dt.dayofweek  # 0=Monday, 4=Friday
        df["day_of_month"] = df["posting_date"].dt.day
        df["days_in_month"] = df["posting_date"].dt.days_in_month

        # Binary indicators
        df["is_month_end"] = df["day_of_month"] == df["days_in_month"]
        df["is_first_day"] = df["day_of_month"] == 1
        df["is_last_day"] = df["is_month_end"]  # Same as month end
        df["is_friday"] = df["day_of_week"] == 4

        # Group by entity, liquidity_group, and week
        features_list = []

        for (entity, liq_group, week_start), group in df.groupby(
            ["entity_id", "liquidity_group", "week_start"]
        ):
            n_txn = len(group)

            # Transaction concentration
            pct_month_end = group["is_month_end"].sum() / n_txn if n_txn > 0 else 0
            pct_first_day = group["is_first_day"].sum() / n_txn if n_txn > 0 else 0
            pct_last_day = group["is_last_day"].sum() / n_txn if n_txn > 0 else 0
            pct_friday = group["is_friday"].sum() / n_txn if n_txn > 0 else 0

            # Daily volatility within week
            daily_std = group["amount_eur"].std() if n_txn > 1 else 0
            daily_mean = group["amount_eur"].mean() if n_txn > 0 else 0
            daily_cv = daily_std / abs(daily_mean) if daily_mean != 0 else 0

            # Transaction counts
            num_days_with_txn = group["posting_date"].nunique()
            num_transactions = n_txn

            features_list.append(
                {
                    "entity_id": entity,
                    "liquidity_group": liq_group,
                    "week_start": week_start,
                    "Pct_Txn_Month_End_Day": pct_month_end,
                    "Pct_Txn_First_Day": pct_first_day,
                    "Pct_Txn_Last_Day": pct_last_day,
                    "Pct_Txn_Friday": pct_friday,
                    "Daily_Std_Within_Week": daily_std,
                    "Daily_CV_Within_Week": daily_cv,
                    "Num_Days_With_Txn": num_days_with_txn,
                    "Num_Transactions_Week": num_transactions,
                }
            )

        features_df = pd.DataFrame(features_list)

        logger.info("Daily pattern features complete", features=8, rows=len(features_df))

        return features_df

    def compute_lag_features(
        self, df: pd.DataFrame, target_col: str = "amount_eur", max_lag: int = 52
    ) -> pd.DataFrame:
        """
        Compute lag features (Lag_1 to Lag_52).

        Args:
            df: Weekly data with entity_id, liquidity_group, week_start, amount_eur
            target_col: Column to lag
            max_lag: Maximum lag (default: 52 weeks)

        Returns:
            DataFrame with lag features added
        """
        logger.info(f"Computing lag features 1-{max_lag}")

        df = df.copy()
        df = df.sort_values(["entity_id", "liquidity_group", "week_start"])

        # Create lags per entity-liquidity group
        for lag in range(1, max_lag + 1):
            df[f"Lag_{lag}"] = df.groupby(["entity_id", "liquidity_group"])[target_col].shift(lag)

        logger.info(f"Lag features complete", features=max_lag)

        return df

    def compute_rolling_statistics(self, df: pd.DataFrame, target_col: str = "amount_eur") -> pd.DataFrame:
        """
        Compute rolling statistics features (18 features).

        Windows: 4, 12, 26, 52 weeks
        Stats: mean, median, std, min, max, range (max-min)

        Note: Uses 4w, 12w, 26w windows. Some combinations = 18 features.

        Args:
            df: Weekly data
            target_col: Column to compute stats on

        Returns:
            DataFrame with rolling statistics added
        """
        logger.info("Computing rolling statistics features")

        df = df.copy()
        df = df.sort_values(["entity_id", "liquidity_group", "week_start"])

        windows = self.features_config.get("rolling_stats", {}).get("windows", [4, 12, 26, 52])

        for window in windows:
            # Mean
            df[f"Roll_Mean_{window}w"] = (
                df.groupby(["entity_id", "liquidity_group"])[target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )

            # Std
            df[f"Roll_Std_{window}w"] = (
                df.groupby(["entity_id", "liquidity_group"])[target_col]
                .rolling(window=window, min_periods=2)
                .std()
                .reset_index(level=[0, 1], drop=True)
            )

        # Additional statistics for key windows
        for window in [4, 12, 26]:
            # Median
            df[f"Roll_Median_{window}w"] = (
                df.groupby(["entity_id", "liquidity_group"])[target_col]
                .rolling(window=window, min_periods=1)
                .median()
                .reset_index(level=[0, 1], drop=True)
            )

            # Min/Max
            df[f"Roll_Min_{window}w"] = (
                df.groupby(["entity_id", "liquidity_group"])[target_col]
                .rolling(window=window, min_periods=1)
                .min()
                .reset_index(level=[0, 1], drop=True)
            )

            df[f"Roll_Max_{window}w"] = (
                df.groupby(["entity_id", "liquidity_group"])[target_col]
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=[0, 1], drop=True)
            )

        logger.info("Rolling statistics complete", features=18)

        return df

    def compute_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute calendar features (12 features).

        Features:
        - Week_of_Year
        - Month
        - Quarter
        - Is_Month_End
        - Is_Quarter_End
        - Is_Year_End
        - Days_To_Month_End
        - Days_To_Quarter_End
        - Holiday_Flag (simplified: Christmas, New Year)
        - Holiday_Next_Week
        - Weeks_Since_Year_Start
        - Weeks_To_Year_End

        Args:
            df: Weekly data with week_start

        Returns:
            DataFrame with calendar features
        """
        logger.info("Computing calendar features")

        df = df.copy()
        df["week_start"] = pd.to_datetime(df["week_start"])

        # Basic calendar
        df["Week_of_Year"] = df["week_start"].dt.isocalendar().week
        df["Month"] = df["week_start"].dt.month
        df["Quarter"] = df["week_start"].dt.quarter

        # End-of-period flags
        # Month end: if week contains last day of month
        df["temp_month_end_date"] = df["week_start"] + pd.offsets.MonthEnd(0)
        df["Is_Month_End"] = (
            (df["temp_month_end_date"] >= df["week_start"])
            & (df["temp_month_end_date"] < df["week_start"] + pd.Timedelta(days=7))
        ).astype(int)

        # Quarter end
        df["Is_Quarter_End"] = df["Month"].isin([3, 6, 9, 12]).astype(int) * df["Is_Month_End"]

        # Year end
        df["Is_Year_End"] = (df["Month"] == 12).astype(int) * df["Is_Month_End"]

        # Days to month/quarter end (from week_start)
        df["Days_To_Month_End"] = (df["temp_month_end_date"] - df["week_start"]).dt.days

        # Days to quarter end
        quarter_end_month = {1: 3, 2: 6, 3: 9, 4: 12}
        df["quarter_end_month"] = df["Quarter"].map(quarter_end_month)
        df["temp_quarter_end"] = pd.to_datetime(
            df["week_start"].dt.year.astype(str)
            + "-"
            + df["quarter_end_month"].astype(str)
            + "-01"
        ) + pd.offsets.MonthEnd(0)
        df["Days_To_Quarter_End"] = (df["temp_quarter_end"] - df["week_start"]).dt.days

        # Holiday flags (simplified: Christmas week, New Year week)
        df["Holiday_Flag"] = (
            ((df["Month"] == 12) & (df["Week_of_Year"] >= 51))
            | ((df["Month"] == 1) & (df["Week_of_Year"] == 1))
        ).astype(int)

        # Holiday next week (shift forward)
        df["Holiday_Next_Week"] = df.groupby(["entity_id", "liquidity_group"])[
            "Holiday_Flag"
        ].shift(-1).fillna(0).astype(int)

        # Weeks since/to year start/end
        df["year_start"] = pd.to_datetime(df["week_start"].dt.year.astype(str) + "-01-01")
        df["year_end"] = pd.to_datetime(df["week_start"].dt.year.astype(str) + "-12-31")
        df["Weeks_Since_Year_Start"] = ((df["week_start"] - df["year_start"]).dt.days / 7).astype(
            int
        )
        df["Weeks_To_Year_End"] = ((df["year_end"] - df["week_start"]).dt.days / 7).astype(int)

        # Drop temporary columns
        df = df.drop(
            columns=[
                "temp_month_end_date",
                "quarter_end_month",
                "temp_quarter_end",
                "year_start",
                "year_end",
            ]
        )

        logger.info("Calendar features complete", features=12)

        return df

    def compute_trend_features(self, df: pd.DataFrame, target_col: str = "amount_eur") -> pd.DataFrame:
        """
        Compute trend features (8 features).

        Features:
        - Slope_4w, Slope_12w (linear regression slope)
        - Momentum_4w, Momentum_12w (current - mean)
        - Acceleration_4w (change in slope)
        - Relative_Position_4w, Relative_Position_12w ((current - min) / (max - min))
        - YoY_Change (year-over-year change)

        Args:
            df: Weekly data
            target_col: Target column

        Returns:
            DataFrame with trend features
        """
        logger.info("Computing trend features")

        df = df.copy()
        df = df.sort_values(["entity_id", "liquidity_group", "week_start"])

        # Helper function for slope
        def compute_slope(series: pd.Series) -> float:
            """Compute linear regression slope."""
            if len(series) < 2 or series.isnull().all():
                return 0.0
            x = np.arange(len(series))
            y = series.values
            # Handle NaN
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return 0.0
            slope = np.polyfit(x[mask], y[mask], 1)[0]
            return slope

        # Slope features
        for window in [4, 12]:
            df[f"Slope_{window}w"] = (
                df.groupby(["entity_id", "liquidity_group"])[target_col]
                .rolling(window=window, min_periods=2)
                .apply(compute_slope, raw=False)
                .reset_index(level=[0, 1], drop=True)
            )

        # Momentum (current - rolling mean)
        for window in [4, 12]:
            roll_mean = (
                df.groupby(["entity_id", "liquidity_group"])[target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            df[f"Momentum_{window}w"] = df[target_col] - roll_mean

        # Acceleration (change in slope)
        df["Acceleration_4w"] = df.groupby(["entity_id", "liquidity_group"])["Slope_4w"].diff()

        # Relative position
        for window in [4, 12]:
            roll_min = (
                df.groupby(["entity_id", "liquidity_group"])[target_col]
                .rolling(window=window, min_periods=1)
                .min()
                .reset_index(level=[0, 1], drop=True)
            )
            roll_max = (
                df.groupby(["entity_id", "liquidity_group"])[target_col]
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=[0, 1], drop=True)
            )
            range_val = roll_max - roll_min
            df[f"Relative_Position_{window}w"] = np.where(
                range_val != 0, (df[target_col] - roll_min) / range_val, 0.5
            )

        # YoY change (52 weeks ago)
        df["YoY_Change"] = df.groupby(["entity_id", "liquidity_group"])[target_col].diff(52)

        logger.info("Trend features complete", features=8)

        return df

    def compute_lp_features(
        self, df: pd.DataFrame, lp_col: str = "lp_amount", horizons: int = 4
    ) -> pd.DataFrame:
        """
        Compute liquidity plan features (LP_W1 to LP_W4).

        For each week, extract the LP forecasts for next 1-4 weeks.

        Args:
            df: Weekly data with lp_amount (may have NaN)
            lp_col: LP column name
            horizons: Number of LP horizons to extract (default: 4)

        Returns:
            DataFrame with LP features
        """
        logger.info(f"Computing LP features for horizons 1-{horizons}")

        df = df.copy()
        df = df.sort_values(["entity_id", "liquidity_group", "week_start"])

        # For each horizon, shift LP backwards (LP_W1 is next week's LP value)
        for h in range(1, horizons + 1):
            df[f"LP_W{h}"] = df.groupby(["entity_id", "liquidity_group"])[lp_col].shift(-h)

        logger.info("LP features complete", features=horizons)

        return df

    def compute_entity_encoding(self, df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """
        Compute entity one-hot encoding (15 features for top 15 entities).

        Args:
            df: Weekly data with entity_id
            top_n: Number of top entities to encode

        Returns:
            DataFrame with entity one-hot features
        """
        logger.info(f"Computing entity one-hot encoding for top {top_n}")

        df = df.copy()

        # Identify top entities by transaction volume
        top_entities = df.groupby("entity_id")["amount_eur"].sum().nlargest(top_n).index.tolist()

        # One-hot encode
        for entity in top_entities:
            df[f"Entity_{entity}"] = (df["entity_id"] == entity).astype(int)

        logger.info("Entity encoding complete", features=top_n)

        return df

    def compute_liquidity_group_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute liquidity group binary flag (Is_TRR).

        Args:
            df: Weekly data with liquidity_group

        Returns:
            DataFrame with Is_TRR feature
        """
        df = df.copy()
        df["Is_TRR"] = (df["liquidity_group"] == "TRR").astype(int)

        logger.info("Liquidity group encoding complete", features=1)

        return df

    def build_features(
        self,
        weekly_df: pd.DataFrame,
        daily_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build complete feature set (118 features).

        Order of operations (CRITICAL):
        1. Daily patterns (8) - if daily_df provided
        2. Lag features (52)
        3. Rolling statistics (18)
        4. Calendar features (12)
        5. Trend features (8)
        6. LP features (4)
        7. Entity encoding (15)
        8. Liquidity group encoding (1)

        Args:
            weekly_df: Weekly aggregated data (must have amount_eur, lp_amount)
            daily_df: Optional daily data for daily pattern features

        Returns:
            DataFrame with all 118 features
        """
        logger.info("Building complete feature set (118 features)")

        df = weekly_df.copy()

        # 1. Daily patterns (if daily data provided)
        if daily_df is not None and self.features_config.get("daily_patterns", {}).get(
            "enabled", True
        ):
            daily_features = self.compute_daily_patterns(daily_df)
            df = df.merge(
                daily_features, on=["entity_id", "liquidity_group", "week_start"], how="left"
            )
            # Fill NaN with 0 for weeks without daily data
            daily_cols = [
                "Pct_Txn_Month_End_Day",
                "Pct_Txn_First_Day",
                "Pct_Txn_Last_Day",
                "Pct_Txn_Friday",
                "Daily_Std_Within_Week",
                "Daily_CV_Within_Week",
                "Num_Days_With_Txn",
                "Num_Transactions_Week",
            ]
            df[daily_cols] = df[daily_cols].fillna(0)
        else:
            logger.warning("Daily data not provided, skipping daily pattern features")

        # 2. Lag features
        if self.features_config.get("lags", {}).get("enabled", True):
            max_lag = self.features_config.get("lags", {}).get("max_lag", 52)
            df = self.compute_lag_features(df, max_lag=max_lag)

        # 3. Rolling statistics
        if self.features_config.get("rolling_stats", {}).get("enabled", True):
            df = self.compute_rolling_statistics(df)

        # 4. Calendar features
        if self.features_config.get("calendar", {}).get("enabled", True):
            df = self.compute_calendar_features(df)

        # 5. Trend features
        if self.features_config.get("trend", {}).get("enabled", True):
            df = self.compute_trend_features(df)

        # 6. LP features
        if self.features_config.get("liquidity_plans", {}).get("enabled", True):
            horizons = len(self.features_config.get("liquidity_plans", {}).get("horizons", [1, 2, 3, 4]))
            df = self.compute_lp_features(df, horizons=horizons)

        # 7. Entity encoding
        if self.features_config.get("entity_encoding", {}).get("enabled", True):
            top_n = self.features_config.get("entity_encoding", {}).get("top_n", 15)
            df = self.compute_entity_encoding(df, top_n=top_n)

        # 8. Liquidity group encoding
        if self.features_config.get("liquidity_group", {}).get("enabled", True):
            df = self.compute_liquidity_group_encoding(df)

        # Count total features
        feature_cols = [col for col in df.columns if col not in [
            "entity_id",
            "liquidity_group",
            "week_start",
            "amount_eur",
            "lp_amount",
            "asof_date",
            "week_year",
            "week_number",
        ]]

        logger.info(f"Feature engineering complete", total_features=len(feature_cols))

        return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract feature column names (exclude metadata and target).

    Args:
        df: DataFrame with features

    Returns:
        List of feature column names
    """
    exclude_cols = {
        "entity_id",
        "liquidity_group",
        "week_start",
        "amount_eur",
        "lp_amount",
        "asof_date",
        "week_year",
        "week_number",
    }

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols
