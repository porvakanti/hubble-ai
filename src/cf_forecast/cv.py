"""
Cross-Validation Module

Implements rolling-origin cross-validation with embargo for time series forecasting.

Key features:
- Rolling-origin splits (expanding window)
- Embargo period (buffer between train and test)
- Minimum history requirement
- No data leakage
- Per-horizon evaluation (W1-W8)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CVSplit:
    """Cross-validation split definition."""

    split_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    embargo_weeks: int


class RollingOriginCV:
    """
    Rolling-origin cross-validation for time series.

    Implements expanding window with embargo to prevent leakage.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize rolling-origin CV.

        Args:
            config: Configuration dictionary with CV settings
        """
        self.config = config
        self.cv_config = config.get("cv", {})
        self.n_splits = self.cv_config.get("n_splits", 5)
        self.embargo_weeks = self.cv_config.get("embargo_weeks", 1)
        self.min_history_weeks = self.cv_config.get("min_history_weeks", 52)
        self.test_size_weeks = self.cv_config.get("test_size_weeks", 8)

        logger.info(
            "Initialized RollingOriginCV",
            n_splits=self.n_splits,
            embargo=self.embargo_weeks,
            min_history=self.min_history_weeks,
        )

    def generate_splits(
        self, df: pd.DataFrame, date_col: str = "week_start"
    ) -> List[CVSplit]:
        """
        Generate rolling-origin CV splits.

        Args:
            df: DataFrame with time series data
            date_col: Date column name

        Returns:
            List of CVSplit objects
        """
        logger.info("Generating rolling-origin CV splits")

        # Get date range
        df[date_col] = pd.to_datetime(df[date_col])
        min_date = df[date_col].min()
        max_date = df[date_col].max()

        total_weeks = int((max_date - min_date).days / 7)

        logger.info(
            "Date range",
            min_date=min_date,
            max_date=max_date,
            total_weeks=total_weeks,
        )

        # Calculate split points
        # Need: min_history + embargo + test_size for first split
        min_weeks_needed = self.min_history_weeks + self.embargo_weeks + self.test_size_weeks

        if total_weeks < min_weeks_needed:
            raise ValueError(
                f"Insufficient data: {total_weeks} weeks available, "
                f"{min_weeks_needed} weeks required"
            )

        # Determine test periods
        available_test_weeks = total_weeks - self.min_history_weeks - self.embargo_weeks
        test_periods_possible = available_test_weeks // self.test_size_weeks

        if test_periods_possible < self.n_splits:
            logger.warning(
                f"Only {test_periods_possible} test periods possible, "
                f"reducing n_splits from {self.n_splits}"
            )
            self.n_splits = test_periods_possible

        splits = []

        for split_id in range(self.n_splits):
            # Test period starts after min_history + embargo + (split_id * test_size)
            test_start_offset_weeks = (
                self.min_history_weeks + self.embargo_weeks + split_id * self.test_size_weeks
            )
            test_start = min_date + timedelta(weeks=test_start_offset_weeks)
            test_end = test_start + timedelta(weeks=self.test_size_weeks)

            # Train ends before embargo period
            train_end = test_start - timedelta(weeks=self.embargo_weeks)
            train_start = min_date

            splits.append(
                CVSplit(
                    split_id=split_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    embargo_weeks=self.embargo_weeks,
                )
            )

        logger.info(f"Generated {len(splits)} CV splits")

        for split in splits:
            logger.debug(
                f"Split {split.split_id}",
                train=f"{split.train_start.date()} to {split.train_end.date()}",
                test=f"{split.test_start.date()} to {split.test_end.date()}",
            )

        return splits

    def get_split_indices(
        self, df: pd.DataFrame, split: CVSplit, date_col: str = "week_start"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get train/test indices for a CV split.

        Args:
            df: DataFrame with time series data
            split: CVSplit object
            date_col: Date column name

        Returns:
            Tuple of (train_indices, test_indices)
        """
        df[date_col] = pd.to_datetime(df[date_col])

        train_mask = (df[date_col] >= split.train_start) & (df[date_col] < split.train_end)
        test_mask = (df[date_col] >= split.test_start) & (df[date_col] < split.test_end)

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        logger.debug(
            f"Split {split.split_id} indices",
            train_size=len(train_indices),
            test_size=len(test_indices),
        )

        return train_indices, test_indices

    def validate_no_leakage(
        self, df: pd.DataFrame, split: CVSplit, date_col: str = "week_start"
    ) -> bool:
        """
        Validate that there's no data leakage in split.

        Checks:
        1. Train data < test data (time ordering)
        2. Embargo gap exists
        3. No overlap

        Args:
            df: DataFrame
            split: CVSplit object
            date_col: Date column name

        Returns:
            True if no leakage, False otherwise
        """
        train_idx, test_idx = self.get_split_indices(df, split, date_col)

        # Check time ordering
        train_max_date = df.loc[train_idx, date_col].max()
        test_min_date = df.loc[test_idx, date_col].min()

        if train_max_date >= test_min_date:
            logger.error("Leakage detected: train data overlaps with test data")
            return False

        # Check embargo
        embargo_gap_weeks = (test_min_date - train_max_date).days / 7
        if embargo_gap_weeks < self.embargo_weeks:
            logger.error(
                f"Insufficient embargo: {embargo_gap_weeks} weeks, "
                f"expected {self.embargo_weeks}"
            )
            return False

        # Check no overlap
        if set(train_idx) & set(test_idx):
            logger.error("Leakage detected: train and test indices overlap")
            return False

        logger.debug(f"Split {split.split_id} validated - no leakage")
        return True


def perform_cv_evaluation(
    df: pd.DataFrame,
    cv: RollingOriginCV,
    model_fn: Any,
    feature_cols: List[str],
    target_col: str = "amount_eur",
    date_col: str = "week_start",
) -> pd.DataFrame:
    """
    Perform cross-validation evaluation.

    Args:
        df: DataFrame with features and target
        cv: RollingOriginCV object
        model_fn: Function that takes (X_train, y_train, X_test) and returns predictions
        feature_cols: List of feature column names
        target_col: Target column name
        date_col: Date column name

    Returns:
        DataFrame with out-of-fold predictions and actuals
    """
    logger.info("Performing CV evaluation")

    splits = cv.generate_splits(df, date_col=date_col)

    all_predictions = []

    for split in splits:
        # Validate no leakage
        if not cv.validate_no_leakage(df, split, date_col):
            raise ValueError(f"Data leakage detected in split {split.split_id}")

        # Get split data
        train_idx, test_idx = cv.get_split_indices(df, split, date_col)

        X_train = df.loc[train_idx, feature_cols]
        y_train = df.loc[train_idx, target_col]
        X_test = df.loc[test_idx, feature_cols]
        y_test = df.loc[test_idx, target_col]

        # Train and predict
        logger.info(
            f"Training split {split.split_id}",
            train_size=len(X_train),
            test_size=len(X_test),
        )

        y_pred = model_fn(X_train, y_train, X_test)

        # Store predictions
        pred_df = df.loc[test_idx, ["entity_id", "liquidity_group", date_col, target_col]].copy()
        pred_df["prediction"] = y_pred
        pred_df["split_id"] = split.split_id

        all_predictions.append(pred_df)

    # Combine all predictions
    oof_predictions = pd.concat(all_predictions, ignore_index=True)

    logger.info(
        "CV evaluation complete",
        total_predictions=len(oof_predictions),
        splits=len(splits),
    )

    return oof_predictions
