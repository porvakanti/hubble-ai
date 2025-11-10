"""
Data Validation Module

Implements data quality checks, schema validation, and governance rules
for actuals, liquidity plans, and FX rates data.

Validation Rules:
- Schema & dtype enforcement
- No duplicates: (entity_id, liquidity_group, week_start)
- Sign policy: TRR ≥ 0, TRP ≤ 0
- Date continuity: ≤1% gaps per entity × liquidity_group
- FX source consistency
- Fail-fast on schema; log issues on quality
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validation check."""

    passed: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

    def __bool__(self) -> bool:
        """Allow truthiness check."""
        return self.passed


class DataValidator:
    """
    Data validator for treasury cash flow forecasting.

    Validates actuals, liquidity plans, and FX rates according to
    business rules and data quality standards.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validator.

        Args:
            config: Configuration dictionary with validation rules
        """
        self.config = config
        self.validation_config = config.get("validation", {})
        logger.info("Initialized DataValidator", config=self.validation_config)

    def validate_actuals(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate actuals (daily transactions) data.

        Expected schema:
        - Entity: str (entity ID)
        - Value Date: date (posting date)
        - Amount Functional Currency: float (EUR)
        - Liquidity Group: str (TRR or TRP)

        Args:
            df: Actuals DataFrame

        Returns:
            ValidationResult with pass/fail status and messages
        """
        errors = []
        warnings = []
        metadata = {}

        logger.info("Validating actuals data", rows=len(df))

        # Schema validation
        required_cols = ["Entity", "Value Date", "Amount Functional Currency", "Liquidity Group"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(
                passed=False, errors=errors, warnings=warnings, metadata=metadata
            )

        # Rename to standard names for consistency
        df_val = df.copy()
        df_val = df_val.rename(
            columns={
                "Entity": "entity_id",
                "Value Date": "posting_date",
                "Amount Functional Currency": "amount_eur",
                "Liquidity Group": "liquidity_group",
            }
        )

        # Date type validation
        try:
            df_val["posting_date"] = pd.to_datetime(df_val["posting_date"])
        except Exception as e:
            errors.append(f"Failed to parse posting_date as datetime: {e}")
            return ValidationResult(
                passed=False, errors=errors, warnings=warnings, metadata=metadata
            )

        # Amount type validation
        try:
            df_val["amount_eur"] = pd.to_numeric(df_val["amount_eur"])
        except Exception as e:
            errors.append(f"Failed to parse amount_eur as numeric: {e}")
            return ValidationResult(
                passed=False, errors=errors, warnings=warnings, metadata=metadata
            )

        # Liquidity group validation
        valid_liq_groups = {"TRR", "TRP"}
        invalid_groups = df_val[~df_val["liquidity_group"].isin(valid_liq_groups)]
        if not invalid_groups.empty:
            unique_invalid = invalid_groups["liquidity_group"].unique().tolist()
            errors.append(
                f"Invalid liquidity groups found: {unique_invalid}. Must be TRR or TRP"
            )

        # Sign policy check
        sign_enforcement = self.validation_config.get("sign_enforcement", True)
        if sign_enforcement:
            trr_violations = df_val[
                (df_val["liquidity_group"] == "TRR") & (df_val["amount_eur"] < 0)
            ]
            trp_violations = df_val[
                (df_val["liquidity_group"] == "TRP") & (df_val["amount_eur"] > 0)
            ]

            if not trr_violations.empty:
                pct = 100 * len(trr_violations) / len(df_val[df_val["liquidity_group"] == "TRR"])
                warnings.append(
                    f"TRR sign violations: {len(trr_violations)} rows ({pct:.2f}%) with negative amounts"
                )

            if not trp_violations.empty:
                pct = 100 * len(trp_violations) / len(df_val[df_val["liquidity_group"] == "TRP"])
                warnings.append(
                    f"TRP sign violations: {len(trp_violations)} rows ({pct:.2f}%) with positive amounts"
                )

        # Check for nulls
        null_amounts = df_val["amount_eur"].isnull().sum()
        if null_amounts > 0:
            warnings.append(f"Found {null_amounts} null amounts ({100*null_amounts/len(df_val):.2f}%)")

        # Date continuity check
        date_gaps = self._check_date_continuity(df_val)
        if date_gaps:
            max_gap_pct = self.validation_config.get("quality", {}).get("max_date_gaps_pct", 0.01)
            for entity, liq_group, gap_pct in date_gaps:
                if gap_pct > max_gap_pct:
                    warnings.append(
                        f"Date gaps for {entity} {liq_group}: {gap_pct*100:.2f}% of days missing"
                    )

        # Duplicates check
        dup_check = df_val.groupby(["entity_id", "liquidity_group", "posting_date"]).size()
        duplicates = dup_check[dup_check > 1]
        if not duplicates.empty:
            warnings.append(
                f"Found {len(duplicates)} duplicate (entity, liquidity_group, date) combinations"
            )

        # Metadata
        metadata["total_rows"] = len(df_val)
        metadata["entities"] = df_val["entity_id"].nunique()
        metadata["date_range"] = (
            df_val["posting_date"].min().strftime("%Y-%m-%d"),
            df_val["posting_date"].max().strftime("%Y-%m-%d"),
        )
        metadata["trr_rows"] = len(df_val[df_val["liquidity_group"] == "TRR"])
        metadata["trp_rows"] = len(df_val[df_val["liquidity_group"] == "TRP"])
        metadata["total_amount"] = float(df_val["amount_eur"].sum())

        passed = len(errors) == 0
        logger.info(
            "Actuals validation complete",
            passed=passed,
            errors=len(errors),
            warnings=len(warnings),
        )

        return ValidationResult(passed=passed, errors=errors, warnings=warnings, metadata=metadata)

    def validate_liquidity_plan(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate liquidity plan data.

        Expected schema (flexible - will extract key columns):
        - Entity (or similar): str
        - Liquidity Group (or similar): str
        - Item's Date (or similar): date
        - Amount (or similar): float

        Args:
            df: Liquidity Plan DataFrame

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        metadata = {}

        logger.info("Validating liquidity plan data", rows=len(df))

        # Try to identify key columns (flexible naming)
        entity_col = self._find_column(df, ["Entity", "Entity ID", "entity_id", "Entity Name"])
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
                "Group Description",
            ],
        )

        missing = []
        if entity_col is None:
            missing.append("Entity")
        if date_col is None:
            missing.append("Date")
        if amount_col is None:
            missing.append("Amount")
        if liq_group_col is None:
            missing.append("Liquidity Group")

        if missing:
            errors.append(f"Could not identify required columns: {missing}")
            return ValidationResult(
                passed=False, errors=errors, warnings=warnings, metadata=metadata
            )

        # Create standardized view
        df_val = df[[entity_col, date_col, amount_col, liq_group_col]].copy()
        df_val.columns = ["entity_id", "plan_date", "lp_amount", "liquidity_group"]

        # Date parsing
        try:
            df_val["plan_date"] = pd.to_datetime(df_val["plan_date"])
        except Exception as e:
            errors.append(f"Failed to parse plan_date: {e}")
            return ValidationResult(
                passed=False, errors=errors, warnings=warnings, metadata=metadata
            )

        # Amount parsing
        try:
            df_val["lp_amount"] = pd.to_numeric(df_val["lp_amount"])
        except Exception as e:
            errors.append(f"Failed to parse lp_amount: {e}")
            return ValidationResult(
                passed=False, errors=errors, warnings=warnings, metadata=metadata
            )

        # Liquidity group validation
        valid_groups = {"TRR", "TRP"}
        invalid = df_val[~df_val["liquidity_group"].isin(valid_groups)]
        if not invalid.empty:
            pct = 100 * len(invalid) / len(df_val)
            warnings.append(
                f"Found {len(invalid)} rows ({pct:.2f}%) with invalid liquidity groups. "
                "Valid: TRR, TRP"
            )

        # Sign policy
        trr_violations = df_val[
            (df_val["liquidity_group"] == "TRR") & (df_val["lp_amount"] < 0)
        ]
        trp_violations = df_val[
            (df_val["liquidity_group"] == "TRP") & (df_val["lp_amount"] > 0)
        ]

        if not trr_violations.empty:
            warnings.append(
                f"LP TRR sign violations: {len(trr_violations)} rows with negative amounts"
            )
        if not trp_violations.empty:
            warnings.append(
                f"LP TRP sign violations: {len(trp_violations)} rows with positive amounts"
            )

        # Nulls
        null_amounts = df_val["lp_amount"].isnull().sum()
        if null_amounts > 0:
            warnings.append(
                f"Found {null_amounts} null LP amounts ({100*null_amounts/len(df_val):.2f}%)"
            )

        # Metadata
        metadata["total_rows"] = len(df_val)
        metadata["entities"] = df_val["entity_id"].nunique()
        metadata["date_range"] = (
            df_val["plan_date"].min().strftime("%Y-%m-%d"),
            df_val["plan_date"].max().strftime("%Y-%m-%d"),
        )

        passed = len(errors) == 0
        logger.info(
            "LP validation complete", passed=passed, errors=len(errors), warnings=len(warnings)
        )

        return ValidationResult(passed=passed, errors=errors, warnings=warnings, metadata=metadata)

    def validate_fx_rates(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate FX rates data.

        Expected schema:
        - Date: date
        - Currency columns: float

        Args:
            df: FX rates DataFrame (ECB format)

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        metadata = {}

        logger.info("Validating FX rates data", rows=len(df))

        # Check for Date column
        if "Date" not in df.columns:
            errors.append("Missing 'Date' column in FX rates")
            return ValidationResult(
                passed=False, errors=errors, warnings=warnings, metadata=metadata
            )

        df_val = df.copy()

        # Parse dates
        try:
            df_val["Date"] = pd.to_datetime(df_val["Date"])
        except Exception as e:
            errors.append(f"Failed to parse Date column: {e}")
            return ValidationResult(
                passed=False, errors=errors, warnings=warnings, metadata=metadata
            )

        # Check for currency columns
        currency_cols = [col for col in df_val.columns if col != "Date"]
        if not currency_cols:
            errors.append("No currency columns found in FX rates")
            return ValidationResult(
                passed=False, errors=errors, warnings=warnings, metadata=metadata
            )

        # Check for nulls
        null_pct_by_currency = {}
        for col in currency_cols:
            null_pct = 100 * df_val[col].isnull().sum() / len(df_val)
            if null_pct > 0:
                null_pct_by_currency[col] = null_pct

        if null_pct_by_currency:
            warnings.append(f"Currencies with missing rates: {null_pct_by_currency}")

        # Date gaps
        df_val = df_val.sort_values("Date")
        date_diffs = df_val["Date"].diff()
        large_gaps = date_diffs[date_diffs > pd.Timedelta(days=7)]
        if not large_gaps.empty:
            warnings.append(f"Found {len(large_gaps)} gaps > 7 days in FX rates")

        # Metadata
        metadata["total_rows"] = len(df_val)
        metadata["currencies"] = len(currency_cols)
        metadata["date_range"] = (
            df_val["Date"].min().strftime("%Y-%m-%d"),
            df_val["Date"].max().strftime("%Y-%m-%d"),
        )

        passed = len(errors) == 0
        logger.info(
            "FX validation complete", passed=passed, errors=len(errors), warnings=len(warnings)
        )

        return ValidationResult(passed=passed, errors=errors, warnings=warnings, metadata=metadata)

    def validate_entity_mapping(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate entity-liquidity mapping.

        Expected schema:
        - Entity: str
        - TRR_active: bool
        - TRP_active: bool

        Args:
            df: Entity mapping DataFrame

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        metadata = {}

        logger.info("Validating entity-liquidity mapping", rows=len(df))

        # Check required columns
        required_cols = ["Entity", "TRR_active", "TRP_active"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(
                passed=False, errors=errors, warnings=warnings, metadata=metadata
            )

        # Check for duplicates
        duplicates = df["Entity"].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate entities in mapping")

        # Check boolean values
        for col in ["TRR_active", "TRP_active"]:
            unique_vals = df[col].unique()
            valid_vals = {True, False, "TRUE", "FALSE", "True", "False", 1, 0}
            invalid = [v for v in unique_vals if v not in valid_vals and pd.notna(v)]
            if invalid:
                warnings.append(f"Non-boolean values in {col}: {invalid}")

        # Check for entities with no active liquidity groups
        df_val = df.copy()
        df_val["TRR_active"] = df_val["TRR_active"].astype(str).str.upper() == "TRUE"
        df_val["TRP_active"] = df_val["TRP_active"].astype(str).str.upper() == "TRUE"

        no_active = df_val[~df_val["TRR_active"] & ~df_val["TRP_active"]]
        if not no_active.empty:
            warnings.append(
                f"{len(no_active)} entities have no active liquidity groups: {no_active['Entity'].tolist()}"
            )

        # Metadata
        metadata["total_entities"] = len(df_val)
        metadata["trr_active_count"] = df_val["TRR_active"].sum()
        metadata["trp_active_count"] = df_val["TRP_active"].sum()
        metadata["both_active"] = (df_val["TRR_active"] & df_val["TRP_active"]).sum()

        passed = len(errors) == 0
        logger.info(
            "Entity mapping validation complete",
            passed=passed,
            errors=len(errors),
            warnings=len(warnings),
        )

        return ValidationResult(passed=passed, errors=errors, warnings=warnings, metadata=metadata)

    def _check_date_continuity(
        self, df: pd.DataFrame
    ) -> List[Tuple[str, str, float]]:
        """
        Check for date gaps per entity-liquidity group.

        Args:
            df: DataFrame with entity_id, liquidity_group, posting_date

        Returns:
            List of (entity, liq_group, gap_pct) tuples for problematic combinations
        """
        gaps = []

        for (entity, liq_group), group in df.groupby(["entity_id", "liquidity_group"]):
            dates = group["posting_date"].sort_values()
            if len(dates) < 2:
                continue

            min_date = dates.min()
            max_date = dates.max()
            expected_days = (max_date - min_date).days + 1
            actual_days = dates.nunique()

            if expected_days > 0:
                gap_pct = 1 - (actual_days / expected_days)
                if gap_pct > 0.01:  # More than 1% missing
                    gaps.append((str(entity), str(liq_group), gap_pct))

        return gaps

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """
        Find first matching column name from candidates.

        Args:
            df: DataFrame
            candidates: List of possible column names

        Returns:
            Matched column name or None
        """
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None


def quick_check(config_path: str) -> None:
    """
    Quick data quality check for CLI usage.

    Args:
        config_path: Path to config.yml
    """
    import yaml

    from cf_forecast.io import get_backend

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize backend
    backend = get_backend(
        profile=config["data_backend"]["profile"],
        config=config["data_backend"]["options"],
    )

    # Initialize validator
    validator = DataValidator(config)

    # Check actuals
    try:
        actuals = backend.read_csv(f"{config['paths']['raw']}actuals_curated.csv")
        result = validator.validate_actuals(actuals)
        print(f"\n✓ Actuals: {'PASS' if result.passed else 'FAIL'}")
        if result.errors:
            print(f"  Errors: {result.errors}")
        if result.warnings:
            print(f"  Warnings: {result.warnings[:3]}")  # Show first 3
        print(f"  Metadata: {result.metadata}")
    except Exception as e:
        print(f"\n✗ Actuals: ERROR - {e}")

    # Check entity mapping
    try:
        mapping = backend.read_csv(f"{config['paths']['reference']}Entity-Liquidity_Map.csv")
        result = validator.validate_entity_mapping(mapping)
        print(f"\n✓ Entity Mapping: {'PASS' if result.passed else 'FAIL'}")
        if result.errors:
            print(f"  Errors: {result.errors}")
        print(f"  Metadata: {result.metadata}")
    except Exception as e:
        print(f"\n✗ Entity Mapping: ERROR - {e}")

    print("\nQuick check complete.")
