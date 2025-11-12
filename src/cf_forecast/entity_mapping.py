"""
Entity-Liquidity Group Tier Mapping

Defines which entity-liquidity combinations should be forecasted with ML (Tier 1)
vs passed through from LP (Tier 2).

Based on: 27-week backtest validation (March-September 2025)
Decision date: 2025-11-11
Documentation: artifacts/backtesting/STAKEHOLDER_SUMMARY.md
"""

from typing import List, Tuple, Dict
import pandas as pd

# ============================================================================
# TIER 2: LP PASSTHROUGH (14 combinations, 3.03% of portfolio)
# ============================================================================

TIER_2_COMBINATIONS: List[Tuple[str, str]] = [
    # Entities not in training data
    ('82J', 'TRR'),   # ASSS USA - not in data
    ('82J', 'TRP'),   # ASSS USA - not in data
    ('25A4', 'TRR'),  # Universal Stainless - not in data
    ('25A4', 'TRP'),  # Universal Stainless - not in data

    # Very poor WAPE + not expected per ref map
    ('T056', 'TRR'),  # 292% WAPE, ref map: "Usually just TRP"
    ('14C1', 'TRR'),  # 126% WAPE, 99 weeks, ref map: "Usually just TRP"
    ('17C7', 'TRR'),  # 101% WAPE, data quality issues (€0.01 actuals)
    ('11G5', 'TRR'),  # 81% WAPE, 77 weeks, ref map: "Usually just TRP"
    ('V002', 'TRR'),  # 54% WAPE, ref map: "Usually just TRP"

    # Insufficient data + poor quality
    ('20B2', 'TRR'),  # 137% WAPE, only 57 weeks

    # Business logic (ref map guidance)
    ('20B2', 'TRP'),  # Ref map: "No LP inputs presently" + consistency (entire entity)
    ('T055', 'TRP'),  # Ref map: "Usually TRR, with small amount of payables"
    ('V265', 'TRP'),  # Ref map: "Usually TRR, with small amount of payables"
    ('86W', 'TRP'),   # Ref map: "Usually TRR, with small amount of payables"
]

# ============================================================================
# TIER 1: ML FORECASTING - HIGH UNCERTAINTY (4 combinations, 0.69% of portfolio)
# ============================================================================

HIGH_UNCERTAINTY_COMBINATIONS: List[Tuple[str, str]] = [
    ('4B9', 'TRP'),   # 58.6% WAPE, €40M - Ref map: "LP updates sometimes missing"
    ('V756', 'TRP'),  # 51.7% WAPE, €147M (0.44% portfolio - material!)
    ('057', 'TRP'),   # 46.3% WAPE, €23M
    ('V508', 'TRP'),  # 45.9% WAPE, €21M
]

# ============================================================================
# TIER 1: ML FORECASTING - STANDARD (16 combinations, 96.28% of portfolio)
# ============================================================================
# All combinations not in TIER_2 or HIGH_UNCERTAINTY

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_tier(entity_id: str, liquidity_group: str) -> int:
    """
    Get tier for entity-liquidity combination.

    Returns:
        1: ML forecasting (Tier 1)
        2: LP passthrough (Tier 2)
    """
    if (entity_id, liquidity_group) in TIER_2_COMBINATIONS:
        return 2
    return 1


def is_high_uncertainty(entity_id: str, liquidity_group: str) -> bool:
    """
    Check if combination should be flagged as high uncertainty.

    High uncertainty forecasts should:
    - Be reported with wider confidence intervals
    - Be monitored more closely
    - Have manual override option in production
    """
    return (entity_id, liquidity_group) in HIGH_UNCERTAINTY_COMBINATIONS


def filter_tier_1_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe to only Tier 1 (ML forecasting) combinations.

    Args:
        df: DataFrame with 'entity_id' and 'liquidity_group' columns

    Returns:
        Filtered DataFrame with only Tier 1 combinations
    """
    mask = df.apply(
        lambda row: get_tier(row['entity_id'], row['liquidity_group']) == 1,
        axis=1
    )
    return df[mask].copy()


def filter_tier_2_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe to only Tier 2 (LP passthrough) combinations.

    Args:
        df: DataFrame with 'entity_id' and 'liquidity_group' columns

    Returns:
        Filtered DataFrame with only Tier 2 combinations
    """
    mask = df.apply(
        lambda row: get_tier(row['entity_id'], row['liquidity_group']) == 2,
        axis=1
    )
    return df[mask].copy()


def add_tier_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'tier' column to dataframe.

    Args:
        df: DataFrame with 'entity_id' and 'liquidity_group' columns

    Returns:
        DataFrame with added 'tier' column
    """
    df = df.copy()
    df['tier'] = df.apply(
        lambda row: get_tier(row['entity_id'], row['liquidity_group']),
        axis=1
    )
    return df


def add_high_uncertainty_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'high_uncertainty' boolean column to dataframe.

    Args:
        df: DataFrame with 'entity_id' and 'liquidity_group' columns

    Returns:
        DataFrame with added 'high_uncertainty' column
    """
    df = df.copy()
    df['high_uncertainty'] = df.apply(
        lambda row: is_high_uncertainty(row['entity_id'], row['liquidity_group']),
        axis=1
    )
    return df


def get_tier_summary() -> Dict:
    """
    Get summary of tier mapping.

    Returns:
        Dictionary with tier statistics
    """
    return {
        'tier_2_count': len(TIER_2_COMBINATIONS),
        'high_uncertainty_count': len(HIGH_UNCERTAINTY_COMBINATIONS),
        'tier_2_combinations': TIER_2_COMBINATIONS,
        'high_uncertainty_combinations': HIGH_UNCERTAINTY_COMBINATIONS,
    }


def print_tier_summary():
    """Print human-readable tier mapping summary."""
    print("=" * 80)
    print("ENTITY-LIQUIDITY TIER MAPPING SUMMARY")
    print("=" * 80)
    print()

    print(f"TIER 2 (LP Passthrough): {len(TIER_2_COMBINATIONS)} combinations")
    print("  - 82J (TRR, TRP): Not in training data")
    print("  - 25A4 (TRR, TRP): Not in training data")
    print("  - T056-TRR: 292% WAPE, not expected")
    print("  - 14C1-TRR: 126% WAPE, not expected")
    print("  - 17C7-TRR: 101% WAPE, data quality")
    print("  - 11G5-TRR: 81% WAPE, insufficient data")
    print("  - V002-TRR: 54% WAPE, not expected")
    print("  - 20B2 (TRR, TRP): Data quality + business logic")
    print("  - T055-TRP, V265-TRP, 86W-TRP: Ref map 'small payables'")
    print()

    print(f"TIER 1 - HIGH UNCERTAINTY: {len(HIGH_UNCERTAINTY_COMBINATIONS)} combinations")
    print("  - 4B9-TRP: 58.6% WAPE (€40M)")
    print("  - V756-TRP: 51.7% WAPE (€147M - material!)")
    print("  - 057-TRP: 46.3% WAPE (€23M)")
    print("  - V508-TRP: 45.9% WAPE (€21M)")
    print()

    print("TIER 1 - STANDARD: ~16 combinations (all others)")
    print("  - Best: V798-TRR (9.6% WAPE)")
    print("  - Largest: 14C1-TRP (19.6% portfolio)")
    print()

    print("PORTFOLIO COVERAGE:")
    print("  - Tier 2 excluded: 3.03%")
    print("  - Tier 1 (ML): 96.97%")
    print("=" * 80)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_no_duplicates():
    """Ensure no entity-liq appears in both Tier 2 and High Uncertainty."""
    tier_2_set = set(TIER_2_COMBINATIONS)
    high_unc_set = set(HIGH_UNCERTAINTY_COMBINATIONS)

    overlap = tier_2_set & high_unc_set
    if overlap:
        raise ValueError(f"Overlap between Tier 2 and High Uncertainty: {overlap}")

    print("✓ Validation passed: No duplicates between Tier 2 and High Uncertainty")


if __name__ == "__main__":
    # Run validation
    validate_no_duplicates()

    # Print summary
    print_tier_summary()
