import pandas as pd
import logging

logger = logging.getLogger(__name__)


def clean_mcv2_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapt to requirements: Use SpatialDimensionValueCode as the country field (country code)
    Actual field list: Id,IndicatorCode,SpatialDimension,SpatialDimensionValueCode,...

    Args:
        df: Raw MCV2 DataFrame loaded from CSV

    Returns:
        pd.DataFrame: Cleaned MCV2 DataFrame with standardized fields
    """
    logger.info("Starting MCV2 data cleaning (using SpatialDimensionValueCode as country field)...")
    df_clean = df.copy()
    original_count = len(df_clean)

    # ---------------------- 1. Core Field Mapping (Key Adjustment: SpatialDimensionValueCode → country) ----------------------
    core_field_mapping = {
        "SpatialDimensionValueCode": "country",  # Key adjustment: Use country code as country field
        "SpatialDimension": "country_name",  # Preserve original spatial dimension as country_name (optional)
        "TimeDimensionValue": "year",  # Year
        "NumericValue": "mcv2_coverage",  # Vaccination coverage rate value
        "ParentLocation": "region"  # Parent region
    }

    # Keep only fields that actually exist in the DataFrame
    existing_core_fields = {
        actual_field: std_field
        for actual_field, std_field in core_field_mapping.items()
        if actual_field in df_clean.columns
    }
    logger.info(f"Core field matching result: {existing_core_fields}")

    # Critical validation: Ensure core fields (country/year/mcv2_coverage) exist
    required_fields = ["country", "year", "mcv2_coverage"]
    matched_required = [f for f in required_fields if f in existing_core_fields.values()]
    if len(matched_required) < 3:
        raise ValueError(
            f"Missing core fields! Matched: {matched_required}, Required: {required_fields}\n"
            f"Actual field list must include: {[k for k, v in core_field_mapping.items() if v in required_fields]}"
        )

    # Extract core fields and standardize column names
    df_clean = df_clean[list(existing_core_fields.keys())].copy()
    df_clean = df_clean.rename(columns=existing_core_fields)
    logger.info(f"Core field extraction complete: {list(df_clean.columns)}")

    # ---------------------- 2. Mandatory Non-Null Processing (Prioritize country field) ----------------------
    # 1. Country field (country code): Remove NaN/empty strings/whitespace-only values
    df_clean = df_clean.dropna(subset=["country"])
    df_clean["country"] = df_clean["country"].astype(str).str.strip()
    df_clean = df_clean[df_clean["country"] != ""]
    country_cleaned = len(df_clean)
    logger.info(
        f"Country field (country code) cleaning: Removed {original_count - country_cleaned} rows with null values")

    # 2. Year field: Convert to integer + filter reasonable range
    df_clean["year"] = pd.to_numeric(df_clean["year"], errors="coerce").astype("Int64")
    df_clean = df_clean.dropna(subset=["year"])
    df_clean = df_clean[df_clean["year"].between(1980, 2025)]
    year_cleaned = len(df_clean)
    logger.info(f"Year field cleaning: Removed {country_cleaned - year_cleaned} rows with invalid years")

    # 3. MCV2 coverage rate: Convert to float + filter 0-100% range
    df_clean["mcv2_coverage"] = pd.to_numeric(df_clean["mcv2_coverage"], errors="coerce")
    df_clean = df_clean.dropna(subset=["mcv2_coverage"])
    df_clean = df_clean[(df_clean["mcv2_coverage"] >= 0) & (df_clean["mcv2_coverage"] <= 100)]
    coverage_cleaned = len(df_clean)
    logger.info(f"Coverage rate cleaning: Removed {year_cleaned - coverage_cleaned} rows with abnormal values")

    # ---------------------- 3. Auxiliary Field Cleaning ----------------------
    # Standardize country code format (uppercase, e.g., chn → CHN)
    df_clean["country"] = df_clean["country"].str.upper()
    # Standardize region field format
    if "region" in df_clean.columns:
        df_clean["region"] = df_clean["region"].astype(str).str.strip().str.title()
    # Standardize original country name field format (optional)
    if "country_name" in df_clean.columns:
        df_clean["country_name"] = df_clean["country_name"].astype(str).str.strip().str.title()

    # ---------------------- 4. Deduplication ----------------------
    duplicate_count = df_clean.duplicated(subset=["country", "year"]).sum()
    df_clean = df_clean.drop_duplicates(subset=["country", "year"])
    logger.info(f"Deduplication complete: Removed {duplicate_count} duplicate records (country code + year)")

    # ---------------------- 5. Final Result Summary ----------------------
    cleaned_count = len(df_clean)
    logger.info(
        f"MCV2 data cleaning completed:\n"
        f"Original: {original_count} rows → Cleaned: {cleaned_count} rows (validity rate {cleaned_count / original_count * 100:.1f}%)\n"
        f"Coverage rate statistics: Avg {df_clean['mcv2_coverage'].mean():.1f}%, Range [{df_clean['mcv2_coverage'].min():.1f}%, {df_clean['mcv2_coverage'].max():.1f}%]\n"
        f"Final fields: {list(df_clean.columns)} (country field uses SpatialDimensionValueCode)"
    )

    return df_clean