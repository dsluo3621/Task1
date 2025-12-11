import pandas as pd
import logging

logger = logging.getLogger(__name__)


def filter_mcv2_data(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    MCV2 data filtering function (adapted for country code filtering)

    Args:
        df: Raw/full MCV2 dataset (country field = country code)
        filters: Filter criteria dictionary supporting:
                 - country: List of country codes (e.g., ["CHN", "USA"])
                 - year_start: Start year (integer)
                 - year_end: End year (integer)
                 - region: List of regions (e.g., ["Africa"])
                 - mcv2_coverage_min: Minimum vaccination coverage rate (float)

    Returns:
        pd.DataFrame: Filtered MCV2 dataset
    """
    logger.info(f"Starting MCV2 data filtering with criteria: {filters}")
    df_filtered = df.copy()

    # 1. Filter by country code (core adaptation)
    if "country" in filters and filters["country"]:
        df_filtered = df_filtered[df_filtered["country"].isin(filters["country"])]
        logger.info(f"Filtered by country codes: {filters['country']}, remaining rows: {len(df_filtered)}")

    # 2. Filter by year range
    if "year_start" in filters and filters["year_start"]:
        year_start = int(filters["year_start"])
        df_filtered = df_filtered[df_filtered["year"] >= year_start]
        logger.info(f"Filtered by start year {year_start}, remaining rows: {len(df_filtered)}")
    if "year_end" in filters and filters["year_end"]:
        year_end = int(filters["year_end"])
        df_filtered = df_filtered[df_filtered["year"] <= year_end]
        logger.info(f"Filtered by end year {year_end}, remaining rows: {len(df_filtered)}")

    # 3. Filter by region
    if "region" in filters and filters["region"]:
        df_filtered = df_filtered[df_filtered["region"].isin(filters["region"])]
        logger.info(f"Filtered by regions: {filters['region']}, remaining rows: {len(df_filtered)}")

    # 4. Filter by minimum coverage rate
    if "mcv2_coverage_min" in filters and filters["mcv2_coverage_min"]:
        coverage_min = float(filters["mcv2_coverage_min"])
        df_filtered = df_filtered[df_filtered["mcv2_coverage"] >= coverage_min]
        logger.info(f"Filtered by minimum coverage rate {coverage_min}%, remaining rows: {len(df_filtered)}")

    logger.info(f"MCV2 data filtering completed, final row count: {len(df_filtered)}")
    return df_filtered


def summarize_mcv2_data(df: pd.DataFrame, group_by: str = "country") -> pd.DataFrame:
    """
    MCV2 data summarization (group by country code/region)

    Args:
        df: Filtered MCV2 dataset
        group_by: Grouping field (country = country code / region)

    Returns:
        pd.DataFrame: Summarized dataset with mean/max/min/count metrics
    """
    if group_by not in df.columns:
        group_by = "country"
        logger.warning(f"Grouping field '{group_by}' does not exist. Defaulting to 'country' (country code)")

    # Group by specified field and calculate key coverage rate statistics
    summary = df.groupby(group_by)["mcv2_coverage"].agg(
        coverage_mean=("mean"),
        coverage_max=("max"),
        coverage_min=("min"),
        record_count=("count")
    ).round(1).reset_index()

    # Rename columns (adapted for visualization compatibility)
    summary.rename(columns={
        group_by: group_by if group_by == "country" else "Region",
        "coverage_mean": "MCV2 Coverage Avg (%)",
        "coverage_max": "MCV2 Coverage Max (%)",
        "coverage_min": "MCV2 Coverage Min (%)",
        "record_count": "Record Count"
    }, inplace=True)

    logger.info(f"MCV2 data summarization completed (grouped by {group_by}), total groups: {len(summary)}")
    return summary


def mcv2_trend_analysis(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """
    MCV2 trend analysis (by country code)

    Args:
        df: Filtered MCV2 dataset
        country_code: Target country code (e.g., CHN)

    Returns:
        pd.DataFrame: Annual coverage rate trend for the specified country code
    """
    # Filter data for specified country code and sort by year
    trend_df = df[df["country"] == country_code].copy()
    trend_df = trend_df.sort_values("year")[["year", "mcv2_coverage"]].reset_index(drop=True)

    # Rename columns for visualization compatibility
    trend_df.rename(columns={
        "year": "Year",
        "mcv2_coverage": "MCV2 Coverage Rate (%)"
    }, inplace=True)

    logger.info(f"Trend analysis completed for {country_code}, total annual records: {len(trend_df)}")
    return trend_df