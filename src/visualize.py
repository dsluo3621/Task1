import matplotlib.pyplot as plt
import pandas as pd
import os
import logging

# Use fonts guaranteed to exist on Windows/macOS/Linux
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # Fix negative sign display
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]  # Explicit fallback
# Suppress font find warnings (optional but clean)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def plot_trend(df: pd.DataFrame, country: str, metric: str = "MCV2 Coverage Rate (%)",
               save_path: str = "../exports/trend_plot.png"):
    """
    Plot MCV2 coverage rate trend chart (fixed month field error, adapted for country codes)

    Args:
        df: Trend data DataFrame (contains "Year"/"year" and coverage rate fields)
        country: Country code (e.g., CHN)
        metric: Y-axis metric name (e.g., MCV2 Coverage Rate (%))
        save_path: Path to save the plot image
    """
    try:
        # Ensure correct data fields (compatible with "Year" and "year" columns)
        x_field = "Year" if "Year" in df.columns else "year"
        y_field = metric if metric in df.columns else (df.columns[1] if len(df.columns) >= 2 else "mcv2_coverage")

        # Data validation
        if len(df) == 0:
            logger.error("Failed to plot trend chart: Data is empty!")
            raise ValueError("Trend data is empty - cannot generate chart")
        if x_field not in df.columns:
            logger.error(
                f"Failed to plot trend chart: No '{x_field}' field found. Available fields: {list(df.columns)}")
            raise ValueError(
                f"Could not interpret value `{x_field}` for `x`. An entry with this name does not appear in `data`.")
        if y_field not in df.columns:
            logger.error(
                f"Failed to plot trend chart: No '{y_field}' field found. Available fields: {list(df.columns)}")
            raise ValueError(
                f"Could not interpret value `{y_field}` for `y`. An entry with this name does not appear in `data`.")

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot trend line (adapted for country code in title)
        ax.plot(
            df[x_field],  # X-axis: Year (fixed month error)
            df[y_field],  # Y-axis: Coverage rate
            marker='o',  # Data point markers
            linewidth=2,  # Line thickness
            markersize=8,  # Marker size
            color='#2E86AB',  # Primary color
            label=f"{country} {metric}"
        )

        # Set title and labels (display country code)
        ax.set_title(
            f"{country} MCV2 Vaccination Coverage Annual Trend ({df[x_field].min()}-{df[x_field].max()})",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)

        # Style optimizations
        ax.grid(True, linestyle='--', alpha=0.7)  # Grid lines
        ax.legend(fontsize=10)  # Legend
        plt.xticks(df[x_field], rotation=45)  # Rotate year labels to prevent overlap
        plt.tight_layout()  # Auto-adjust layout

        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Trend chart plotted successfully: {save_path} (Country Code: {country})")
    except Exception as e:
        logger.error(f"Failed to plot trend chart: {str(e)}")
        raise  # Re-raise exception for upper-level handling


def plot_grouped_summary(df: pd.DataFrame, metric: str, top_n: int = 10,
                         save_path: str = "../exports/grouped_plot.png"):
    """
    Plot grouped summary comparison chart (adapted for country codes as X-axis)

    Args:
        df: Summary data DataFrame (contains country code/region and statistical metrics)
        metric: Comparison metric (e.g., MCV2 Coverage Avg (%))
        top_n: Number of top groups to display
        save_path: Path to save the plot image
    """
    try:
        # Data validation
        if len(df) == 0:
            logger.error("Failed to plot grouped comparison chart: Data is empty!")
            raise ValueError("Summary data is empty - cannot generate chart")
        if metric not in df.columns:
            logger.error(
                f"Failed to plot grouped chart: No '{metric}' field found. Available fields: {list(df.columns)}")
            raise ValueError(
                f"Could not interpret value `{metric}` for `y`. An entry with this name does not appear in `data`.")

        # Determine X-axis field (prioritize country code, then region)
        x_field = "country" if "country" in df.columns else (df.columns[0] if len(df.columns) >= 1 else "Group")
        if x_field not in df.columns:
            logger.error(
                f"Failed to plot grouped chart: No '{x_field}' field found. Available fields: {list(df.columns)}")
            raise ValueError(
                f"Could not interpret value `{x_field}` for `x`. An entry with this name does not appear in `data`.")

        # Sort by metric and take top N groups
        df_sorted = df.sort_values(metric, ascending=False).head(top_n)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot bar chart (X-axis = country code)
        bars = ax.bar(
            df_sorted[x_field],  # X-axis: Country code
            df_sorted[metric],  # Y-axis: Coverage metric
            color='#A23B72',  # Primary color
            alpha=0.8,  # Transparency
            edgecolor='white',  # Border color
            linewidth=1
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.5,
                f"{height:.1f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

        # Set title and labels (adapted for country code)
        ax.set_title(
            f"MCV2 {metric} - Top {top_n} (By Country Code)",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel("Country Code" if x_field == "country" else x_field, fontsize=12)
        ax.set_ylabel(metric, fontsize=12)

        # Style optimizations
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)  # Show only Y-axis grid
        plt.xticks(rotation=45, ha='right')  # Rotate country code labels to prevent overlap
        plt.tight_layout()  # Auto-adjust layout

        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Grouped comparison chart plotted successfully: {save_path} (X-axis: {x_field})")
    except Exception as e:
        logger.error(f"Failed to plot grouped comparison chart: {str(e)}")
        raise  # Re-raise exception for upper-level handling