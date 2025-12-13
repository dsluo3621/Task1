import logging
import os
import pandas as pd


def init_logger(log_file: str = "../logs/mcv2_analysis.log"):
    """Initialize the logging system"""
    # Create log directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure log format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),  # Write to file
            logging.StreamHandler()  # Output to console
        ]
    )
    logging.info("Logging system initialized successfully")


def export_to_csv(df: pd.DataFrame, save_path: str) -> bool:
    """Export DataFrame to CSV file"""
    try:
        # Create export directory (throw error if parent directory does not exist)
        parent_dir = os.path.dirname(save_path)
        if not os.path.exists(parent_dir):
            raise PermissionError(f"Directory does not exist and cannot be created: {parent_dir}")

        # Export to CSV (do not retain index, encoding is utf-8)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        logging.info(f"Data exported successfully: {save_path}")
        return True
    except PermissionError as e:
        logging.error(f"Data export failed (insufficient permissions/directory does not exist): {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Data export failed: {str(e)}")
        return False