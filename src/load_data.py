import pandas as pd
import sqlite3
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_csv_to_df(file_path: str = "../data/MCV2.csv") -> Optional[pd.DataFrame]:
    """Load MCV2.csv (adapted for actual 25-field structure)"""
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        logger.info(f"Successfully loaded MCV2 data from: {file_path}, raw row count: {len(df)}")
        logger.debug(f"Actual field list: {list(df.columns)}")
        return df
    except FileNotFoundError:
        logger.error(f"MCV2 file not found: {file_path}, please check if path is ../data/MCV2.csv")
        return None
    except Exception as e:
        logger.error(f"Failed to load MCV2 data: {str(e)}")
        return None


def df_to_sqlite(df: pd.DataFrame, db_name: str = "vaccine_data.db", table_name: str = "mcv2_vaccination") -> bool:
    """
    Write MCV2 data to SQLite (adapted for country field as country code)

    Args:
        df: Cleaned MCV2 DataFrame
        db_name: Name of SQLite database file
        table_name: Name of target table for MCV2 data

    Returns:
        bool: True if write succeeds, False if fails
    """
    try:
        # Final validation before write: ensure no null values
        null_check = {col: df[col].isnull().sum() for col in df.columns}
        if any(v > 0 for v in null_check.values()):
            logger.error(f"Null values detected before write: {null_check}, cannot write to database")
            return False

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # 1. Drop old table to avoid schema conflicts
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

        # 2. Create new table (country = country code, added country_name field)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country TEXT NOT NULL,                 -- Country code (SpatialDimensionValueCode, non-null)
            country_name TEXT,                     -- Original country name (SpatialDimension, optional)
            year INTEGER NOT NULL,                 -- Year (non-null)
            mcv2_coverage FLOAT NOT NULL,          -- Vaccination coverage rate (non-null)
            region TEXT,                           -- Region (optional)
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_sql)
        conn.commit()
        logger.info(f"Created new table {table_name}, 'country' field uses country codes (SpatialDimensionValueCode)")

        # 3. Only write columns that exist in the table
        table_columns = [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name})")]
        df_to_write = df[[col for col in df.columns if col in table_columns]]

        # 4. Execute write operation
        df_to_write.to_sql(table_name, conn, if_exists="append", index=False)
        conn.close()

        logger.info(
            f"Successfully wrote MCV2 data: {len(df_to_write)} rows, 'country' field is {df_to_write['country'].dtype} type (country code)")
        return True
    except Exception as e:
        logger.error(f"Failed to write MCV2 data to database: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return False


def load_sqlite_to_df(db_name: str = "vaccine_data.db", table_name: str = "mcv2_vaccination") -> Optional[pd.DataFrame]:
    """Load MCV2 data from SQLite (country field as country code)

    Args:
        db_name: Name of SQLite database file
        table_name: Name of target table containing MCV2 data

    Returns:
        Optional[pd.DataFrame]: Loaded data if successful, None if failed/empty
    """
    try:
        conn = sqlite3.connect(db_name)
        # Check if table exists first
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        if not cursor.fetchone():
            logger.warning(f"{table_name} table does not exist, returning None")
            conn.close()
            return None

        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        if len(df) == 0:
            logger.warning(f"{table_name} table contains no data, returning None")
            return None
        logger.info(
            f"Loaded MCV2 data from database: {len(df)} rows, 'country' field uses codes (e.g., {df['country'].head(1).values[0]})")
        return df
    except sqlite3.OperationalError:
        logger.warning(f"{table_name} table does not exist, please load CSV data first")
        return None
    except Exception as e:
        logger.error(f"Failed to load MCV2 data from database: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return None