import sys
import os

# Resolve Python path issues (add if ModuleNotFoundError occurs)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.load_data import load_csv_to_df, df_to_sqlite, load_sqlite_to_df
from src.clean_data import clean_mcv2_data
from src.filter_summary import filter_mcv2_data, summarize_mcv2_data, mcv2_trend_analysis
from src.visualize import plot_trend, plot_grouped_summary
from src.export_log import init_logger, export_to_csv
import logging

# Initialize logging system
init_logger()
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("          MCV2 Vaccine Coverage Data Insight Dashboard")
    print("          (Filter by Country Code <e.g., CHN, USA>)")
    print("=" * 60)

    # Step 1: Load MCV2 data to SQLite database (execute on first run)
    csv_path = "../data/MCV2.csv"  # Path to MCV2 dataset
    # Core fix: Check if DataFrame exists (use is None) instead of direct boolean check
    db_df = load_sqlite_to_df(table_name="mcv2_vaccination")
    if db_df is None:  # Load from CSV if no data in database
        logger.info("MCV2 table not found/no data in database, loading from CSV...")
        raw_df = load_csv_to_df(csv_path)
        if raw_df is None:
            print("Error: Failed to load MCV2.csv, please check the file path!")
            return
        # Clean data (use MCV2-specific cleaning function)
        clean_df = clean_mcv2_data(raw_df)
        # Write cleaned data to SQLite (specify MCV2 dedicated table)
        if not df_to_sqlite(clean_df, table_name="mcv2_vaccination"):
            print("Error: Failed to write MCV2 data to database!")
            return
        print("MCV2 data loaded and cleaned successfully, stored in local database.")
        # Reload data from database to memory
        df = load_sqlite_to_df(table_name="mcv2_vaccination")
    else:
        print("Successfully connected to local database, MCV2 data loaded.")
        df = db_df  # Use database data directly

    # Step 2: Validate loaded data
    if df is None or len(df) == 0:
        print("Error: No MCV2 data found in database! Ensure CSV file is loaded correctly.")
        return

    # Step 3: CLI interactive menu (adapted for country code filtering)
    current_df = df  # Initialize with full MCV2 dataset
    current_summary_df = None  # Initialize summary data
    current_trend_df = None    # Initialize trend analysis data

    # Preload all country codes for user reference
    all_country_codes = sorted(df["country"].unique())
    logger.info(f"Available country codes: {all_country_codes[:10]}... (Total: {len(all_country_codes)})")

    while True:
        print("\n" + "-" * 60)
        print("Please select a function:")
        print("1. MCV2 Data Filter (Country Code/Year/Region/Coverage Threshold)")
        print("2. MCV2 Data Summary (Group by Country Code/Region)")
        print("3. MCV2 Trend Analysis (Annual coverage changes by Country Code)")
        print("4. MCV2 Visualization (Trend Chart / Group Comparison Chart)")
        print("5. Export MCV2 Data (CSV Format)")
        print("6. View All Available Country Codes")
        print("7. Exit System")
        print("-" * 60)

        choice = input("Enter function number (1-7): ").strip()

        if choice == "1":
            # Function 1: MCV2 Data Filter (core adjustment: filter by country code)
            print("\n=== MCV2 Data Filter (Filter by Country Code) ===")
            # Show hint: inform user to input country codes (e.g., CHN, USA)
            print(f"Tip: Example country codes: {all_country_codes[:5]} (Total: {len(all_country_codes)})")
            country_codes = input("1. Target country codes (multiple separated by commas, e.g., CHN,USA; press Enter for all): ").strip()

            year_start = input("2. Start year (e.g., 2010; press Enter for no limit): ").strip()
            year_end = input("3. End year (e.g., 2020; press Enter for no limit): ").strip()
            region = input("4. Region (e.g., Africa; press Enter for all): ").strip()
            coverage_min = input("5. Minimum coverage rate (e.g., 50; press Enter for no limit): ").strip()

            # Build filter dictionary (adapted for country codes)
            filters = {}
            if country_codes:
                # Split multiple country codes and convert to uppercase (compatible with lowercase input)
                code_list = [code.strip().upper() for code in country_codes.split(",")]
                # Validate if input country codes exist
                invalid_codes = [code for code in code_list if code not in all_country_codes]
                if invalid_codes:
                    print(f"Warning: The following country codes do not exist: {invalid_codes} (will be ignored)!")
                    code_list = [code for code in code_list if code not in invalid_codes]
                if code_list:
                    filters["country"] = code_list  # Key remains "country" (matches database field)
                else:
                    print("Warning: No valid country codes, will not filter by country!")

            if year_start:
                filters["year_start"] = year_start
            if year_end:
                filters["year_end"] = year_end
            if region:
                filters["region"] = [region.strip().title()]  # Standardize to title case
            if coverage_min:
                filters["mcv2_coverage_min"] = coverage_min

            # Execute MCV2 filtering
            filtered_df = filter_mcv2_data(df, filters)
            print(f"\nFiltering complete! {len(filtered_df)} records found (country code filter applied)")
            print("Filter results preview (first 5 rows):")
            # Prioritize showing country code, year, coverage rate in preview
            preview_cols = ["country", "year", "mcv2_coverage", "region"]
            preview_cols = [col for col in preview_cols if col in filtered_df.columns]
            print(filtered_df[preview_cols].head().to_string(index=False))
            current_df = filtered_df  # Save filtered data for subsequent use

        elif choice == "2":
            # Function 2: MCV2 Data Summary (group by country code)
            print("\n=== MCV2 Data Summary (Group by Country Code) ===")
            group_by = input("Select grouping field (country/region; default=country): ").strip() or "country"
            # Ensure grouping field exists
            if group_by not in current_df.columns:
                print(f"Error: Grouping field '{group_by}' does not exist. Defaulting to 'country' (country code).")
                group_by = "country"
            # Execute MCV2 summary
            summary_df = summarize_mcv2_data(current_df, group_by)
            print("\n MCV2 Data Summary Results (Grouped by Country Code):")
            print(summary_df.to_string(index=False))
            current_summary_df = summary_df  # Save summary data

        elif choice == "3":
            # Function 3: MCV2 Trend Analysis (by country code)
            print("\n=== MCV2 Trend Analysis (By Country Code) ===")
            print(f"Tip: Example country codes: {all_country_codes[:5]}")
            country_code = input("Enter target country code (e.g., CHN): ").strip().upper()
            if not country_code:
                print("Error: Country code cannot be empty!")
                continue
            if country_code not in all_country_codes:
                print(f"Error: Country code '{country_code}' does not exist! Available codes: {all_country_codes[:10]}...")
                continue
            # Execute trend analysis
            trend_df = mcv2_trend_analysis(current_df, country_code)
            if len(trend_df) == 0:
                print(f"No MCV2 trend data found for {country_code}!")
                continue
            print(f"\n{country_code} MCV2 Coverage Annual Trend:")
            print(trend_df.to_string(index=False))
            current_trend_df = trend_df  # Save trend data

        elif choice == "4":
            # Function 4: MCV2 Visualization (adapted for country code)
            print("\n=== MCV2 Visualization (By Country Code) ===")
            if current_trend_df is None and current_summary_df is None:
                print("Please first run 'Trend Analysis' or 'Data Summary' to generate visualization data!")
                continue
            vis_choice = input("Select visualization type (1-Trend Chart  2-Group Comparison Chart): ").strip()
            if vis_choice == "1" and current_trend_df is not None:
                country_code = input("Enter country code for trend chart title: ").strip().upper() or "Target Country Code"
                # Plot trend chart (adapted for country code)
                plot_trend(
                    current_trend_df,
                    country=country_code,
                    metric="MCV2 Coverage Rate (%)",
                    save_path="../exports/mcv2_trend_plot.png"
                )
                print("MCV2 Trend Chart saved to: exports/mcv2_trend_plot.png (country code in title)")
            elif vis_choice == "2" and current_summary_df is not None:
                metric = input("Select comparison metric (e.g., MCV2 Coverage Avg (%)): ").strip() or "MCV2 Coverage Avg (%)"
                if metric not in current_summary_df.columns:
                    print(f"Error: Metric '{metric}' does not exist! Available metrics: {list(current_summary_df.columns[1:])}")
                    continue
                top_n = input("Enter number of top groups to display (default=10): ").strip() or 10
                # Plot group comparison chart (X-axis = country code)
                plot_grouped_summary(
                    current_summary_df,
                    metric=metric,
                    top_n=int(top_n),
                    save_path="../exports/mcv2_grouped_plot.png"
                )
                print("MCV2 Group Comparison Chart saved to: exports/mcv2_grouped_plot.png (country codes on X-axis)")
            else:
                print("Invalid selection or no visualization data available!")

        elif choice == "5":
            # Function 5: Export MCV2 Data (include country code)
            print("\n=== MCV2 Data Export (Include Country Code) ===")
            # Prioritize exporting filtered/summary/trend data; export full data if none exist
            if current_df is not None:
                export_df = current_df
            elif current_summary_df is not None:
                export_df = current_summary_df
            elif current_trend_df is not None:
                export_df = current_trend_df
            else:
                export_df = df
            file_name = input("Enter export file name (default=mcv2_data_with_code.csv): ").strip() or "mcv2_data_with_code.csv"
            export_path = f"../exports/{file_name}"
            if export_to_csv(export_df, export_path):
                print(f" MCV2 data exported successfully to: {export_path} (includes country code field)")
            else:
                print(" Failed to export data!")

        elif choice == "6":
            # New function: View all available country codes
            print("\n=== List of All Available Country Codes ===")
            # Pagination display (avoid overwhelming output)
            page_size = 20
            total_pages = (len(all_country_codes) + page_size - 1) // page_size
            print(f"Total country codes: {len(all_country_codes)} (split into {total_pages} pages)")
            page_num = input(f"Enter page number (1-{total_pages}; press Enter for Page 1): ").strip() or "1"
            try:
                page_num = int(page_num)
                if 1 <= page_num <= total_pages:
                    start_idx = (page_num - 1) * page_size
                    end_idx = min(start_idx + page_size, len(all_country_codes))
                    codes_page = all_country_codes[start_idx:end_idx]
                    # Display in columns for readability
                    print(f"\nPage {page_num} Country Codes ({start_idx + 1}-{end_idx}):")
                    for i in range(0, len(codes_page), 5):
                        print("  ".join(codes_page[i:i + 5]))
                else:
                    print(f"Error: Page number must be between 1 and {total_pages}!")
            except ValueError:
                print("Error: Page number must be a number!")

        elif choice == "7":
            # Function 7: Exit system
            print("\nThank you for using the MCV2 Data Dashboard (Filter by Country Code)! Goodbye!")
            logger.info("User exited MCV2 Data Dashboard (Country Code Filter Mode)")
            break

        else:
            print("Error: Invalid input! Please enter a number between 1 and 7.")


if __name__ == "__main__":
    main()