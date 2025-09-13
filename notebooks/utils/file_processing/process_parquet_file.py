# ----------------------------------------------------------------------------------
# This module orchestrates the processing of a single, large parquet file
# containing raw chess game data. It reads the file in manageable batches,
# passes each batch to the processing function, and tracks overall progress.
# The main goal is to process files that are too large to fit into memory
# in a robust and efficient manner.
# ----------------------------------------------------------------------------------

import duckdb
import time
from typing import Dict, Optional
from pathlib import Path
import sys

# Ensure the project root is in the system path to allow for absolute imports.
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from notebooks.utils.file_processing.types_and_classes import (  # noqa: E402
    ProcessingConfig,
    PerformanceTracker,
)
from notebooks.utils.file_processing.process_game_batch import (
    process_batch,
)  # noqa: E402
from notebooks.utils.database.db_utils import get_db_connection  # noqa: E402


def process_parquet_file(
    config: ProcessingConfig,
    file_context: Optional[Dict] = None,
) -> bool:
    """
    Process a single parquet file in batches, saving the results to a DuckDB database.

    This function reads a large parquet file chunk by chunk, handing off each
    chunk to `process_batch` for filtering, analysis, and database insertion.
    It manages the lifecycle of the database connection for the file and reports
    detailed performance metrics upon completion.

    Args:
        config: The configuration object for this specific file processing job.
        file_context: An optional dictionary containing context for multi-file
                      processing runs, used for logging and ETA calculations.

    Returns:
        True if the file was processed successfully, False otherwise.
    """
    db_con = None
    try:
        # Establish a single database connection for the entire file.
        db_con = get_db_connection(config.db_path)
        perf_tracker = PerformanceTracker()

        # Use a temporary DuckDB connection to get metadata about the file.
        # This avoids loading the whole file into memory.
        with duckdb.connect() as temp_con:
            total_rows = temp_con.execute(
                f"SELECT COUNT(*) FROM '{config.parquet_path}'"
            ).fetchone()[0]

        if total_rows == 0:
            print("File is empty, skipping.")
            return True  # Successfully "processed" an empty file.

        total_batches = (total_rows + config.batch_size - 1) // config.batch_size
        print(
            f"Processing {total_rows:,} rows in {total_batches} batches of size {config.batch_size:,}"
        )

        file_start_time = time.time()

        for batch_num in range(total_batches):
            offset = batch_num * config.batch_size

            # Simple progress display for the current file
            print(f"\n--- Starting Batch {batch_num + 1}/{total_batches} ---")

            perf_tracker.start_batch()

            # Read a batch from the parquet file.
            # Excluding 'movetext' saves significant memory and processing time.
            batch_query = f"SELECT * EXCLUDE movetext FROM '{config.parquet_path}' LIMIT {config.batch_size} OFFSET {offset}"

            # We use a new connection for the read to avoid potential conflicts
            # with the main DB connection, though this is largely a precaution.
            with duckdb.connect() as read_con:
                batch_df = read_con.execute(batch_query).df()

            if batch_df.empty:
                print("    Skipping empty batch.")
                continue

            # Hand off the batch to the core processing function.
            process_batch(
                batch_df=batch_df,
                con=db_con,
                config=config,
                perf_tracker=perf_tracker,
            )

            batch_time = perf_tracker.end_batch(len(batch_df))
            print(f"    Batch processed in {batch_time:.2f}s.")

            # Log file-level ETA
            batches_processed = batch_num + 1
            time_elapsed = time.time() - file_start_time
            avg_time_per_batch = time_elapsed / batches_processed
            batches_remaining = total_batches - batches_processed
            eta_seconds = batches_remaining * avg_time_per_batch
            print(
                f"    File ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}"
            )

        summary = perf_tracker.get_summary()
        print("\nFile Processing Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        return True
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        # Potentially log the error to a file or more robust logging system here.
        return False
    finally:
        # Ensure the database connection is always closed.
        if db_con:
            db_con.close()
            print("Database connection closed.")
