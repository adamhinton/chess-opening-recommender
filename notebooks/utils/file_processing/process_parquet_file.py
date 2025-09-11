import time
from typing import Dict, Optional
import duckdb
from .types_and_classes import (
    ProcessingConfig,
    PlayerStats,
    PerformanceTracker,
)

from .process_game_batch import process_batch
from .save_and_load_progress import (
    load_progress,
    save_progress,
)
from pathlib import Path


def process_parquet_file(
    config: ProcessingConfig,
    players_data: Dict[str, PlayerStats],
    log_frequency: int = 50_000,
    file_context: Optional[Dict] = None,
) -> bool:
    """
    Process a single parquet file in batches, updating a shared players_data dictionary.
    This function is a direct copy from 03_parquet_performance.ipynb.

    Args:
        config: Processing configuration for this specific file.
        players_data: The shared dictionary of player statistics to update.
        log_frequency: How often to log progress within a batch.
        file_context: Dictionary with context for multi-file processing.

    Returns:
        True if processing was successful, False otherwise.
    """
    try:
        # Initialize DuckDB connection
        con = duckdb.connect()

        # Reset progress file for a new run to avoid conflicts.
        progress_path = Path(config.save_dir) / "processing_progress_parquet.json"
        if progress_path.exists():
            progress_path.unlink()

        _, start_batch = load_progress(config)
        perf_tracker = PerformanceTracker()

        total_rows = con.execute(
            f"SELECT COUNT(*) FROM '{config.parquet_path}'"
        ).fetchone()[0]
        if total_rows == 0:
            print("File is empty, skipping.")
            return True  # Return True to allow deletion of empty file

        total_batches = (total_rows + config.batch_size - 1) // config.batch_size
        print(
            f"Will process {total_rows:,} rows in {total_batches} batches of size {config.batch_size:,}"
        )

        file_start_time = time.time()

        batch_num = start_batch
        while batch_num * config.batch_size < total_rows:
            offset = batch_num * config.batch_size

            # ETA calculations
            if batch_num > start_batch:
                batches_processed_this_file = batch_num - start_batch
                time_elapsed_this_file = time.time() - file_start_time
                avg_time_per_batch_this_file = (
                    time_elapsed_this_file / batches_processed_this_file
                )
                batches_remaining_this_file = total_batches - batch_num
                file_eta_seconds = (
                    batches_remaining_this_file * avg_time_per_batch_this_file
                )

                if file_context:
                    current_file_num = file_context.get("current_file_num", 1)
                    total_files = file_context.get("total_files", 1)

                    # Estimate total batches processed so far across all files
                    # Assuming all files have similar number of batches
                    total_batches_so_far = (
                        (current_file_num - 1) * total_batches
                    ) + batches_processed_this_file
                    total_expected_batches = total_files * total_batches

                    time_elapsed_total = time.time() - file_context.get(
                        "total_start_time", file_start_time
                    )
                    avg_time_per_batch_total = time_elapsed_total / total_batches_so_far

                    total_batches_remaining = (
                        total_expected_batches - total_batches_so_far
                    )
                    total_eta_seconds = (
                        total_batches_remaining * avg_time_per_batch_total
                    )

                    print(
                        f"\n--- File {current_file_num}/{total_files} | Batch {batch_num + 1}/{total_batches} (Offset {offset:,}) ---"
                    )
                    print(
                        f"    File ETA: {file_eta_seconds/60:.2f} mins | Total ETA: {total_eta_seconds/60:.2f} mins"
                    )
                else:
                    print(
                        f"\nProcessing batch {batch_num + 1}/{total_batches} (offset {offset:,})"
                    )
                    print(f"    File ETA: {file_eta_seconds/60:.2f} mins")

            else:
                print(
                    f"\nProcessing batch {batch_num + 1}/{total_batches} (offset {offset:,})"
                )

            perf_tracker.start_batch()

            batch_query = f"SELECT * EXCLUDE(Site, UTCDate, UTCTime, movetext) FROM '{config.parquet_path}' LIMIT {config.batch_size} OFFSET {offset}"
            batch_df = con.execute(batch_query).df()

            if batch_df.empty:
                break

            process_batch(
                batch_df,
                players_data,
                config,
                log_frequency,
                perf_tracker,
                file_context,
            )

            batch_time = perf_tracker.end_batch(len(batch_df))
            print(
                f"    Processed batch in {batch_time:.2f}s. Players: {len(players_data):,}"
            )

            batch_num += 1
            if batch_num % config.save_interval == 0:
                save_progress(players_data, batch_num, config, perf_tracker)

        save_progress(players_data, batch_num, config, perf_tracker)

        summary = perf_tracker.get_summary()
        print("\nFile Processing Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        return True
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return False
    finally:
        if "con" in locals():
            con.close()
