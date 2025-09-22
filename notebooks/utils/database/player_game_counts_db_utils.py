# ----------------------------------------------------------------------------------
# This is a database to track data for a step in our games processing pipeline.
# We want to find the most active players on Lichess to process their data for our AI model.
# We will download and examine raw games parquet datasets one by one, getting the number of games played by each username.
# This local database saves the number of games played by each user; that number of games accumulates as more data is added.
# It also tracks which files have been processed to avoid duplicates.

# Key Features:
# - Tracks the number of games played by each player (`player_game_counts` table).
# - Tracks which parquet files have been downloaded (`downloaded_files` table).
# - Ensures no duplicate downloads by checking the `downloaded_files` table.
# - Supports efficient updates and queries for millions of players and files.
#
# Purpose:
# - The `player_game_counts` table will be used to identify the 50k most active players
#   for further analysis and feeding into the chess opening recommender AI model.
# - The `downloaded_files` table ensures that we do not re-download or process the same
#   parquet files multiple times, saving time and resources.
#
# Scalability:
# - DuckDB is optimized for analytical workloads and can handle millions of rows
#   efficiently. However, as the database grows, periodic maintenance (e.g., VACUUM)
#   may be required to keep performance optimal.
#
# Usage:
# - Call `setup_player_game_counts_table` to initialize the database schema.
# - Use `record_file_download` and `is_file_already_downloaded` to manage file tracking.
# - Use `update_player_game_count` to update the game counts for players.
# ----------------------------------------------------------------------------------

from typing import Optional
import duckdb
import time


def setup_player_game_counts_table(con: duckdb.DuckDBPyConnection) -> None:
    """
    Sets up the database schema for tracking player game counts and downloaded files.

    Tables:
    - `player_game_counts`: Tracks the number of games played by each player.
    - `downloaded_files`: Tracks which parquet files have been downloaded.

    Args:
        con: An active DuckDB connection.
    """
    print("Initializing player game counts and downloaded files tables...")

    # Table for tracking player game counts
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS player_game_counts (
            username     VARCHAR PRIMARY KEY,
            num_games    INTEGER DEFAULT 0
        );
        """
    )

    # Table for tracking downloaded files
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS downloaded_files (
            file_name     VARCHAR NOT NULL,
            year          INTEGER NOT NULL,
            month         INTEGER NOT NULL,
            downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (file_name, year, month)
        );
        """
    )
    print("Tables initialized successfully.")


def record_file_download(
    con: duckdb.DuckDBPyConnection, file_name: str, year: int, month: int
) -> None:
    """
    Records a file download in the database to avoid duplicate downloads.

    Args:
        con: An active DuckDB connection.
        file_name: The name of the downloaded file.
        year: The year the file corresponds to.
        month: The month the file corresponds to.
    """
    con.execute(
        """
        INSERT INTO downloaded_files (file_name, year, month)
        VALUES (?, ?, ?)
        ON CONFLICT DO NOTHING;
        """,
        (file_name, year, month),
    )
    print(f"Recorded download: {file_name} ({year}-{month})")


def is_file_already_downloaded(
    con: duckdb.DuckDBPyConnection, file_name: str, year: int, month: int
) -> bool:
    """
    Checks if a file has already been downloaded.

    Args:
        con: An active DuckDB connection.
        file_name: The name of the file to check.
        year: The year the file corresponds to.
        month: The month the file corresponds to.

    Returns:
        True if the file has already been downloaded, False otherwise.
    """
    result = con.execute(
        """
        SELECT 1 FROM downloaded_files
        WHERE file_name = ? AND year = ? AND month = ? LIMIT 1;
        """,
        (file_name, year, month),
    ).fetchone()
    return result is not None


def update_all_player_game_counts():

def vacuum_and_optimize(con: duckdb.DuckDBPyConnection) -> float:
    """
    Reclaims disk space and optimizes storage for all tables in the database.

    Args:
        con: An active DuckDB connection.

    Returns:
        The elapsed time in seconds for the operation.
    """
    start = time.time()
    print("Running VACUUM on the database...")
    con.execute("VACUUM;")
    elapsed = time.time() - start
    print(f"VACUUM complete. Elapsed time: {elapsed:.2f} seconds.")
    return elapsed
