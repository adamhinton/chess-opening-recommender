# ----------------------------------------------------------------------------------
# This module provides utility functions for interacting with the DuckDB database
# that stores all processed chess game statistics. It handles connection management
# and initial schema setup, ensuring that the rest of the application can reliably
# interact with a well-defined database structure.
#
# The overarching idea here is to measure players' performances with different openingss, so that the chess opening recommender can suggest the best openings for a player based on their past games.

# This will be fed to our AI model on a per-player basis, so it can learn from the player's historical data.

# Partitioning update (2025-09-14):
#   - The player_opening_stats table is now partitioned by ECO first letter (A-E, other)
#   - All schema, constraints, and foreign keys are preserved in each partitioned table
# ----------------------------------------------------------------------------------

import duckdb
from pathlib import Path
import time


def get_db_connection(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """
    Establishes and returns a connection to the DuckDB database file.

    This function is the single entry point for creating database connections,
    ensuring consistency across the application. The connect function will create
    the database file if it doesn't already exist.

    Args:
        db_path: The file system path to the DuckDB database file.

    Returns:
        An active DuckDB database connection object.
    """
    return duckdb.connect(database=str(db_path), read_only=False)


def setup_database(con: duckdb.DuckDBPyConnection):
    """
    Sets up the database schema by creating the necessary tables if they don't already exist.

    This function implements a normalized schema to efficiently store player and opening
    statistics, minimizing data redundancy and reducing storage footprint.

    - `player`: Stores a unique entry for each player, identified by an auto-incrementing
      `player_id`. This avoids repeating player names throughout the database.
    - `opening`: Stores a unique entry for each chess opening, identified by an
      auto-incrementing `opening_id`. This avoids repeating lengthy opening names.
    - `player_opening_stats`: The core table linking players and openings. It stores
      game results (wins, draws, losses) for a specific player, with a specific
      opening, playing as a specific color. This structure is highly efficient for
      both storage and querying.

    Partitioning update:
    - Creates 6 partitioned stats tables: player_opening_stats_A, ..., _E, _other
    - Creates a unifying view 'player_opening_stats' for compatibility

    Args:
        con: An active DuckDB connection.
    """
    print("Initializing database schema...")

    # Create sequences for auto-incrementing primary keys if they don't exist.
    con.execute("CREATE SEQUENCE IF NOT EXISTS player_id_seq START 1;")
    con.execute("CREATE SEQUENCE IF NOT EXISTS opening_id_seq START 1;")

    # The name is UNIQUE to ensure we don't have duplicate player entries.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS player (
            id   INTEGER PRIMARY KEY DEFAULT nextval('player_id_seq'),
            name VARCHAR UNIQUE,
            title       VARCHAR
        );
    """
    )

    # The ECO code and name combination is the unique identifier for an opening.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS opening (
            id  INTEGER PRIMARY KEY DEFAULT nextval('opening_id_seq'),
            eco         VARCHAR,
            name        VARCHAR,
            UNIQUE(eco, name)
        );
    """
    )

    # Partitioned stats tables (A-E, other)
    for letter in list("ABCDE") + ["other"]:
        table = f"player_opening_stats_{letter}"
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                player_id   INTEGER,
                opening_id  INTEGER,
                color       VARCHAR(1) NOT NULL CHECK (color IN ('w', 'b')),
                num_wins    INTEGER DEFAULT 0,
                num_draws   INTEGER DEFAULT 0,
                num_losses  INTEGER DEFAULT 0,
                PRIMARY KEY (player_id, opening_id, color),
                FOREIGN KEY (player_id) REFERENCES player(id),
                FOREIGN KEY (opening_id) REFERENCES opening(id)
            );
        """
        )

    # Unifying view for compatibility and global queries
    union_selects = "\nUNION ALL\n".join(
        [
            f"SELECT * FROM player_opening_stats_{letter}"
            for letter in list("ABCDE") + ["other"]
        ]
    )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW player_opening_stats AS
        {union_selects};
    """
    )
    print("Database tables and partitioned stats tables are ready.")


def vacuum_and_optimize(con: duckdb.DuckDBPyConnection) -> float:
    """
    Reclaims disk space and optimizes storage for all tables in the database.

    Returns:
        The elapsed time in seconds for the operation.
    """
    start = time.time()
    print("Running VACUUM on the database...")
    con.execute("VACUUM;")
    elapsed = time.time() - start
    print(f"VACUUM complete. Elapsed time: {elapsed:.2f} seconds.")
    return elapsed
