# ----------------------------------------------------------------------------------
# This module provides utility functions for interacting with the DuckDB database
# that stores all processed chess game statistics. It handles connection management
# and initial schema setup, ensuring that the rest of the application can reliably
# interact with a well-defined database structure.

# We will be downloading large parquet files of raw games data, processing them, and saving the results to this locally-stored database.
# ----------------------------------------------------------------------------------

import duckdb
from pathlib import Path


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

    Args:
        con: An active DuckDB connection.
    """
    print("Initializing database schema...")
    # Use SERIAL as a convenient alias for BIGINT AUTO_INCREMENT.
    # The player_name is UNIQUE to ensure we don't have duplicate player entries.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS player (
            player_id   BIGINT IDENTITY PRIMARY KEY,
            player_name VARCHAR UNIQUE NOT NULL,
            title       VARCHAR
        );
    """
    )

    # The ECO code is the unique identifier for an opening.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS opening (
            opening_id  BIGINT IDENTITY PRIMARY KEY,
            eco         VARCHAR UNIQUE NOT NULL,
            name        VARCHAR NOT NULL
        );
    """
    )

    # This is the main statistics table.
    # The PRIMARY KEY is a composite of player, opening, and color, ensuring one
    # unique record for each combination.
    # The CHECK constraint on 'color' ensures data integrity.
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS player_opening_stats (
            player_id   BIGINT,
            opening_id  BIGINT,
            color       VARCHAR(1) NOT NULL CHECK (color IN ('w', 'b')),
            num_wins    INTEGER DEFAULT 0,
            num_draws   INTEGER DEFAULT 0,
            num_losses  INTEGER DEFAULT 0,
            PRIMARY KEY (player_id, opening_id, color),
            FOREIGN KEY (player_id) REFERENCES player(player_id),
            FOREIGN KEY (opening_id) REFERENCES opening(opening_id)
        );
    """
    )
    print("Database tables are ready.")
