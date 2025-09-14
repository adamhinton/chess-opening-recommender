# ----------------------------------------------------------------------------------
# This module is responsible for processing a single batch of chess games. It
# filters games based on defined criteria, extracts relevant information, and
# efficiently updates a DuckDB database with player and opening statistics.
# The key design goal is to leverage DuckDB's SQL capabilities to perform these
# updates in a highly vectorized and parallelized manner, maximizing throughput.
# ----------------------------------------------------------------------------------

import pandas as pd
import duckdb
from typing import Optional
import sys
from pathlib import Path

# Ensure the project root is in the system path to allow for absolute imports.
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from notebooks.utils.file_processing.types_and_classes import (  # noqa: E402
    ProcessingConfig,
    PerformanceTracker,
)


def process_batch(
    batch_df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    config: ProcessingConfig,
    perf_tracker: Optional[PerformanceTracker] = None,
) -> None:
    """
    Processes a batch of games and updates the database with aggregated stats
    using a SQL-centric, vectorized approach.

    This function orchestrates the entire batch processing pipeline within DuckDB:
    1.  Loads the raw pandas DataFrame into a temporary table.
    2.  Filters for valid games using a single, powerful SQL query.
    3.  Inserts new players and openings into their respective tables.
    4.  Aggregates game statistics (wins, draws, losses).
    5.  Performs a bulk UPSERT into the main player_opening_stats table.

    This SQL-first methodology is substantially faster than row-by-row processing
    in Python, as it delegates the heavy lifting to DuckDB's optimized,
    columnar execution engine.

    Args:
        batch_df: A pandas DataFrame containing a batch of games.
        con: An active DuckDB connection.
        config: The processing configuration.
        perf_tracker: An optional performance tracking object.
    """
    if batch_df.empty:
        print("    Skipping empty batch.")
        return

    # Register the pandas DataFrame as a temporary virtual table in DuckDB.
    # This avoids the overhead of writing the data to disk and allows DuckDB
    # to query it directly from memory.
    temp_table_name = "raw_games_batch"
    con.register(temp_table_name, batch_df)

    # 1. Filter valid games and store them in a new temporary table.
    # This single SQL query replaces the slow, row-by-row `is_valid_game`
    # function. It applies all filtering logic in one pass, which is highly
    # efficient in a columnar database like DuckDB.
    time_control_pattern = "|".join([tc.lower() for tc in config.allowed_time_controls])

    filter_query = f"""
    CREATE OR REPLACE TEMP TABLE valid_games AS
    SELECT *
    FROM {temp_table_name}
    WHERE
        -- Filter out games involving bots by checking player titles.
        (WhiteTitle IS NULL OR WhiteTitle NOT LIKE '%%BOT%%') AND
        (BlackTitle IS NULL OR BlackTitle NOT LIKE '%%BOT%%') AND
        -- Filter by minimum player rating.
        WhiteElo >= {config.min_player_rating} AND
        BlackElo >= {config.min_player_rating} AND
        -- Filter by maximum rating difference between players.
        abs(WhiteElo - BlackElo) <= {config.max_elo_difference_between_players} AND
        -- Filter by allowed time controls using a regex match on the 'Event' field.
        regexp_matches(lower(Event), '{time_control_pattern}') AND
        -- Ensure the game has a standard result.
        Result IN ('1-0', '0-1', '1/2-1/2') AND
        -- Ensure essential fields are not null.
        White IS NOT NULL AND Black IS NOT NULL AND ECO IS NOT NULL;
    """
    con.execute(filter_query)

    # Update performance tracker with the number of games that passed the filter.
    num_valid_games = con.execute("SELECT COUNT(*) FROM valid_games").fetchone()[0]
    if perf_tracker:
        perf_tracker.accepted_games += num_valid_games
        perf_tracker.filtered_games += len(batch_df) - num_valid_games

    if num_valid_games == 0:
        print("    No valid games in this batch after filtering.")
        con.unregister(temp_table_name)
        return

    # 2. Extract unique players and openings from the filtered games.
    # These temporary tables will be used to efficiently insert new entities
    # into the main 'player' and 'opening' tables.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE batch_players AS
            SELECT DISTINCT White AS name, WhiteTitle as title FROM valid_games
            UNION ALL
            SELECT DISTINCT Black AS name, BlackTitle as title FROM valid_games;

        CREATE OR REPLACE TEMP TABLE batch_openings AS
            SELECT DISTINCT ECO as eco, Opening as name FROM valid_games;
    """
    )

    # 3. Insert new players and openings into their main tables.
    # The `ON CONFLICT DO NOTHING` clause is a highly efficient way to insert
    # only the new entities, ignoring any duplicates that already exist in the
    # target table. This avoids costly pre-checks.
    con.execute(
        "INSERT INTO player (name, title) SELECT name, title FROM batch_players ON CONFLICT(name) DO NOTHING;"
    )
    con.execute(
        "INSERT INTO opening (eco, name) SELECT eco, name FROM batch_openings ON CONFLICT(eco) DO NOTHING;"
    )

    # 4. Aggregate game statistics using pure SQL.
    # This is the core of the new logic. It performs several steps in one go:
    #   - Joins `valid_games` with the `player` and `opening` tables to get the foreign keys.
    #   - Unpivots the data to create one row per player-per-game.
    #   - Calculates the result for each player (win, loss, draw).
    #   - Groups by player, opening, and color to count the final stats.
    stats_query = """
    CREATE OR REPLACE TEMP TABLE aggregated_stats AS
    WITH game_results AS (
        -- First, join games with players and openings to resolve names to IDs.
        SELECT
            wp.id AS white_id,
            bp.id AS black_id,
            op.id AS opening_id,
            -- Determine the result from White's perspective.
            CASE
                WHEN g.Result = '1-0' THEN 'win'
                WHEN g.Result = '0-1' THEN 'loss'
                ELSE 'draw'
            END AS white_result
        FROM valid_games AS g
        JOIN player AS wp ON g.White = wp.name
        JOIN player AS bp ON g.Black = bp.name
        JOIN opening AS op ON g.ECO = op.eco
    ),
    -- Unpivot the data to create one row per player-game combination.
    -- This is necessary for aggregating stats for both White and Black players.
    unpivoted_results AS (
        SELECT white_id AS player_id, opening_id, 'w' AS color, white_result AS result FROM game_results
        UNION ALL
        SELECT black_id AS player_id, opening_id, 'b' AS color,
            -- Invert the result for the Black player.
            CASE
                WHEN white_result = 'win' THEN 'loss'
                WHEN white_result = 'loss' THEN 'win'
                ELSE 'draw'
            END AS result
        FROM game_results
    )
    -- Finally, group by player, opening, and color to aggregate the results.
    SELECT
        player_id,
        opening_id,
        color,
        COUNT(CASE WHEN result = 'win' THEN 1 END) AS wins,
        COUNT(CASE WHEN result = 'draw' THEN 1 END) AS draws,
        COUNT(CASE WHEN result = 'loss' THEN 1 END) AS losses
    FROM unpivoted_results
    GROUP BY player_id, opening_id, color;
    """
    con.execute(stats_query)

    # 5. Execute a single bulk UPSERT to update the main statistics table.
    # This is the final, critical step. `ON CONFLICT DO UPDATE` allows us to
    # either insert a new row or update an existing one in a single atomic
    # operation. We add the new results to the existing counts, ensuring that
    # statistics are cumulative.
    upsert_query = """
    INSERT INTO player_opening_stats (player_id, opening_id, color, num_wins, num_draws, num_losses)
    SELECT player_id, opening_id, color, wins, draws, losses
    FROM aggregated_stats
    ON CONFLICT (player_id, opening_id, color) DO UPDATE SET
        num_wins = player_opening_stats.num_wins + excluded.num_wins,
        num_draws = player_opening_stats.num_draws + excluded.num_draws,
        num_losses = player_opening_stats.num_losses + excluded.num_losses;
    """
    con.execute(upsert_query)

    num_combinations = con.execute("SELECT COUNT(*) FROM aggregated_stats").fetchone()[
        0
    ]
    print(f"    Processed {num_valid_games:,} games.")
    print(f"    Updated stats for {num_combinations:,} player-opening combinations.")

    # Clean up by unregistering the virtual table to free up memory.
    con.unregister(temp_table_name)
