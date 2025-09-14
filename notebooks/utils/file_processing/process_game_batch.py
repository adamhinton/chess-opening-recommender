# ----------------------------------------------------------------------------------
# This module is responsible for processing a single batch of chess games. It
# filters games based on defined criteria, extracts relevant information, and
# efficiently updates a DuckDB database with player and opening statistics.
#
# Partitioning update (2025-09-14):
#   - Aggregated stats are now routed to partitioned tables by ECO first letter (A-E, other)
#   - A helper function determines the correct table for each ECO code
#   - This improves write performance and future query flexibility
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

def get_target_table(eco_code: str) -> str:
    """
    Returns the partitioned stats table name for a given ECO code.
    A–E go to their own table, all others to 'other'.
    """
    first_letter = eco_code[0].upper() if eco_code else ""
    if first_letter in "ABCDE":
        return f"player_opening_stats_{first_letter}"
    return "player_opening_stats_other"


def process_batch(
    batch_df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    config: ProcessingConfig,
    perf_tracker: Optional[PerformanceTracker] = None,
) -> None:
    """
    Process a batch of games and update the database with aggregated stats.

    Partitioning update:
    - After aggregation, split stats by ECO first letter and upsert into the correct table.
    - This is done in SQL for efficiency.
    """
    if batch_df.empty:
        print("    Skipping empty batch.")
        return

    temp_table = "raw_games_batch"
    con.register(temp_table, batch_df)

    time_control_pattern = "|".join(tc.lower() for tc in config.allowed_time_controls)

    # 1. Filter valid games
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE valid_games AS
        SELECT *
        FROM {temp_table}
        WHERE
            (WhiteTitle IS NULL OR WhiteTitle NOT LIKE '%%BOT%%')
            AND (BlackTitle IS NULL OR BlackTitle NOT LIKE '%%BOT%%')
            AND WhiteElo >= {config.min_player_rating}
            AND BlackElo >= {config.min_player_rating}
            AND abs(WhiteElo - BlackElo) <= {config.max_elo_difference_between_players}
            AND regexp_matches(lower(Event), '{time_control_pattern}')
            AND Result IN ('1-0','0-1','1/2-1/2')
            AND White IS NOT NULL AND Black IS NOT NULL AND ECO IS NOT NULL;
    """
    )

    num_valid_games = con.execute("SELECT COUNT(*) FROM valid_games").fetchone()[0]

    if perf_tracker:
        perf_tracker.accepted_games += num_valid_games
        perf_tracker.filtered_games += len(batch_df) - num_valid_games

    if num_valid_games == 0:
        print("    No valid games in this batch after filtering.")
        con.unregister(temp_table)
        return

    # 2. Extract unique players & openings
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE batch_players AS
            SELECT DISTINCT White AS name, WhiteTitle AS title FROM valid_games
            UNION ALL
            SELECT DISTINCT Black AS name, BlackTitle AS title FROM valid_games;

        CREATE OR REPLACE TEMP TABLE batch_openings AS
            SELECT DISTINCT ECO AS eco, Opening AS name FROM valid_games;
    """
    )

    # 3. Insert new entities
    con.execute(
        """
        INSERT INTO player (name, title)
        SELECT name, title FROM batch_players
        ON CONFLICT(name) DO NOTHING;
    """
    )
    con.execute(
        """
        INSERT INTO opening (eco, name)
        SELECT eco, name FROM batch_openings
        ON CONFLICT(eco) DO NOTHING;
    """
    )

    # 4. Aggregate stats (with ECO code for partitioning)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE aggregated_stats AS
        WITH game_results AS (
            SELECT
                wp.id AS white_id,
                bp.id AS black_id,
                op.id AS opening_id,
                op.eco AS eco_code,
                CASE
                    WHEN g.Result = '1-0' THEN 'win'
                    WHEN g.Result = '0-1' THEN 'loss'
                    ELSE 'draw'
                END AS white_result
            FROM valid_games g
            JOIN player wp ON g.White = wp.name
            JOIN player bp ON g.Black = bp.name
            JOIN opening op ON g.ECO = op.eco
        ),
        unpivoted AS (
            SELECT white_id AS player_id, opening_id, eco_code, 'w' AS color, white_result AS result
            FROM game_results
            UNION ALL
            SELECT black_id, opening_id, eco_code, 'b',
                CASE
                    WHEN white_result = 'win' THEN 'loss'
                    WHEN white_result = 'loss' THEN 'win'
                    ELSE 'draw'
                END
            FROM game_results
        )
        SELECT
            player_id,
            opening_id,
            eco_code,
            color,
            COUNT(CASE WHEN result = 'win' THEN 1 END) AS wins,
            COUNT(CASE WHEN result = 'draw' THEN 1 END) AS draws,
            COUNT(CASE WHEN result = 'loss' THEN 1 END) AS losses
        FROM unpivoted
        GROUP BY player_id, opening_id, eco_code, color;
    """
    )

    # 5. Bulk UPSERT into each partitioned table
    # For each partition, insert only the relevant rows
    for letter in list("ABCDE") + ["other"]:
        if letter == "other":
            where_clause = "WHERE NOT (upper(left(eco_code, 1)) IN ('A','B','C','D','E'))"
        else:
            where_clause = f"WHERE upper(left(eco_code, 1)) = '{letter}'"
        table = f"player_opening_stats_{letter}"
        con.execute(f"""
            INSERT INTO {table} (player_id, opening_id, color, num_wins, num_draws, num_losses)
            SELECT player_id, opening_id, color, wins, draws, losses
            FROM aggregated_stats
            {where_clause}
            ON CONFLICT (player_id, opening_id, color) DO UPDATE SET
                num_wins  = {table}.num_wins  + excluded.num_wins,
                num_draws = {table}.num_draws + excluded.num_draws,
                num_losses= {table}.num_losses+ excluded.num_losses;
        """)

    num_combos = con.execute("SELECT COUNT(*) FROM aggregated_stats").fetchone()[0]
    print(f"    Processed {num_valid_games:,} games.")
    print(f"    Updated stats for {num_combos:,} player-opening combinations (partitioned by ECO letter).")

    con.unregister(temp_table)
