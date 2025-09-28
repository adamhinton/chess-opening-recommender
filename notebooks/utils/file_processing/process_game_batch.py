# ----------------------------------------------------------------------------------
# This module is responsible for processing a single batch of chess games. It
# filters games based on defined criteria, extracts relevant information, and
# efficiently updates a DuckDB database with player and opening statistics.
#
#   Aggregated stats are routed to partitioned tables by ECO first letter (A-E, other)
#   - A helper function determines the correct table for each ECO code
#   - This improves write performance and future query flexibility

# Note: WHen a player already has stats for a given opening, we update the existing row rather than inserting a new one. So we add to rows such as num_losses etc. -------------------------------------------------------------

import pandas as pd
import duckdb
from typing import Optional
import sys
from pathlib import Path
import time
import traceback

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
    eligible_players: set[str],
    perf_tracker: Optional[PerformanceTracker] = None,
) -> None:
    print("[process_batch] Starting batch processing...")
    timing_details = {}
    if batch_df.empty:
        print("    Skipping empty batch.")
        return

    temp_table = "raw_games_batch"
    print(
        f"[process_batch] Registering temp table: {temp_table} with {len(batch_df)} rows"
    )
    con.register(temp_table, batch_df)

    eligible_players_df = pd.DataFrame(list(eligible_players), columns=["username"])
    print(
        f"[process_batch] Registering eligible_players_view with {len(eligible_players_df)} usernames"
    )
    con.register("eligible_players_view", eligible_players_df)

    start_time = time.time()
    time_control_pattern = "|".join(tc.lower() for tc in config.allowed_time_controls)

    # Filtering valid games
    print("[process_batch] Filtering valid games...")
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE valid_games AS
            SELECT *
            FROM {temp_table}
            WHERE
                (White IN (SELECT username FROM eligible_players_view) OR Black IN (SELECT username FROM eligible_players_view))
                AND (WhiteTitle IS NULL OR WhiteTitle NOT LIKE '%%BOT%%')
                AND (BlackTitle IS NULL OR BlackTitle NOT LIKE '%%BOT%%')
                AND WhiteElo >= {config.min_player_rating}
                AND BlackElo >= {config.min_player_rating}
                AND abs(WhiteElo - BlackElo) <= {config.max_elo_difference_between_players}
                AND regexp_matches(lower(Event), '{time_control_pattern}')
                AND Result IN ('1-0','0-1','1/2-1/2')
                AND White IS NOT NULL AND Black IS NOT NULL AND ECO IS NOT NULL;
            """
        )
    except Exception as e:
        print("[ERROR] Filtering valid games failed:", e)
        traceback.print_exc()
        raise
    timing_details["filter_valid_games"] = time.time() - start_time

    num_valid_games = con.execute("SELECT COUNT(*) FROM valid_games").fetchone()[0]
    print(f"[process_batch] Valid games after filtering: {num_valid_games}")

    if perf_tracker:
        perf_tracker.accepted_games += num_valid_games
        perf_tracker.filtered_games += len(batch_df) - num_valid_games

    if num_valid_games == 0:
        print("    No valid games in this batch after filtering.")
        con.unregister(temp_table)
        con.unregister("eligible_players_view")
        return

    # Extracting unique players & openings
    print("[process_batch] Extracting unique players and openings...")
    start_time = time.time()
    try:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE batch_players AS
                SELECT DISTINCT name, title FROM (
                    SELECT White AS name, WhiteTitle AS title FROM valid_games
                    UNION ALL
                    SELECT Black AS name, BlackTitle AS title FROM valid_games
                ) AS all_players
                WHERE name IN (SELECT username FROM eligible_players_view);

            CREATE OR REPLACE TEMP TABLE batch_openings AS
                SELECT DISTINCT ECO AS eco, Opening AS name FROM valid_games;
            """
        )
    except Exception as e:
        print("[ERROR] Extracting players/openings failed:", e)
        traceback.print_exc()
        raise
    timing_details["extract_players_openings"] = time.time() - start_time

    # Inserting new entities
    print("[process_batch] Inserting new players and openings...")
    start_time = time.time()
    try:
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
            ON CONFLICT(eco, name) DO NOTHING;
            """
        )
    except Exception as e:
        print("[ERROR] Inserting players/openings failed:", e)
        traceback.print_exc()
        raise
    timing_details["insert_entities"] = time.time() - start_time

    # Aggregating stats
    print("[process_batch] Aggregating stats...")
    start_time = time.time()
    try:
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
                LEFT JOIN player wp ON g.White = wp.name
                LEFT JOIN player bp ON g.Black = bp.name
                JOIN opening op ON g.ECO = op.eco AND g.Opening = op.name
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
            WHERE player_id IS NOT NULL
            GROUP BY player_id, opening_id, eco_code, color;
            """
        )
    except Exception as e:
        print("[ERROR] Aggregating stats failed:", e)
        traceback.print_exc()
        raise
    timing_details["aggregate_stats"] = time.time() - start_time

    # Bulk upsert into partitioned tables
    print("[process_batch] Bulk upsert into partitioned tables...")
    start_time = time.time()
    partition_timing_details = {}
    for letter in list("ABCDE") + ["other"]:
        partition_start_time = time.time()
        if letter == "other":
            where_clause = (
                "WHERE NOT (upper(left(eco_code, 1)) IN ('A','B','C','D','E'))"
            )
        else:
            where_clause = f"WHERE upper(left(eco_code, 1)) = '{letter}'"
        table = f"player_opening_stats_{letter}"
        print(f"[process_batch] Upserting partition '{letter}' into table {table}...")
        try:
            row_count = con.execute(
                f"SELECT COUNT(*) FROM aggregated_stats {where_clause}"
            ).fetchone()[0]
            print(
                f"[process_batch] Partition '{letter}' has {row_count} rows to upsert."
            )
            upsert_sql = f"""
                INSERT INTO {table} (player_id, opening_id, color, num_wins, num_draws, num_losses)
                SELECT player_id, opening_id, color, wins, draws, losses
                FROM aggregated_stats
                {where_clause}
                ON CONFLICT (player_id, opening_id, color) DO UPDATE SET
                    num_wins  = {table}.num_wins  + excluded.num_wins,
                    num_draws = {table}.num_draws + excluded.num_draws,
                    num_losses= {table}.num_losses+ excluded.num_losses;
            """
            print(
                f"[process_batch] Executing upsert SQL for partition '{letter}':\n{upsert_sql}"
            )
            con.execute(upsert_sql)
        except Exception as e:
            print(f"[ERROR] Upsert failed for partition '{letter}':", e)
            traceback.print_exc()
            print(f"[process_batch] Upsert SQL that failed:\n{upsert_sql}")
            raise
        partition_timing_details[letter] = time.time() - partition_start_time
        print(
            f"[process_batch] Finished upsert for partition '{letter}' in {partition_timing_details[letter]:.2f}s."
        )

    timing_details["bulk_upsert"] = time.time() - start_time

    print("\n--- Partition Timing Metrics ---")
    for partition, duration in partition_timing_details.items():
        print(f"Partition {partition}: {duration:.2f}s")

    num_combos = con.execute("SELECT COUNT(*) FROM aggregated_stats").fetchone()[0]
    print(f"    Processed {num_valid_games:,} games.")
    print(
        f"    Updated stats for {num_combos:,} player-opening combinations (partitioned by ECO letter)."
    )

    con.unregister(temp_table)
    con.unregister("eligible_players_view")

    print("\n--- Batch Timing Metrics ---")
    for step, duration in timing_details.items():
        print(f"{step}: {duration:.2f}s")
