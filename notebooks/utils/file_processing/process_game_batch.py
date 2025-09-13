# ----------------------------------------------------------------------------------
# This module is responsible for processing a single batch of chess games. It
# filters games based on defined criteria, extracts relevant information, and
# efficiently updates a DuckDB database with player and opening statistics.
# The key design goal is to perform these updates in a batched manner to
# minimize database transactions and maximize throughput, which is critical
# when processing billions of games.
# ----------------------------------------------------------------------------------

import pandas as pd
import duckdb
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple, List
import sys
from pathlib import Path

# Ensure the project root is in the system path to allow for absolute imports.
# This is crucial for making the script runnable from different locations.
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from notebooks.utils.file_processing.types_and_classes import (  # noqa: E402
    ProcessingConfig,
    PerformanceTracker,
)


def is_valid_game(row: pd.Series, config: ProcessingConfig) -> bool:
    """
    Checks if a single game meets the predefined filtering criteria.

    This function is a gatekeeper, ensuring that only high-quality, relevant
    games are included in the statistical analysis. It filters out games with
    bots, low-rated players, large rating disparities, and undesirable time
    controls.

    Args:
        row: A pandas Series representing a single game from the dataset.
        config: The global processing configuration object.

    Returns:
        True if the game is valid, False otherwise.
    """
    # Filter out games involving bots by checking player titles.
    if (
        "BOT" in str(row.get("WhiteTitle", "")).upper()
        or "BOT" in str(row.get("BlackTitle", "")).upper()
    ):
        return False

    # Filter out games where either player is below the minimum rating.
    if (
        row["WhiteElo"] < config.min_player_rating
        or row["BlackElo"] < config.min_player_rating
    ):
        return False

    # Filter out games with a large rating difference between players.
    if (
        abs(row["WhiteElo"] - row["BlackElo"])
        > config.max_elo_difference_between_players
    ):
        return False

    # Filter based on allowed time controls. The "Event" field often contains
    # this information (e.g., "Rated Blitz game").
    event_lower = row["Event"].lower()
    if not any(tc.lower() in event_lower for tc in config.allowed_time_controls):
        return False

    # Filter out games with non-standard results (e.g., abandoned).
    if row["Result"] not in {"1-0", "0-1", "1/2-1/2"}:
        return False

    return True


def _get_or_create_ids(
    con: duckdb.DuckDBPyConnection,
    entity_type: str,
    names: Set[str],
    cache: Dict[str, int],
    extra_data: Optional[Dict[str, Tuple]] = None,
) -> None:
    """
    A generic helper to fetch existing IDs or create new entries in the database.

    This function optimizes database interactions by first attempting to fetch all
    required IDs (for players or openings) in a single query. For entities that
    don't exist, it performs a single bulk insert. This is far more efficient
    than querying or inserting one by one.

    Args:
        con: Active DuckDB connection.
        entity_type: The type of entity ('player' or 'opening').
        names: A set of unique names (player names or ECO codes) to look up.
        cache: A dictionary used as an in-memory cache for mapping names to IDs.
        extra_data: Optional dictionary for providing additional data for new
                    entries (e.g., 'name' for openings, 'title' for players).
    """
    if not names:
        return

    # Fetch existing IDs from the database and update the cache.
    name_col = "player_name" if entity_type == "player" else "eco"
    id_col = "player_id" if entity_type == "player" else "opening_id"
    placeholders = ", ".join(["?"] * len(names))
    existing_df = con.execute(
        f"SELECT {name_col}, {id_col} FROM {entity_type} WHERE {name_col} IN ({placeholders})",
        list(names),
    ).df()

    for _, row in existing_df.iterrows():
        cache[row[name_col]] = row[id_col]

    # Identify names that were not found in the database.
    new_names = names - set(cache.keys())
    if not new_names:
        return

    # Insert new entries in a single bulk operation.
    new_entries = []
    if entity_type == "player":
        new_entries = [(name, extra_data.get(name, (None,))) for name in new_names]
        con.executemany(
            f"INSERT INTO {entity_type} (player_name, title) VALUES (?, ?)", new_entries
        )
    elif entity_type == "opening":
        new_entries = [
            (eco, extra_data.get(eco, ("Unknown Opening",))) for eco in new_names
        ]
        con.executemany(
            f"INSERT INTO {entity_type} (eco, name) VALUES (?, ?)", new_entries
        )

    # Retrieve the newly created IDs and update the cache.
    new_df = con.execute(
        f"SELECT {name_col}, {id_col} FROM {entity_type} WHERE {name_col} IN ({placeholders})",
        list(new_names),
    ).df()
    for _, row in new_df.iterrows():
        cache[row[name_col]] = row[id_col]


def process_batch(
    batch_df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    config: ProcessingConfig,
    perf_tracker: Optional[PerformanceTracker] = None,
) -> None:
    """
    Processes a batch of games and updates the database with aggregated stats.

    This is the core function for data processing. It orchestrates the filtering
    of games, the lookup/creation of player and opening IDs, the aggregation of
    game results, and the final bulk update to the database.

    Args:
        batch_df: A pandas DataFrame containing a batch of games.
        con: An active DuckDB connection.
        config: The processing configuration.
        perf_tracker: An optional performance tracking object.
    """
    # 1. Filter valid games
    valid_games_mask = batch_df.apply(is_valid_game, axis=1, config=config)
    valid_games_df = batch_df[valid_games_mask].copy()

    if perf_tracker:
        perf_tracker.accepted_games += len(valid_games_df)
        perf_tracker.filtered_games += len(batch_df) - len(valid_games_df)

    if valid_games_df.empty:
        print("    No valid games in this batch.")
        return

    # 2. Collect unique players and openings from the batch
    player_names = set(valid_games_df["White"]) | set(valid_games_df["Black"])
    openings_eco = set(valid_games_df["ECO"])
    player_titles = {
        row["White"]: row.get("WhiteTitle") for _, row in valid_games_df.iterrows()
    }
    player_titles.update(
        {row["Black"]: row.get("BlackTitle") for _, row in valid_games_df.iterrows()}
    )
    opening_names = {
        row["ECO"]: row.get("Opening", "Unknown Opening")
        for _, row in valid_games_df.iterrows()
    }

    # 3. Get or create IDs for all players and openings
    player_id_cache: Dict[str, int] = {}
    opening_id_cache: Dict[str, int] = {}
    _get_or_create_ids(con, "player", player_names, player_id_cache, player_titles)
    _get_or_create_ids(con, "opening", openings_eco, opening_id_cache, opening_names)

    # 4. Aggregate game results in memory
    stats_updates = defaultdict(lambda: defaultdict(int))
    for _, game in valid_games_df.iterrows():
        white_id = player_id_cache.get(game["White"])
        black_id = player_id_cache.get(game["Black"])
        opening_id = opening_id_cache.get(game["ECO"])

        if not all([white_id, black_id, opening_id]):
            continue  # Should not happen if caches are populated correctly

        if game["Result"] == "1-0":  # White wins
            stats_updates[(white_id, opening_id, "w")]["wins"] += 1
            stats_updates[(black_id, opening_id, "b")]["losses"] += 1
        elif game["Result"] == "0-1":  # Black wins
            stats_updates[(white_id, opening_id, "w")]["losses"] += 1
            stats_updates[(black_id, opening_id, "b")]["wins"] += 1
        elif game["Result"] == "1/2-1/2":  # Draw
            stats_updates[(white_id, opening_id, "w")]["draws"] += 1
            stats_updates[(black_id, opening_id, "b")]["draws"] += 1

    # 5. Prepare data for bulk UPSERT
    update_data: List[Tuple[int, int, str, int, int, int]] = []
    for (player_id, opening_id, color), results in stats_updates.items():
        update_data.append(
            (
                player_id,
                opening_id,
                color,
                results["wins"],
                results["draws"],
                results["losses"],
            )
        )

    if not update_data:
        return

    # 6. Execute bulk UPSERT operation
    # This is the most critical performance step. The ON CONFLICT clause tells
    # DuckDB to update the existing record if a primary key collision occurs,
    # otherwise it inserts a new record. This single command handles all our
    # statistical updates for the batch.
    con.executemany(
        """
        INSERT INTO player_opening_stats (player_id, opening_id, color, num_wins, num_draws, num_losses)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (player_id, opening_id, color) DO UPDATE SET
            num_wins = num_wins + excluded.num_wins,
            num_draws = num_draws + excluded.num_draws,
            num_losses = num_losses + excluded.num_losses;
        """,
        update_data,
    )
    print(f"    Updated stats for {len(update_data)} player-opening combinations.")
