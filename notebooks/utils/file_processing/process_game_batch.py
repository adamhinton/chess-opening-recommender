# This processes a batch of lichess games
# I originally thought I was using this somewhere, but now I can't find where? I think I must have used it originally then refactored to not need it
# I'm scared to delete it tbh
# TODO delete this if possible

import pandas as pd
import time
from typing import Dict, Optional
from notebooks.utils.file_processing.types_and_classes import (
    PlayerStats,
    ProcessingConfig,
    PerformanceTracker,
)


def is_valid_game(row: pd.Series, config: ProcessingConfig) -> bool:
    """
    Helper function to check if a game meets the filtering criteria. This ensures only relevant, informative games are processed.

    Args:
        row: A row from the DataFrame representing a game
        config: Processing configuration

    Returns:
        True if the game passes our filters, False otherwise
    """
    # Check player ratings
    if (
        row["WhiteElo"] < config.min_player_rating
        or row["BlackElo"] < config.min_player_rating
    ):
        return False

    # Check rating difference
    if (
        abs(row["WhiteElo"] - row["BlackElo"])
        > config.max_elo_difference_between_players
    ):
        return False

    # "Event" column on game contains time control, they're titled like "Rated Blitz Games"
    # Check that the time control is in the allowed time controls (case insensitive)
    event_lower = row["Event"].lower()
    if not any(tc.lower() in event_lower for tc in config.allowed_time_controls):
        return False

    # Check for valid result
    # If it's something weird that's not a win loss or draw, toss it out
    if row["Result"] not in {"1-0", "0-1", "1/2-1/2"}:
        return False

    return True


def process_batch(
    batch_df: pd.DataFrame,
    players_data: Dict[str, PlayerStats],
    config: ProcessingConfig,
    log_frequency: int = 5000,
    perf_tracker: Optional[PerformanceTracker] = None,
    file_context: Optional[Dict] = None,
) -> None:
    """
    Process a batch of games and update the main players_data dictionary directly.
    This function iterates through each game in a batch, filters it, and then
    updates the statistics for the white and black players in the provided
    players_data dictionary. This in-place update strategy avoids the need
    for merging separate dictionaries later, preventing data duplication bugs.

    Args:
        batch_df: DataFrame containing a batch of games.
        players_data: The main dictionary of player statistics to be updated.
        config: Processing configuration.
        log_frequency: Log progress after processing this many games.
        perf_tracker: Performance tracker object to update with filtering stats.
        file_context: Dictionary with context about the multi-file processing job.
    """
    start_time = time.time()
    total_rows = len(batch_df)

    # Tracking for filtered vs. accepted games in this batch
    batch_accepted = 0
    batch_filtered = 0

    # Process each game in the batch
    for i, (_, game) in enumerate(batch_df.iterrows()):
        # Log progress periodically within the batch
        if (i + 1) % log_frequency == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total_rows - (i + 1)) / rate if rate > 0 else 0

            # Calculate acceptance rate for this batch so far
            processed_so_far = batch_accepted + batch_filtered
            acceptance_rate = (
                (batch_accepted / processed_so_far * 100) if processed_so_far > 0 else 0
            )

            # Multi-file progress
            multi_file_str = ""
            if file_context:
                files_remaining = (
                    file_context["total_files"] - file_context["current_file_num"]
                )

                # Estimate total progress
                rows_done_in_prev_files = (
                    file_context["current_file_num"] - 1
                ) * file_context["avg_rows_per_file"]
                rows_done_in_current_file = i + 1
                total_rows_processed = (
                    rows_done_in_prev_files + rows_done_in_current_file
                )

                total_elapsed = time.time() - file_context["total_start_time"]
                overall_rate = (
                    total_rows_processed / total_elapsed if total_elapsed > 0 else 0
                )

                remaining_rows = (
                    file_context["total_rows_estimate"] - total_rows_processed
                )
                total_eta_seconds = (
                    remaining_rows / overall_rate if overall_rate > 0 else 0
                )

                multi_file_str = (
                    f"File {file_context['current_file_num']}/{file_context['total_files']} "
                    f"({files_remaining} left) - Total ETA: {total_eta_seconds / 60:.1f} min"
                )

            print(
                f"Progress: {i+1:,}/{total_rows:,} ({(i+1)/total_rows*100:.1f}%) - "
                f"Rate: {rate:.1f} games/sec - File ETA: {eta/60:.1f} min - {multi_file_str}"
            )
            print(
                f"Batch filtering: Accepted {batch_accepted:,}, Filtered {batch_filtered:,} (Acceptance rate: {acceptance_rate:.1f}%)"
            )

        # Filter out invalid games
        if not is_valid_game(game, config):
            batch_filtered += 1
            if perf_tracker:
                perf_tracker.filtered_games += 1
            continue

        # Mark as accepted
        batch_accepted += 1
        if perf_tracker:
            perf_tracker.accepted_games += 1

        # Extract relevant fields
        white_player = game["White"]
        black_player = game["Black"]

        # Handle potential missing values
        try:
            white_elo = int(game.get("WhiteElo", 0))
            black_elo = int(game.get("BlackElo", 0))
        except (ValueError, TypeError):
            white_elo = 0
            black_elo = 0

        result = game["Result"]
        eco_code = game.get("ECO", "Unknown")
        opening_name = game.get("Opening", "Unknown Opening")

        # Process white player's game
        if white_player not in players_data:
            players_data[white_player] = {
                "rating": white_elo,
                "white_games": {},
                "black_games": {},
                "num_games_total": 0,
            }

        # Update white player's data
        if eco_code not in players_data[white_player]["white_games"]:
            players_data[white_player]["white_games"][eco_code] = {
                "opening_name": opening_name,
                "results": {
                    "num_games": 0,
                    "num_wins": 0,
                    "num_losses": 0,
                    "num_draws": 0,
                    "score_percentage_with_opening": 0,
                },
            }

        # Update game counts
        players_data[white_player]["num_games_total"] += 1
        players_data[white_player]["white_games"][eco_code]["results"]["num_games"] += 1

        # Update result counts
        if result == "1-0":  # White win
            players_data[white_player]["white_games"][eco_code]["results"][
                "num_wins"
            ] += 1
        elif result == "0-1":  # Black win (white loss)
            players_data[white_player]["white_games"][eco_code]["results"][
                "num_losses"
            ] += 1
        elif result == "1/2-1/2":  # Draw
            players_data[white_player]["white_games"][eco_code]["results"][
                "num_draws"
            ] += 1

        # Update score percentage
        wins = players_data[white_player]["white_games"][eco_code]["results"][
            "num_wins"
        ]
        draws = players_data[white_player]["white_games"][eco_code]["results"][
            "num_draws"
        ]
        total = players_data[white_player]["white_games"][eco_code]["results"][
            "num_games"
        ]
        score = (wins + (draws * 0.5)) / total * 100 if total > 0 else 0
        players_data[white_player]["white_games"][eco_code]["results"][
            "score_percentage_with_opening"
        ] = round(score, 1)

        # Similarly process black player's game
        if black_player not in players_data:
            players_data[black_player] = {
                "rating": black_elo,
                "white_games": {},
                "black_games": {},
                "num_games_total": 0,
            }

        # Update black player's data
        if eco_code not in players_data[black_player]["black_games"]:
            players_data[black_player]["black_games"][eco_code] = {
                "opening_name": opening_name,
                "results": {
                    "num_games": 0,
                    "num_wins": 0,
                    "num_losses": 0,
                    "num_draws": 0,
                    "score_percentage_with_opening": 0,
                },
            }

        # Update game counts
        players_data[black_player]["num_games_total"] += 1
        players_data[black_player]["black_games"][eco_code]["results"]["num_games"] += 1

        # Update result counts
        if result == "0-1":  # Black win
            players_data[black_player]["black_games"][eco_code]["results"][
                "num_wins"
            ] += 1
        elif result == "1-0":  # White win (black loss)
            players_data[black_player]["black_games"][eco_code]["results"][
                "num_losses"
            ] += 1
        elif result == "1/2-1/2":  # Draw
            players_data[black_player]["black_games"][eco_code]["results"][
                "num_draws"
            ] += 1

        # Update score percentage
        wins = players_data[black_player]["black_games"][eco_code]["results"][
            "num_wins"
        ]
        draws = players_data[black_player]["black_games"][eco_code]["results"][
            "num_draws"
        ]
        total = players_data[black_player]["black_games"][eco_code]["results"][
            "num_games"
        ]
        score = (wins + (draws * 0.5)) / total * 100 if total > 0 else 0
        players_data[black_player]["black_games"][eco_code]["results"][
            "score_percentage_with_opening"
        ] = round(score, 1)

    # Final progress update
    elapsed = time.time() - start_time
    rate = total_rows / elapsed if elapsed > 0 else 0
    acceptance_rate = (batch_accepted / total_rows * 100) if total_rows > 0 else 0
    print(
        f"Completed {total_rows:,} games in {elapsed:.1f} seconds - Rate: {rate:.1f} games/sec"
    )
    print(
        f"Batch filtering stats: Accepted {batch_accepted:,}, Filtered {batch_filtered:,} (Acceptance rate: {acceptance_rate:.1f}%)"
    )
