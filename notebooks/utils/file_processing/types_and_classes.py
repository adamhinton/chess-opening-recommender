# __________________
# This is types and classes for our raw-data processing pipeline.
# __________________

from typing import Dict, TypedDict, Optional, Union, Literal, Set, Any
import time
import psutil
import copy

# Define types for game results
GameResult = Literal["1-0", "0-1", "1/2-1/2", "*"]


class OpeningResults(TypedDict):
    """Statistics for a player's results with a particular opening."""

    opening_name: str
    results: Dict[str, Union[int, float]]


class PlayerStats(TypedDict):
    """Statistics for an individual player."""

    rating: int
    title: Optional[
        str
    ]  # Checking for players with BOT in their name to filter out bot games
    white_games: Dict[str, OpeningResults]  # ECO code -> results
    black_games: Dict[str, OpeningResults]  # ECO code -> results
    num_games_total: int


class ProcessingConfig:
    """Configuration for the game processing pipeline.
    Contains parameters for filtering games, batch processing, and parallelization.
    This is designed to ensure that the processing of raw chess game data yields usable results efficiently.
    """

    def __init__(
        self,
        # Computer efficiency and organization stuff
        parquet_path: str,
        batch_size: int = 100_000,
        save_interval: int = 1,
        save_dir: str = "../data/processed",
        # Chess game filtering stuff
        # Neither the black or white player can be below this rating
        min_player_rating: int = 1200,
        # Players can't be more than 100 rating points apart
        max_elo_difference_between_players: int = 100,
        # Exclude bullet and daily games by default
        allowed_time_controls: Optional[Set[str]] = None,
    ):
        # Notes on game filters:
        # Didn't exclude unrated games because our dataset contains only rated games.
        # Also didn't have to filter out bot games, because only games between two humans are rated --- I think so, at least.
        # See here to look at the data I used: https://huggingface.co/datasets/Lichess/standard-chess-games

        self.parquet_path = (
            parquet_path  # Path to the Parquet file containing raw game data
        )
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.save_dir = save_dir  # Directory to save intermediate results
        self.min_player_rating = min_player_rating
        self.max_elo_difference_between_players = max_elo_difference_between_players

        # Default to common time controls if none specified
        # Exclude bullet and daily games because they're unrepresentative
        if allowed_time_controls is None:
            self.allowed_time_controls = {"Blitz", "Rapid", "Classical"}
        else:
            self.allowed_time_controls = allowed_time_controls

    def replace(self, **kwargs: Any) -> "ProcessingConfig":
        """Creates a new ProcessingConfig instance with updated attributes.
        Useful for making carbon copies of config with slightly tweaked parameters for different files.
        """
        new_config = copy.copy(self)
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{key}'"
                )
        return new_config


class PerformanceTracker:
    """Track and report performance metrics during processing."""

    def __init__(self):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.total_games = 0
        self.batch_times = []
        self.batch_sizes = []
        self.memory_usage = []

        # Tracking for filtered vs. accepted games
        self.accepted_games = 0
        self.filtered_games = 0

    def start_batch(self):
        """Mark the start of a new batch."""
        self.batch_start_time = time.time()

    def end_batch(self, batch_size: int):
        """Mark the end of a batch and record metrics."""
        end_time = time.time()
        batch_time = end_time - self.batch_start_time

        self.total_games += batch_size
        self.batch_times.append(batch_time)
        self.batch_sizes.append(batch_size)

        # Record memory usage
        mem = psutil.virtual_memory()
        self.memory_usage.append(
            {
                "percent": mem.percent,
                "used_gb": mem.used / (1024**3),
                "available_gb": mem.available / (1024**3),
            }
        )

        return batch_time

    def log_progress(self, force: bool = False):
        """Log progress information if enough time has passed or if forced."""
        current_time = time.time()

        # Log if it's been more than 5 seconds since the last log or if forced
        if force or (current_time - self.last_log_time) >= 5:
            elapsed_total = current_time - self.start_time
            games_per_sec = self.total_games / elapsed_total if elapsed_total > 0 else 0

            # Calculate recent performance (last 5 batches or fewer)
            recent_batches = min(5, len(self.batch_times))
            if recent_batches > 0:
                recent_time = sum(self.batch_times[-recent_batches:])
                recent_games = sum(self.batch_sizes[-recent_batches:])
                recent_rate = recent_games / recent_time if recent_time > 0 else 0

                # Get the latest memory usage
                latest_mem = (
                    self.memory_usage[-1]
                    if self.memory_usage
                    else {"percent": 0, "used_gb": 0, "available_gb": 0}
                )

                # Calculate acceptance rate
                total_processed = self.accepted_games + self.filtered_games
                acceptance_rate = (
                    (self.accepted_games / total_processed * 100)
                    if total_processed > 0
                    else 0
                )

                print(
                    f"Processed {self.total_games:,} games in {elapsed_total:.2f} seconds"
                )
                print(
                    f"Accepted: {self.accepted_games:,} games, Filtered: {self.filtered_games:,} games (Acceptance rate: {acceptance_rate:.1f}%)"
                )
                print(f"Overall rate: {games_per_sec:.1f} games/sec")
                print(f"Recent rate: {recent_rate:.1f} games/sec")
                print(
                    f"Memory usage: {latest_mem['percent']}% (Used: {latest_mem['used_gb']:.1f}GB, "
                    f"Available: {latest_mem['available_gb']:.1f}GB)"
                )
                print("-" * 40)

            self.last_log_time = current_time

    def get_summary(self):
        """Get a summary of all performance metrics."""
        end_time = time.time()
        total_time = end_time - self.start_time

        avg_batch_time = (
            sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        )
        max_batch_time = max(self.batch_times) if self.batch_times else 0
        min_batch_time = min(self.batch_times) if self.batch_times else 0

        avg_batch_size = (
            sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
        )

        overall_rate = self.total_games / total_time if total_time > 0 else 0

        # Calculate filtering stats
        total_processed = self.accepted_games + self.filtered_games
        acceptance_rate = (
            (self.accepted_games / total_processed * 100) if total_processed > 0 else 0
        )

        return {
            "total_games": self.total_games,
            "total_time_sec": total_time,
            "avg_batch_time_sec": avg_batch_time,
            "min_batch_time_sec": min_batch_time,
            "max_batch_time_sec": max_batch_time,
            "avg_batch_size": avg_batch_size,
            "overall_rate_games_per_sec": overall_rate,
            "memory_usage": self.memory_usage,
            # Add filtering stats
            "accepted_games": self.accepted_games,
            "filtered_games": self.filtered_games,
            "acceptance_rate_percent": round(acceptance_rate, 1),
        }
