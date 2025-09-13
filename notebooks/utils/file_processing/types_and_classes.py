# ----------------------------------------------------------------------------------
# This file defines the configuration and performance tracking classes for the
# raw data processing pipeline. It ensures a consistent and well-documented
# structure for managing processing parameters.
# ----------------------------------------------------------------------------------
import time
import psutil
import copy
from typing import Set, Any
from pathlib import Path


class ProcessingConfig:
    """
    Configuration for the game processing pipeline.

    This class holds all the parameters that control the data processing workflow,
    from file paths and batch sizes to the specific filters applied to chess games.
    Using a class for configuration ensures that all parts of the pipeline access
    parameters in a consistent, predictable, and well-documented way.
    """

    def __init__(
        self,
        # --- System & File Paths ---
        parquet_path: str,
        db_path: str | Path,
        batch_size: int = 100_000,
        # --- Chess Game Filters ---
        # Players must have at least this rating to be included.
        min_player_rating: int = 1200,
        # The ELO rating difference between players cannot exceed this value.
        max_elo_difference_between_players: int = 100,
        # A set of allowed time controls (e.g., "Blitz", "Rapid").
        # Game "Event" names must contain one of these strings.
        allowed_time_controls: Set[str] | None = None,
    ):
        """
        Initializes the configuration object.

        Args:
            parquet_path: The full path to the raw parquet file to be processed.
            db_path: The path to the DuckDB database file where results will be stored.
            batch_size: The number of games to process in a single in-memory batch.
            min_player_rating: The minimum rating for both players in a game.
            max_elo_difference_between_players: The maximum allowed rating difference.
            allowed_time_controls: A set of strings for filtering game time controls.
                                   If None, defaults to {"Blitz", "Rapid", "Classical"}.
        """
        # Notes on game filters:
        # - Unrated games are not explicitly filtered, as the Lichess dataset primarily
        #   contains rated games.
        # - Bot games are filtered out within the processing logic by checking player titles.
        #   See: https://huggingface.co/datasets/Lichess/standard-chess-games

        self.parquet_path = parquet_path
        self.db_path = Path(db_path)
        self.batch_size = batch_size
        self.min_player_rating = min_player_rating
        self.max_elo_difference_between_players = max_elo_difference_between_players

        # Default to common, competitive time controls if none are specified.
        # This excludes "Bullet" and "Correspondence" (Daily), which can be less
        # representative of standard opening theory application.
        if allowed_time_controls is None:
            self.allowed_time_controls = {"Blitz", "Rapid", "Classical"}
        else:
            self.allowed_time_controls = allowed_time_controls

    def replace(self, **kwargs: Any) -> "ProcessingConfig":
        """
        Creates a new ProcessingConfig instance with updated attributes.

        This is a convenience method to produce a modified copy of the configuration,
        which is particularly useful when processing multiple files in a loop, as each
        file requires its own `parquet_path`.

        Example:
            new_config = base_config.replace(parquet_path="path/to/new_file.parquet")

        Returns:
            A new ProcessingConfig instance with the specified attributes updated.
        """
        # Create a shallow copy of the current instance
        new_config = copy.copy(self)
        # Update the attributes from the provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
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
