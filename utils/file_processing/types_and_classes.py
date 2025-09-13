from pathlib import Path
from typing import Any, Set


class ProcessingConfig:
    """
    Configuration for the game processing pipeline.
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
        Returns a new ProcessingConfig object with updated values.

        Args:
            **kwargs: Keyword arguments corresponding to ProcessingConfig attributes.

        Returns:
            A new ProcessingConfig object with updated values.
        """
        updated_config = self.__dict__.copy()
        updated_config.update(kwargs)
        return ProcessingConfig(**updated_config)
