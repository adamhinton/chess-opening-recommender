"""Type definitions for fold-in inference pipeline.

This module defines all data structures used in the inference pipeline for
recommending openings to new players (fold-in users). It provides type safety,
autocomplete support, and clear documentation of data contracts between pipeline stages.

Key Structures:
- OpeningStatsRow: TypedDict for DataFrame row access with autocomplete
- RawOpeningStats: Individual opening statistics before transformation
- PlayerData: Complete player profile with opening history
- ProcessedOpening: Opening data after Bayesian shrinkage and normalization
- ModelInput: Final structure ready for HuggingFace model inference

Usage:
    from utils.foldin_data_processing.types import PlayerData, OpeningStatsRow

    # Type-safe DataFrame access
    row: OpeningStatsRow = df.iloc[0].to_dict()
    print(row['eco'])  # IDE autocomplete works

    # Pipeline data flow
    player_data = PlayerData(...)
    processed = transform_player_for_inference(player_data, artifacts)
    predictions = model.predict(processed.to_dict())
"""

from dataclasses import dataclass
from typing import TypedDict, List, Dict, Optional, cast
import pandas as pd
import numpy as np


# ============================================================================
# DATAFRAME TYPE DEFINITIONS
# ============================================================================


class OpeningStatsRow(TypedDict):
    """Type definition for rows in opening_stats_df DataFrame.

    Provides autocomplete and type checking when accessing DataFrame columns.
    Used in Step 1 (data extraction) before conversion to dataclasses.

    Fields correspond to database schema for player-opening statistics.
    """

    player_id: int  # Database ID of the player
    opening_id: int  # Database ID of the opening (not yet remapped to training ID)
    eco: str  # ECO code (e.g., 'C21', 'B02')
    opening_name: str  # Full name of the opening
    num_games: int  # Total games played with this opening
    num_wins: int  # Number of wins
    num_draws: int  # Number of draws
    num_losses: int  # Number of losses


def get_typed_row(df: pd.DataFrame, index: int) -> OpeningStatsRow:
    """Extract a type-safe dictionary from a DataFrame row.

    Args:
        df: DataFrame with OpeningStatsRow schema
        index: Row index to extract

    Returns:
        Typed dictionary with autocomplete support

    Example:
        row = get_typed_row(player_stats_df, 0)
        print(row['opening_name'])  # Autocomplete works in IDE
    """
    return cast(OpeningStatsRow, df.iloc[index].to_dict())


# ============================================================================
# PIPELINE DATA STRUCTURES
# ============================================================================


@dataclass
class RawOpeningStats:
    """Raw statistics for a single player-opening pair from database/API.

    This is the intermediate representation between DataFrame rows and
    processed data. Used when individual opening access is needed.

    Note: opening_id is still the database ID, not remapped to training ID yet.
    """

    opening_id: int  # Database ID (will be remapped to training ID in Step 2)
    eco: str  # ECO code
    opening_name: str  # Full opening name including ECO code (I think)
    num_games: int  # Games played
    num_wins: int  # Win count
    num_draws: int  # Draw count
    num_losses: int  # Loss count

    @property
    def raw_score(self) -> float:
        """Calculate raw performance score: (num_wins + 0.5*num_draws) / total_games.

        Returns:
            Score between 0.0 and 1.0, or 0.0 if no games played
        """
        return (
            (self.num_wins + 0.5 * self.num_draws) / self.num_games
            if self.num_games > 0
            else 0.0
        )


@dataclass
class PlayerData:
    """Complete player profile before model transformation.

    This is the main data container passed between pipeline stages.
    Keeps opening data as DataFrame for efficient vectorized operations
    while providing type-safe access to player metadata.

    The opening_stats_df should conform to OpeningStatsRow schema.
    """

    player_id: int  # Database ID (will be mapped to training ID if player exists in training set)
    name: str  # Player username
    rating: int  # Current rating for the specified color
    color: str  # 'w' for white, 'b' for black
    opening_stats_df: (
        pd.DataFrame
    )  # Opening statistics (see OpeningStatsRow for schema)

    def total_games(self) -> int:
        """Calculate total games across all openings (vectorized)."""
        return int(self.opening_stats_df["num_games"].sum())

    def total_wins(self) -> int:
        """Calculate total wins across all openings (vectorized)."""
        return int(self.opening_stats_df["num_wins"].sum())

    def mean_score(self) -> float:
        """Calculate mean performance score across all openings (vectorized)."""
        df = self.opening_stats_df
        scores = (df["num_wins"] + 0.5 * df["num_draws"]) / df["num_games"]
        return float(scores.mean())

    def filter_by_games(self, min_games: int) -> pd.DataFrame:
        """Filter openings by minimum game threshold.

        Args:
            min_games: Minimum number of games required

        Returns:
            Filtered DataFrame copy
        """
        return self.opening_stats_df[
            self.opening_stats_df["num_games"] >= min_games
        ].copy()

    def get_opening_stats(self, opening_id: int) -> RawOpeningStats:
        """Get statistics for a specific opening as dataclass.

        Args:
            opening_id: Database opening ID

        Returns:
            RawOpeningStats object for the specified opening

        Raises:
            IndexError: If opening_id not found
        """
        row = self.opening_stats_df[
            self.opening_stats_df["opening_id"] == opening_id
        ].iloc[0]
        return RawOpeningStats(
            opening_id=int(row["opening_id"]),
            eco=row["eco"],
            opening_name=row["opening_name"],
            num_games=int(row["num_games"]),
            num_wins=int(row["num_wins"]),
            num_draws=int(row["num_draws"]),
            num_losses=int(row["num_losses"]),
        )


@dataclass
class ProcessedOpening:
    """Opening data after all transformations (Step 2 output).

    This represents a single opening after:
    - ID remapping (database -> training)
    - ECO code parsing and categorization
    - Bayesian shrinkage adjustment
    - Confidence weighting calculation

    Used primarily for documentation; in practice, data stays in DataFrame
    until final conversion to ModelInput arrays.
    """

    training_opening_id: int  # Remapped ID for model input
    eco_letter_cat: int  # ECO letter category: 0=A, 1=B, 2=C, 3=D, 4=E
    eco_number_cat: int  # ECO number parsed as integer (0-99)
    adjusted_score: float  # Score after Bayesian shrinkage toward opening mean
    confidence: float  # Weight for loss function (higher = more games)
    num_games: int  # Original game count (for reference/debugging)


@dataclass
class ModelInput:
    """Final data structure ready for HuggingFace model inference.

    This is what gets sent to the model API. All player and opening data
    are converted to parallel numpy arrays for efficient batch processing.

    For fold-in (new) users, training_player_id is None and the model uses
    only rating_z and opening features to make predictions.

    Arrays have parallel indices: opening_ids[i] corresponds to eco_letter_cats[i],
    eco_number_cats[i], scores[i], and confidence[i].
    """

    # Player features
    training_player_id: Optional[int]  # None for fold-in users, int for known users
    rating_z: float  # Z-score normalized rating

    # Opening features (parallel arrays, length N = number of openings)
    opening_ids: np.ndarray  # int64, shape (N,) - training opening IDs
    eco_letter_cats: np.ndarray  # int64, shape (N,) - ECO letter categories (0-4)
    eco_number_cats: np.ndarray  # int64, shape (N,) - ECO numbers (0-99)
    scores: np.ndarray  # float32, shape (N,) - adjusted scores
    confidence: np.ndarray  # float32, shape (N,) - confidence weights

    # Metadata (not sent to model, used for post-processing)
    opening_names: List[str]  # Opening names in same order as arrays

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict for HuggingFace API.

        Returns:
            Dictionary with all fields converted to Python native types
            (numpy arrays -> lists, numpy scalars -> Python scalars)
        """
        return {
            "player_id": self.training_player_id,  # None for fold-in
            "rating_z": float(self.rating_z),
            "opening_ids": self.opening_ids.tolist(),
            "eco_letter_cats": self.eco_letter_cats.tolist(),
            "eco_number_cats": self.eco_number_cats.tolist(),
            "scores": self.scores.tolist(),
            "confidence": self.confidence.tolist(),
        }

    def __len__(self) -> int:
        """Return number of openings in this input."""
        return len(self.opening_ids)
