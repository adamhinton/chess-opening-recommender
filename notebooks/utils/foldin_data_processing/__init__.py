"""
Fold-in data processing utilities for chess opening recommendations.

This package contains all the utilities needed to process a single player's
game data and prepare it for model inference.
"""

from .types import PlayerData, ModelInput, OpeningStatsRow
from .pipeline import PipelineConfig, PipelineArtifacts, process_player_for_inference

__all__ = [
    # Types
    "PlayerData",
    "ModelInput",
    "OpeningStatsRow",
    # Pipeline
    "PipelineConfig",
    "PipelineArtifacts",
    "process_player_for_inference",
]
