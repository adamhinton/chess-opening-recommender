import sys
from pathlib import Path

# Determine paths to ensure imports work correctly
current_file = Path.cwd()
project_root = current_file.parent  # Move up to the project root

# Add both to path to ensure imports work regardless of structure
if str(current_file) not in sys.path:
    sys.path.append(str(current_file))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from notebooks.utils.file_processing.types_and_classes import (  # noqa: E402
    PlayerStats,
    ProcessingConfig,
    PerformanceTracker,
)

from typing import Dict, Optional  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
import pickle  # noqa: E402


def save_progress(
    players_data: Dict[str, PlayerStats],
    batch_num: int,
    config: ProcessingConfig,
    perf_tracker: Optional[PerformanceTracker] = None,
) -> None:
    """
    Save current progress to disk atomically to prevent corruption.

    Args:
        players_data: Current player statistics
        batch_num: Current batch number
        config: Processing configuration
        perf_tracker: Performance tracker object
    """
    # Create save directory if it doesn't exist
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save player data
    player_data_path = save_dir / "player_stats_parquet.pkl"

    # For large datasets, pickle can be more efficient than JSON
    with open(player_data_path, "wb") as f:
        pickle.dump(players_data, f)

    # Save progress information
    progress_path = save_dir / "processing_progress_parquet.json"

    # Create a serializable version of the config (convert set to list)
    config_dict = vars(config).copy()
    if "allowed_time_controls" in config_dict and isinstance(
        config_dict["allowed_time_controls"], set
    ):
        config_dict["allowed_time_controls"] = list(
            config_dict["allowed_time_controls"]
        )

    progress_info = {
        "last_batch_processed": batch_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_players": len(players_data),
        "config": config_dict,
    }

    # Add performance metrics if available
    if perf_tracker:
        progress_info["performance"] = perf_tracker.get_summary()

    # Atomic write: write to a temporary file first, then rename
    temp_progress_path = progress_path.with_suffix(".json.tmp")
    try:
        with open(temp_progress_path, "w") as f:
            json.dump(progress_info, f, indent=2)
        temp_progress_path.rename(progress_path)
    except Exception as e:
        print(f"Error saving progress: {e}")
        if temp_progress_path.exists():
            temp_progress_path.unlink()

    print(
        f"Saved progress after batch {batch_num}. "
        + f"Current data includes {len(players_data)} players."
    )


def load_progress(config: ProcessingConfig) -> tuple[Dict[str, PlayerStats], int]:
    """
    Load previous progress from disk, handling potential file corruption.

    Args:
        config: Processing configuration

    Returns:
        Tuple of (player_data, last_batch_processed)
    """
    player_data_path = Path(config.save_dir) / "player_stats_parquet.pkl"
    progress_path = Path(config.save_dir) / "processing_progress_parquet.json"

    # Default values if no saved progress
    players_data: Dict[str, PlayerStats] = {}
    last_batch = 0

    # Load player data if it exists
    if player_data_path.exists():
        try:
            with open(player_data_path, "rb") as f:
                players_data = pickle.load(f)
            print(f"Loaded player data with {len(players_data)} players.")
        except Exception as e:
            print(f"Error loading player data: {e}")
            players_data = {}

    # Load progress info if it exists
    if progress_path.exists():
        try:
            with open(progress_path, "r") as f:
                progress_info = json.load(f)
                last_batch = progress_info.get("last_batch_processed", 0)
            print(f"Resuming from batch {last_batch}.")
        except json.JSONDecodeError:
            print(
                f"Warning: Could not decode {progress_path}. File may be corrupt. Starting from scratch."
            )
            last_batch = 0
            players_data = {}  # If progress is corrupt, start player data from scratch
        except Exception as e:
            print(f"Error loading progress info: {e}")
            last_batch = 0
            players_data = {}

    return players_data, last_batch
