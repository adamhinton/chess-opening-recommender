# _________________________________
# This util serves to process multiple raw files at once, rather than feeding them manually one by one.
# Specifically, feeding raw parquet files from a directory, each containing millions of rows of games.
# We will have many many many files to process, so this is a very helpful time saver.

# Folder selection:
# This will open a dialog for the user to select a directory.

# Dupe checks:
# Note that the processor will check parquet files for dupes. If a file has already been processed, it will be skipped.
# So you can just keep adding parquet files to the same directory and selecting that directory over and over again, and it's smart enough to skip files it has already processed.
# This is a nice way to keep all your raw data in one place, and just keep adding to it over time.
# ________________________________


import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
import tkinter as tk
from tkinter import filedialog

current_dir = Path(__file__).parent
notebooks_dir = current_dir.parent.parent
if str(notebooks_dir) not in sys.path:
    sys.path.append(str(notebooks_dir))
from utils.file_processing.raw_data_file_dupe_checks import (  # noqa: E402
    FileRegistry,
)


def select_directory() -> Optional[str]:
    """
    Show a directory picker dialog to select a directory containing parquet files.

    Returns:
        Optional[str]: The selected directory path or None if canceled
    """
    try:
        # Hide the main tkinter window
        root = tk.Tk()
        root.withdraw()

        # Show the directory selection dialog
        folder_path = filedialog.askdirectory(
            title="Select Directory with Parquet Files"
        )

        return folder_path if folder_path else None
    except Exception as e:
        print(f"Error showing directory picker: {e}")
        print("Please enter the directory path manually.")
        return input("Directory path: ")


def find_parquet_files(directory: str) -> List[str]:
    """
    Find all parquet files in the specified directory.

    Args:
        directory: Path to directory to search

    Returns:
        List of absolute paths to parquet files
    """
    if not directory:
        return []

    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"Directory {directory} does not exist or is not a directory")
        return []

    # Find all files with .parquet extension
    parquet_files = list(directory_path.glob("*.parquet"))
    return [str(p) for p in parquet_files]


def dummy_process_function(file_path: str, **kwargs) -> Dict:
    """
    Dummy function to satisfy the process_function argument.
    The actual processing is handled in the notebook.
    """
    return {"processed_file": file_path}


def process_multiple_files(
    directory: Optional[str] = None,
    batch_size: Optional[int] = 100_000,  # Default value
    min_player_rating: int = 1200,
    max_elo_difference: int = 100,
    allowed_time_controls: Optional[Set[str]] = None,
    save_dir: str = "../data/processed",
) -> Dict:
    """
    Process all parquet files in the specified directory with automatic duplicate detection.

    Args:
        directory: Path to directory containing parquet files (if None, will show picker)
        batch_size: Batch size for processing (if None, will determine optimal size)
        min_player_rating: Minimum player rating for games to be included
        max_elo_difference: Maximum rating difference between players
        allowed_time_controls: Set of allowed time controls (e.g. {"Blitz", "Rapid", "Classical"})
        save_dir: Directory to save processed data

    Returns:
        Dictionary with processing statistics
    """
    # Select directory if not provided
    if directory is None:
        directory = select_directory()
        if directory is None:
            print("No directory selected. Exiting.")
            return {"error": 1}

    # Find parquet files
    parquet_files = find_parquet_files(directory)
    if not parquet_files:
        print(f"No parquet files found in {directory}")
        return {"files_found": 0}

    print(f"Found {len(parquet_files)} parquet files in {directory}")

    # Use the utility function to filter out processed files
    registry = FileRegistry()
    all_files = [str(p) for p in parquet_files]

    new_files = []
    skipped_count = 0
    for file_path in all_files:
        if registry.is_file_processed(file_path):
            registry.mark_file_skipped(file_path)
            skipped_count += 1
        else:
            new_files.append(file_path)

    if not new_files:
        print("No new files to process.")
        return {
            "files_found": len(all_files),
            "files_skipped": skipped_count,
            "files_processed": 0,
        }

    # Set default time controls if none provided
    if allowed_time_controls is None:
        allowed_time_controls = {"Blitz", "Rapid", "Classical"}

    print(f"\nWill process {len(new_files)} files with the following parameters:")
    print(f"- Batch size: {batch_size:,}")
    print(f"- Min player rating: {min_player_rating}")
    print(f"- Max rating difference: {max_elo_difference}")
    print(f"- Allowed time controls: {', '.join(allowed_time_controls)}")
    print(f"- Save directory: {save_dir}")

    # Return processing configuration for notebook to use
    return {
        "files_to_process": new_files,
        "batch_size": batch_size,
        "min_player_rating": min_player_rating,
        "max_elo_difference": max_elo_difference,
        "allowed_time_controls": allowed_time_controls,
        "save_dir": save_dir,
        "directory": directory,
        "files_found": len(all_files),
        "files_skipped": skipped_count,
    }


if __name__ == "__main__":
    # Example usage when run directly
    config = process_multiple_files()

    if "files_to_process" in config and config["files_to_process"]:
        print("\nThe following files would be processed:")
        for file_path in config["files_to_process"]:
            print(f"  - {Path(file_path).name}")

    print(
        "\nTo process these files, import and use this module in a notebook or script."
    )
    print("Example:")
    print(
        "  from notebooks.utils.file_processing.process_multiple_raw_files import process_multiple_files"
    )
    print("  config = process_multiple_files()")
    print("  # Then use config in your notebook to process the files")
