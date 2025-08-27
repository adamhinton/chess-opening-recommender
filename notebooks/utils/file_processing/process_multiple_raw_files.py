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
# _________________________________

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
import tkinter as tk
from tkinter import filedialog

# Add the parent directory to the path to allow importing from other modules
current_dir = Path(__file__).parent
notebooks_dir = current_dir.parent.parent
if str(notebooks_dir) not in sys.path:
    sys.path.append(str(notebooks_dir))

# Try to import the FileRegistry
try:
    from notebooks.utils.file_processing.raw_data_file_dupe_checks import FileRegistry

    print("Successfully imported FileRegistry")
except ImportError:
    print("Could not import FileRegistry - creating a simplified version")

    # Implement a simplified version of FileRegistry if the real one can't be imported
    class FileRegistry:
        """Simplified version of FileRegistry to track processed files."""

        def __init__(self):
            self.registry_path = (
                Path(notebooks_dir) / "data/processed/file_registry.json"
            )
            self.processed_files = set()
            self._load_registry()

        def _load_registry(self):
            """Load the registry from disk if it exists."""
            import json

            if self.registry_path.exists():
                try:
                    with open(self.registry_path, "r") as f:
                        registry_data = json.load(f)
                        self.processed_files = set(
                            registry_data.get("processed_files", [])
                        )
                except Exception as e:
                    print(f"Warning: Could not load registry file: {e}")
                    self.processed_files = set()

        def _save_registry(self):
            """Save the registry to disk."""
            import json

            try:
                os.makedirs(self.registry_path.parent, exist_ok=True)
                with open(self.registry_path, "w") as f:
                    json.dump({"processed_files": list(self.processed_files)}, f)
            except Exception as e:
                print(f"Warning: Could not save registry file: {e}")

        def is_file_processed(self, file_path: str) -> bool:
            """Check if a file has been processed."""
            return str(file_path) in self.processed_files

        def mark_file_processed(self, file_path: str) -> None:
            """Mark a file as processed."""
            self.processed_files.add(str(file_path))
            self._save_registry()

        def mark_file_skipped(self, file_path: str) -> None:
            """Mark a file as skipped."""
            # For our purposes, skipped is the same as processed
            self.processed_files.add(str(file_path))
            self._save_registry()


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


def process_directory(directory: Optional[str] = None) -> Dict[str, int]:
    """
    Process all parquet files in the specified directory.

    Args:
        directory: Path to directory containing parquet files (if None, will show picker)

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

    # Initialize file registry for duplicate detection
    try:
        registry = FileRegistry()
    except Exception as e:
        print(f"Error initializing file registry: {e}")
        print("Continuing without duplicate detection")
        registry = None

    # Filter out already processed files
    new_files = []
    skipped_files = []

    for file_path in parquet_files:
        if registry and registry.is_file_processed(file_path):
            print(f"Skipping already processed file: {Path(file_path).name}")
            skipped_files.append(file_path)
            try:
                registry.mark_file_skipped(file_path)
            except Exception as e:
                print(f"Warning: Could not mark file as skipped: {e}")
        else:
            new_files.append(file_path)

    if not new_files:
        print("No new files to process.")
        return {
            "files_found": len(parquet_files),
            "files_skipped": len(skipped_files),
            "files_processed": 0,
        }

    print(
        f"Will process {len(new_files)} new files out of {len(parquet_files)} total files."
    )

    # Return the list of new files to process - the actual processing will be done in the notebook
    return {
        "directory": directory,
        "files_found": len(parquet_files),
        "files_skipped": len(skipped_files),
        "files_to_process": new_files,
    }


def get_optimal_batch_size() -> int:
    """
    Calculate the optimal batch size based on available system memory.

    Returns:
        Recommended batch size (number of rows)
    """
    try:
        import psutil

        # Get available memory in GB
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Use 30% of available memory, assuming 1KB per row
        memory_for_batch_gb = available_memory_gb * 0.3
        optimal_batch_size = int(memory_for_batch_gb * 1024**3 / 1024)  # 1KB per row

        # Round to nearest 10,000
        optimal_batch_size = max(10_000, round(optimal_batch_size / 10_000) * 10_000)

        return optimal_batch_size
    except ImportError:
        # Default batch size if psutil is not available
        return 100_000


def process_multiple_files(
    directory: Optional[str] = None,
    batch_size: Optional[int] = None,
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
    # Get directory and find files
    result = process_directory(directory)

    if (
        "error" in result
        or "files_to_process" not in result
        or not result["files_to_process"]
    ):
        print("No files to process. Exiting.")
        return result

    files_to_process = result["files_to_process"]

    # Determine batch size if not provided
    if batch_size is None:
        batch_size = get_optimal_batch_size()
        print(f"Using automatically determined batch size: {batch_size:,}")

    # Set default time controls if none provided
    if allowed_time_controls is None:
        allowed_time_controls = {"Blitz", "Rapid", "Classical"}

    print(
        f"\nWill process {len(files_to_process)} files with the following parameters:"
    )
    print(f"- Batch size: {batch_size:,}")
    print(f"- Min player rating: {min_player_rating}")
    print(f"- Max rating difference: {max_elo_difference}")
    print(f"- Allowed time controls: {', '.join(allowed_time_controls)}")
    print(f"- Save directory: {save_dir}")

    # Return processing configuration for notebook to use
    return {
        "files_to_process": files_to_process,
        "batch_size": batch_size,
        "min_player_rating": min_player_rating,
        "max_elo_difference": max_elo_difference,
        "allowed_time_controls": allowed_time_controls,
        "save_dir": save_dir,
        "directory": result.get("directory", ""),
        "files_found": result.get("files_found", 0),
        "files_skipped": result.get("files_skipped", 0),
    }


if __name__ == "__main__":
    # Example usage when run directly
    result = process_directory()
    print(f"Processing result: {result}")

    if "files_to_process" in result and result["files_to_process"]:
        print("The following files would be processed:")
        for file_path in result["files_to_process"]:
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
