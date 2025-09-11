"""
********************************************************************************
* This module handles file tracking to avoid duplicate processing of parquet
* files in the chess opening recommender system.
*
* It provides a FileRegistry class that tracks processed files using metadata
* fingerprints to uniquely identify files regardless of naming collisions.
* The registry stores information about each file including its size,
* modification time, and a partial hash of the file header.
*
* For more context: I upload raw .parquet files with over a million chess games for processing. This makes sure we don't process the same parquet file twice.
********************************************************************************
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any


class FileRegistry:
    """
    Tracks which HuggingFace (HF) raw games dataset files have already been processed.
    Uses HF-provided metadata (month, filename, size, etag) to fingerprint files.
    The order of operations is:
    1. Call the HF API to list available files for a given month (or whatever time period)
    2. For each file, BEFORE downloading it, check if it has already been processed using is_file_processed
    3. Download and process only new files
    This is because the files are 1GB each, so it's very wasteful to download files we already processed.
    """

    def __init__(self, registry_path: str = "../data/processed/file_registry.json"):
        self.registry_path = Path(registry_path)
        self.processed_files: Dict[str, Dict[str, Any]] = {}
        self.skipped_files: Dict[str, Dict[str, Any]] = {}
        self.load_registry()

    # ---------------- Core persistence ----------------
    def load_registry(self) -> None:
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                self.processed_files = data.get("processed_files", {})
                self.skipped_files = data.get("skipped_files", {})
        else:
            self.processed_files = {}
            self.skipped_files = {}

    def save_registry(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(
                {
                    "processed_files": self.processed_files,
                    "skipped_files": self.skipped_files,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=2,
            )

    # ---------------- Dupe checks ----------------
    def is_file_processed(
        self, month: str, filename: str, size: int, etag: str
    ) -> bool:
        """Return True if this file (for given month) is already processed."""
        return (
            month in self.processed_files
            and filename in self.processed_files[month]
            and self.processed_files[month][filename].get("etag") == etag
        )

    # ---------------- Markers ----------------
    def mark_file_processed(
        self, month: str, filename: str, size: int, etag: str
    ) -> None:
        self.processed_files.setdefault(month, {})[filename] = {
            "size_bytes": size,
            "etag": etag,
            "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save_registry()

    def mark_file_skipped(self, month: str, filename: str, reason: str) -> None:
        self.skipped_files.setdefault(month, {})[filename] = {
            "skip_reason": reason,
            "skip_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save_registry()

    # ---------------- Introspection ----------------
    def list_processed_files(self, month: str) -> Dict[str, Any]:
        return (
            self.processed_files
            if month is None
            else self.processed_files.get(month, {})
        )

    def list_skipped_files(self, month: str) -> Dict[str, Any]:
        return (
            self.skipped_files if month is None else self.skipped_files.get(month, {})
        )


def process_multiple_files(
    file_paths: List[str],
    process_function,
    **kwargs,
):
    """
    Process multiple local files (already filtered for dupes before this step).

    Args:
        file_paths: List of paths to files to process
        process_function: Function to process each file
                          Should accept a file path as its first argument
        **kwargs: Additional arguments to pass to the process_function

    Returns:
        Dictionary containing results from the process_function
    """
    if not file_paths:
        print("No files to process.")
        return {}

    results = {}
    for file_path in file_paths:
        print(f"\nProcessing file: {Path(file_path).name}")
        try:
            file_results = process_function(file_path, **kwargs)

            if isinstance(results, dict) and isinstance(file_results, dict):
                results.update(file_results)
            elif not results:
                results = file_results
        except Exception as e:
            print(f"Error processing file {Path(file_path).name}: {e}")

    return results
