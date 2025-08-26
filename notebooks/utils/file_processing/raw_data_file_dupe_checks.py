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
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any, Optional


class FileRegistry:
    """
    Tracks processed parquet files to avoid duplicate processing.

    This class maintains a registry of processed files using a combination of
    filename, size, modification time, and a partial file hash as a unique
    fingerprint. This approach allows reliable detection of duplicates even
    when files have the same name but different content.
    """

    def __init__(self, registry_path: str = "../data/processed/file_registry.json"):
        """
        Initialize the file registry.

        Args:
            registry_path: Path to the JSON file where the registry is stored
        """
        self.registry_path = Path(registry_path)
        self.processed_files: Dict[str, Dict[str, Any]] = {}
        self.skipped_files: List[Dict[str, Any]] = []
        self.load_registry()

    def load_registry(self) -> None:
        """Load the existing registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                    self.processed_files = data.get("processed_files", {})
                    self.skipped_files = data.get("skipped_files", [])
                    print(
                        f"Loaded registry with {len(self.processed_files)} processed files"
                    )
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading registry: {e}. Creating new registry.")
                self.processed_files = {}
                self.skipped_files = []
        else:
            print("No existing registry found. Creating new registry.")
            self.processed_files = {}
            self.skipped_files = []

    def save_registry(self) -> None:
        """Save the registry to disk."""
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

    def _get_file_fingerprint(self, file_path: str) -> str:
        """
        Generate a unique fingerprint for a file based on metadata.

        Args:
            file_path: Path to the file

        Returns:
            A string fingerprint that uniquely identifies the file

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file stats
        stats = path.stat()
        size = stats.st_size
        mtime = stats.st_mtime

        # Calculate hash of first 8KB for extra uniqueness
        # (helpful for identical size/timestamp files)
        with open(path, "rb") as f:
            file_header = f.read(8 * 1024)  # Read first 8KB
            header_hash = hashlib.md5(file_header).hexdigest()

        # Combine into a unique fingerprint
        fingerprint = f"{path.name}_{size}_{mtime}_{header_hash[:8]}"
        return fingerprint

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get complete metadata for a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing file metadata
        """
        path = Path(file_path)
        stats = path.stat()

        with open(path, "rb") as f:
            file_header = f.read(8 * 1024)
            header_hash = hashlib.md5(file_header).hexdigest()

        return {
            "filename": path.name,
            "full_path": str(path.absolute()),
            "size_bytes": stats.st_size,
            "modified_time": stats.st_mtime,
            "modified_time_readable": time.ctime(stats.st_mtime),
            "header_hash": header_hash[:8],
            "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def is_file_processed(self, file_path: str) -> bool:
        """
        Check if a file has been fully processed.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file has been processed, False otherwise
        """
        try:
            fingerprint = self._get_file_fingerprint(file_path)
            return fingerprint in self.processed_files
        except FileNotFoundError:
            return False

    def mark_file_processed(self, file_path: str) -> None:
        """
        Mark a file as fully processed.

        Args:
            file_path: Path to the file to mark as processed
        """
        fingerprint = self._get_file_fingerprint(file_path)
        metadata = self.get_file_metadata(file_path)
        self.processed_files[fingerprint] = metadata
        self.save_registry()

    def mark_file_skipped(
        self, file_path: str, reason: str = "Already processed"
    ) -> None:
        """
        Record a file that was skipped during processing.

        Args:
            file_path: Path to the file that was skipped
            reason: Reason for skipping the file
        """
        try:
            metadata = self.get_file_metadata(file_path)
            metadata["skip_reason"] = reason
            metadata["skip_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.skipped_files.append(metadata)
            self.save_registry()
        except FileNotFoundError:
            print(
                f"Warning: Attempted to mark non-existent file as skipped: {file_path}"
            )

    def list_processed_files(self) -> List[Dict[str, Any]]:
        """
        Return a list of all processed files with metadata.

        Returns:
            List of dictionaries containing file metadata
        """
        return list(self.processed_files.values())

    def list_skipped_files(self) -> List[Dict[str, Any]]:
        """
        Return a list of all skipped files with metadata.

        Returns:
            List of dictionaries containing file metadata
        """
        return self.skipped_files


def process_multiple_files(
    file_paths: List[str],
    process_function,
    registry_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Process multiple files while avoiding duplicates.

    Args:
        file_paths: List of paths to files to process
        process_function: Function to process each file
                        Should accept a file path as its first argument
        registry_path: Path to the file registry JSON file
        **kwargs: Additional arguments to pass to the process_function

    Returns:
        Dictionary containing results from the process_function
    """
    # Initialize file registry
    registry = FileRegistry(registry_path) if registry_path else FileRegistry()

    # Filter out already processed files
    new_files = []
    for file_path in file_paths:
        if registry.is_file_processed(file_path):
            print(f"Skipping already processed file: {Path(file_path).name}")
            registry.mark_file_skipped(file_path)
            continue
        new_files.append(file_path)

    if not new_files:
        print("No new files to process.")
        return {}

    print(
        f"Found {len(new_files)} new files to process out of {len(file_paths)} total files."
    )

    # Process each new file
    results = {}
    for file_path in new_files:
        print(f"\nProcessing file: {Path(file_path).name}")
        try:
            # Process the file using the provided function
            file_results = process_function(file_path, **kwargs)

            # If processing was successful, mark file as processed
            registry.mark_file_processed(file_path)

            # Merge results
            if isinstance(results, dict) and isinstance(file_results, dict):
                # Assuming results is a dict - modify this if your results have a different structure
                results.update(file_results)
            elif not results:
                results = file_results

        except Exception as e:
            print(f"Error processing file {Path(file_path).name}: {e}")

    return results
