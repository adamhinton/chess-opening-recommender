# __________________________________
# This is utils for downloading raw parquet data files from Hugging Face.
# Each file is about 1GB in size.
# Note that dupe checks will (should) have been done before calling this download function; so we can (should) assume that any file here has not already been downloaded and processed.
# __________________________________

from pathlib import Path
import shutil
from typing import Optional

from huggingface_hub import hf_hub_download


def download_single_parquet_file(
    repo_id: str,
    repo_type: str,
    file_to_download: str,
    local_dir: Path,
    year: int,
    month: int,
) -> Optional[Path]:
    """
    Downloads a single parquet file from a Hugging Face repository,
    renames it, and saves it to a specified local directory.
    Note that the downloaded file will be processed, and then immediately deleted.

    Args:
        repo_id: The ID of the repository on Hugging Face (e.g., "Lichess/standard-chess-games").
        repo_type: The type of the repository (e.g., "dataset").
        file_to_download: The path-like filename within the repository to download.
        local_dir: The local directory (as a Path object) to save the file in.
        year: The year associated with the file, for naming purposes.
        month: The month associated with the file, for naming purposes.

    Returns:
        The local path (Path object) to the downloaded file if successful, otherwise None.
    """
    try:
        # Download (HF handles caching)
        downloaded_path_str = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=file_to_download,
        )
        downloaded_path = Path(downloaded_path_str)

        # Create a descriptive local filename
        target_filename = f"{year}-{month:02d}-{downloaded_path.name}"
        target_path = local_dir / target_filename

        # Ensure the local directory exists
        local_dir.mkdir(parents=True, exist_ok=True)

        # Copy the file to our desired flat directory structure
        shutil.copy(downloaded_path, target_path)
        print(f"File saved to {target_path}")
        return target_path
    except Exception as e:
        print(f"Failed to download or save {file_to_download}. Error: {e}")
        return None
