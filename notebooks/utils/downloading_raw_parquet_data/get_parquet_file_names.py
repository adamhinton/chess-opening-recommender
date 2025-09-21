# _____________________________
# A big part of our raw data processing pipelines is downloading parquet files - 1GB files of raw games data

# We get these from the HuggingFace API

# There are a variable number of files per month, with names like train-00000-of-00065.parquet

# The best way to download files for a certain month is to first get a list of all the file names

# That's what this funciton is for.
# _____________________________

from huggingface_hub import HfApi


def get_parquet_file_names(year: int, month: int) -> list[str]:
    """
    Given a year and month, return a list of parquet file names for that month.

    Args:
        year: An integer representing the year (e.g., 2025).
        month: An integer representing the month (1-12).
    Returns:
        A list of parquet file names for the specified year and month.
    """
    # Initialize the HuggingFace API
    api = HfApi()

    # File names in the remote repo are structured like:
    # data/year=2025/month=03/train-00001-of-00065.parquet
    # Construct the target prefix for filtering files
    target_prefix = f"data/year={year}/month={month:02d}/"

    # Fetch all file names from the repository
    files = api.list_repo_files(
        repo_id="Lichess/standard-chess-games", repo_type="dataset"
    )

    # Filter files that match the target year and month
    all_file_names_in_month = [f for f in files if f.startswith(target_prefix)]

    return all_file_names_in_month
