# _____________________________
# A big part of our raw data processing pipelines is downloading parquet files - 1GB files of raw games data

# We get these from the HuggingFace API

# There are a variable number of files per month, with names like train-00000-of-00065.parquet

# The best way to download files for a certain month is to first get a list of all the file names

# That's what this funciton is for.
# _____________________________


def get_parquet_file_names(month: str) -> list[str]:
    """
    Given a month in 'YYYY-MM' format, return a list of parquet file names for that month.

    Args:
        month: A string representing the month in 'YYYY-MM' format.
    Returns:
        A list of parquet file names for the specified month.
    """
