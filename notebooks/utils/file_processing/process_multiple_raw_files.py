# _________________________________
# This util serves to process multiple raw files at once, rather than feeding them manually one by one.
# Specifically, feeding raw parquet files from a directory, each containing millions of rows of games.
# We will have many many many files to process, so this is a very helpful time saver.abs

# Folder selection:
# This will open a dialog for the user to select a directory.

# Dupe checks:
# Note that the processor will check parquet files for dupes. If a file has already been processed, it will be skipped.
# So you can just keep adding parquet files to the same directory and selecting that directory over and over again, and it's smart enough to skip files it has already processed.abs
# This is a nice way to keep all your raw data in one place, and just keep adding to it over time.
# _________________________________
