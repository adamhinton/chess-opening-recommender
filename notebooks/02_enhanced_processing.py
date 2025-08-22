"""
Enhanced Chess PGN Processing with Detailed Time Tracking

This script implements optimized processing for large PGN files with
comprehensive time tracking at every stage of the process.
"""

import chess.pgn
import zstandard as zstd
import io
import time
import json
import os
import multiprocessing
from pathlib import Path
import pickle
from dataclasses import dataclass, field
import platform
import psutil

# Import only what we need
from concurrent.futures import ProcessPoolExecutor
from typing import (
    TypedDict,
    Optional,
    Dict,
    List,
    Set,
    Iterator,
    Tuple,
    Any,
    Literal,
    Union,
)
import cProfile
import pstats
from collections import defaultdict
import gc

# ========== Performance Monitoring ==========


class TimingStats:
    """Class to track timing statistics throughout the processing pipeline."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}
        self.counters = defaultdict(int)
        self.total_start_time = time.time()

    def start(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()

    def end(self, operation: str) -> float:
        """
        End timing an operation and record the time taken.
        Returns the elapsed time.
        """
        if operation not in self.start_times:
            return 0.0

        elapsed = time.time() - self.start_times[operation]
        self.timings[operation].append(elapsed)
        del self.start_times[operation]
        return elapsed

    def increment_counter(self, counter_name: str, amount: int = 1) -> None:
        """Increment a named counter."""
        self.counters[counter_name] += amount

    def get_total_time(self, operation: str) -> float:
        """Get the total time spent on an operation."""
        return sum(self.timings[operation])

    def get_average_time(self, operation: str) -> float:
        """Get the average time per operation."""
        times = self.timings[operation]
        if not times:
            return 0.0
        return sum(times) / len(times)

    def get_count(self, operation: str) -> int:
        """Get the number of times an operation was performed."""
        return len(self.timings[operation])

    def get_rate(self, operation: str, counter: str) -> float:
        """Get the rate (counter / total_time) for an operation."""
        total_time = self.get_total_time(operation)
        count = self.counters[counter]
        if total_time == 0:
            return 0.0
        return count / total_time

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of all timing statistics."""
        summary = {}
        for op in self.timings:
            summary[op] = {
                "total_time": self.get_total_time(op),
                "average_time": self.get_average_time(op),
                "count": self.get_count(op),
            }

        # Add counter information
        summary["counters"] = dict(self.counters)

        # Add derived metrics
        summary["overall"] = {"total_time": time.time() - self.total_start_time}

        # Add rates
        for op in self.timings:
            for counter in self.counters:
                rate_key = f"{op}_{counter}_rate"
                summary["rates"] = summary.get("rates", {})
                summary["rates"][rate_key] = self.get_rate(op, counter)

        return summary

    def print_summary(self) -> None:
        """Print a formatted summary of the timing statistics."""
        summary = self.get_summary()

        print("\n===== TIMING STATISTICS =====")

        # Print overall time
        print(f"\nTotal execution time: {summary['overall']['total_time']:.2f} seconds")

        # Print counters
        print("\nCounters:")
        for counter, value in summary["counters"].items():
            print(f"  {counter}: {value:,}")

        # Print operation timings
        print("\nOperation Timings:")
        for op in sorted(self.timings.keys()):
            stats = summary[op]
            print(f"  {op}:")
            print(f"    Total: {stats['total_time']:.2f} sec")
            print(f"    Count: {stats['count']:,}")
            print(f"    Average: {stats['average_time']:.6f} sec")

        # Print rates
        if "rates" in summary:
            print("\nProcessing Rates:")
            for rate_name, rate_value in summary["rates"].items():
                readable_name = rate_name.replace("_", " ").replace(
                    "rate", "per second"
                )
                print(f"  {readable_name}: {rate_value:.2f}/sec")


# Create a global timing stats object
timing = TimingStats()

# ========== Type Definitions ==========

# Define types for game results
GameResult = Literal["1-0", "0-1", "1/2-1/2", "*"]

# Time control categories
TimeControlCategory = Literal[
    "bullet", "blitz", "rapid", "classical", "correspondence", "unknown"
]


class GameHeaders(TypedDict, total=False):
    """Type for game headers extracted from PGN."""

    Event: str
    Site: str
    Date: str
    White: str
    Black: str
    Result: GameResult
    WhiteElo: str
    BlackElo: str
    ECO: str
    Opening: str
    TimeControl: str
    Termination: str
    WhiteRatingDiff: str
    BlackRatingDiff: str


class OpeningResults(TypedDict):
    """Statistics for a player's results with a particular opening."""

    opening_name: str
    results: Dict[str, Union[int, float]]


# Player statistics structure
class PlayerStats(TypedDict):
    """Statistics for an individual player."""

    rating: int
    white_games: Dict[str, OpeningResults]  # ECO code -> results
    black_games: Dict[str, OpeningResults]  # ECO code -> results
    num_games_total: int


# Configuration parameters with defaults
@dataclass
class ProcessingConfig:
    """Configuration for the game processing pipeline."""

    # Filtering parameters
    min_elo: int = 1500  # Minimum player Elo to include
    exclude_time_controls: Set[str] = field(
        default_factory=lambda: {"bullet", "hyperbullet", "ultrabullet"}
    )
    min_moves: Optional[int] = 10  # Minimum number of moves in game (None to disable)

    # Processing parameters
    chunk_size: int = 100_000  # Number of games to process in each chunk
    max_chunks: Optional[int] = (
        None  # Maximum number of chunks to process (None for all)
    )
    save_interval: int = 1  # Save after processing this many chunks

    # File paths
    save_dir: str = "../data/processed"
    player_data_file: str = "player_stats.json"
    progress_file: str = "processing_progress.json"

    # Parallelization
    use_parallel: bool = True
    num_processes: int = max(
        1, multiprocessing.cpu_count() - 1
    )  # Use all but one CPU core

    # Advanced performance tuning
    buffer_size_mb: int = 64  # Size of zstd decompression buffer in MB
    batch_size: int = 250  # Number of games to process in each task
    process_start_method: str = (
        "fork"  # Process start method: 'fork', 'spawn', or 'forkserver'
    )
    use_process_pool: bool = (
        True  # Use ProcessPoolExecutor instead of Pool for better control
    )
    max_memory_percent: float = (
        80.0  # Maximum memory usage percentage before throttling
    )

    # Performance tracking
    enable_detailed_timing: bool = True  # Track detailed timing statistics
    profile_critical_sections: bool = False  # Use cProfile for detailed profiling

    # I/O optimization
    io_mode: str = (
        "buffered"  # "buffered" or "direct" (direct uses more memory but is faster)
    )

    def __post_init__(self):
        """Initialize and adjust config based on system capabilities."""
        # Set defaults if not provided
        if self.num_processes <= 0:
            self.num_processes = multiprocessing.cpu_count(logical=True) or 2

        # On Apple Silicon, we can use all cores efficiently
        is_apple_silicon = platform.processor() == "arm"
        if is_apple_silicon and self.num_processes < multiprocessing.cpu_count(
            logical=True
        ):
            print("Apple Silicon detected - increasing core utilization")
            self.num_processes = multiprocessing.cpu_count(logical=True)

        # Optimize batch size based on core count
        if self.batch_size == 250:  # Only if using default
            cores = psutil.cpu_count(physical=True) or 1
            self.batch_size = max(100, 1000 // cores)


# ========== Helper Functions ==========


def categorize_time_control(time_control: str) -> TimeControlCategory:
    """
    Categorize the time control string into standard categories.

    Args:
        time_control: The time control string from the game headers

    Returns:
        The category of time control
    """
    # Handle missing or malformed time control
    if not time_control or time_control == "?" or time_control == "Unknown":
        return "unknown"

    # Split time control into initial time and increment
    # Format is typically "initial+increment" like "180+2" (3 minutes + 2 second increment)
    parts = time_control.split("+")

    try:
        # Initial time in seconds
        initial_time = int(parts[0])

        # Categorize based on initial time
        if initial_time < 180:  # Less than 3 minutes
            return "bullet"
        elif initial_time < 600:  # 3-10 minutes
            return "blitz"
        elif initial_time < 1800:  # 10-30 minutes
            return "rapid"
        elif initial_time <= 6000:  # 30-100 minutes
            return "classical"
        else:  # More than 100 minutes
            return "correspondence"
    except (ValueError, IndexError):
        # Handle correspondence format like "1 day"
        if "day" in time_control.lower():
            return "correspondence"
        return "unknown"


def should_include_game(headers: GameHeaders, config: ProcessingConfig) -> bool:
    """
    Determine if a game should be included based on filtering criteria.

    Args:
        headers: The game headers
        config: Processing configuration

    Returns:
        True if the game should be included, False otherwise
    """
    timing.start("filter_game")

    # Skip games with missing essential information
    if not all(key in headers for key in ["White", "Black", "Result"]):
        timing.end("filter_game")
        return False

    # Skip games with too low Elo
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
        if white_elo < config.min_elo or black_elo < config.min_elo:
            timing.end("filter_game")
            return False
    except ValueError:
        timing.end("filter_game")
        return False

    # Skip games with excluded time controls
    if "TimeControl" in headers:
        time_control = headers.get("TimeControl", "")
        tc_category = categorize_time_control(time_control)

        if tc_category in config.exclude_time_controls:
            timing.end("filter_game")
            return False

    # Skip abandoned or incomplete games
    if headers.get("Result", "") == "*":
        timing.end("filter_game")
        return False

    # Game passes all filters
    timing.end("filter_game")
    return True


def check_min_moves(game: chess.pgn.Game, min_moves: Optional[int]) -> bool:
    """
    Check if a game has at least the minimum number of moves.

    Args:
        game: The chess game to check
        min_moves: Minimum number of moves (None to disable check)

    Returns:
        True if the game meets the minimum move requirement
    """
    if min_moves is None:
        return True

    timing.start("check_moves")

    # Count moves
    move_count = 0
    board = game.board()
    for _ in game.mainline_moves():
        move_count += 1
        if move_count >= min_moves:
            timing.end("check_moves")
            return True

    timing.end("check_moves")
    return False


# ========== Game Processing Functions ==========


def optimized_game_reader(
    file_path: str, config: ProcessingConfig
) -> Iterator[chess.pgn.Game]:
    """
    Enhanced generator that reads games more efficiently using optimized buffer sizes.

    Args:
        file_path: Path to the compressed PGN file
        config: Processing configuration

    Yields:
        Chess games that pass the filtering criteria
    """
    # Calculate buffer size in bytes
    buffer_size = config.buffer_size_mb * 1024 * 1024

    with open(file_path, "rb") as f:
        timing.start("init_decompressor")
        # Configure the decompressor with a large window size for better performance
        dctx = zstd.ZstdDecompressor()

        # Use a larger read_size for better throughput
        stream_reader = dctx.stream_reader(f, read_size=buffer_size)

        # Configure text stream
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
        timing.end("init_decompressor")

        games_processed = 0
        games_included = 0
        start_time = time.time()
        last_report_time = start_time

        while True:
            # Check if we need to throttle due to memory pressure
            if games_processed % 1000 == 0:
                mem_usage = psutil.virtual_memory().percent
                if mem_usage > config.max_memory_percent:
                    print(
                        f"Memory usage high ({mem_usage}%). Pausing briefly to let GC catch up."
                    )
                    time.sleep(2)  # Give system time to free memory

            # Read the next game with optimized parser
            timing.start("read_game")
            game = chess.pgn.read_game(text_stream)
            read_time = timing.end("read_game")

            # Check if we've reached the end of the file
            if game is None:
                break

            games_processed += 1
            timing.increment_counter("games_processed", 1)

            # Extract headers for filtering
            timing.start("extract_headers")
            headers = dict(game.headers)
            timing.end("extract_headers")

            # Check if game meets inclusion criteria
            if should_include_game(headers, config):
                # If min_moves is specified, check if the game has enough moves
                if config.min_moves is not None:
                    if not check_min_moves(game, config.min_moves):
                        continue

                games_included += 1
                timing.increment_counter("games_included", 1)
                yield game

            # Print progress periodically (every 5000 games or 30 seconds)
            current_time = time.time()
            if games_processed % 5_000 == 0 or (current_time - last_report_time) >= 30:
                elapsed = current_time - start_time
                rate = games_processed / elapsed if elapsed > 0 else 0
                inclusion_rate = (
                    (games_included / games_processed) * 100
                    if games_processed > 0
                    else 0
                )

                # Get average times
                avg_read_time = (
                    timing.get_average_time("read_game") * 1000
                )  # Convert to ms
                avg_filter_time = (
                    timing.get_average_time("filter_game") * 1000
                )  # Convert to ms
                avg_move_check_time = (
                    timing.get_average_time("check_moves") * 1000
                )  # Convert to ms

                print(
                    f"Processed: {games_processed:,} games, Included: {games_included:,} "
                    + f"({inclusion_rate:.1f}%) at {rate:.1f} games/sec"
                )
                print(
                    f"  Time per game: Read={avg_read_time:.2f}ms, Filter={avg_filter_time:.2f}ms"
                    + (
                        f", Move check={avg_move_check_time:.2f}ms"
                        if config.min_moves is not None
                        else ""
                    )
                )

                # Update last report time if we're reporting based on time
                if (current_time - last_report_time) >= 30:
                    last_report_time = current_time

                # Suggest turning off min_moves if it's taking a significant amount of time
                if config.min_moves is not None and avg_move_check_time > avg_read_time:
                    print(
                        "  OPTIMIZATION SUGGESTION: Move checking is expensive. Consider setting min_moves=None "
                        + "if filtering by move count is not critical."
                    )


def process_game_chunk(chunk: List[chess.pgn.Game]) -> Dict[str, PlayerStats]:
    """
    Process a chunk of games and return player statistics.

    Args:
        chunk: A list of chess games to process

    Returns:
        Dictionary mapping player usernames to their statistics
    """
    timing.start("process_chunk")
    players_data: Dict[str, PlayerStats] = {}

    for game in chunk:
        timing.start("process_single_game")
        headers = dict(game.headers)

        white_player = headers["White"]
        black_player = headers["Black"]
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
        result = headers["Result"]
        eco_code = headers.get("ECO", "Unknown")
        opening_name = headers.get("Opening", "Unknown Opening")

        # Process white player's game
        timing.start("update_player_data")
        if white_player not in players_data:
            players_data[white_player] = {
                "rating": white_elo,
                "white_games": {},
                "black_games": {},
                "num_games_total": 0,
            }

        # Update white player's data
        if eco_code not in players_data[white_player]["white_games"]:
            players_data[white_player]["white_games"][eco_code] = {
                "opening_name": opening_name,
                "results": {
                    "num_games": 0,
                    "num_wins": 0,
                    "num_losses": 0,
                    "num_draws": 0,
                    "score_percentage_with_opening": 0,
                },
            }

        # Update game counts
        players_data[white_player]["num_games_total"] += 1
        players_data[white_player]["white_games"][eco_code]["results"]["num_games"] += 1

        # Update result counts
        if result == "1-0":  # White win
            players_data[white_player]["white_games"][eco_code]["results"][
                "num_wins"
            ] += 1
        elif result == "0-1":  # Black win (white loss)
            players_data[white_player]["white_games"][eco_code]["results"][
                "num_losses"
            ] += 1
        elif result == "1/2-1/2":  # Draw
            players_data[white_player]["white_games"][eco_code]["results"][
                "num_draws"
            ] += 1

        # Update score percentage
        wins = players_data[white_player]["white_games"][eco_code]["results"][
            "num_wins"
        ]
        draws = players_data[white_player]["white_games"][eco_code]["results"][
            "num_draws"
        ]
        total = players_data[white_player]["white_games"][eco_code]["results"][
            "num_games"
        ]
        score = (wins + (draws * 0.5)) / total * 100 if total > 0 else 0
        players_data[white_player]["white_games"][eco_code]["results"][
            "score_percentage_with_opening"
        ] = round(score, 1)

        # Similarly process black player's game
        if black_player not in players_data:
            players_data[black_player] = {
                "rating": black_elo,
                "white_games": {},
                "black_games": {},
                "num_games_total": 0,
            }

        # Update black player's data
        if eco_code not in players_data[black_player]["black_games"]:
            players_data[black_player]["black_games"][eco_code] = {
                "opening_name": opening_name,
                "results": {
                    "num_games": 0,
                    "num_wins": 0,
                    "num_losses": 0,
                    "num_draws": 0,
                    "score_percentage_with_opening": 0,
                },
            }

        # Update game counts
        players_data[black_player]["num_games_total"] += 1
        players_data[black_player]["black_games"][eco_code]["results"]["num_games"] += 1

        # Update result counts
        if result == "0-1":  # Black win
            players_data[black_player]["black_games"][eco_code]["results"][
                "num_wins"
            ] += 1
        elif result == "1-0":  # White win (black loss)
            players_data[black_player]["black_games"][eco_code]["results"][
                "num_losses"
            ] += 1
        elif result == "1/2-1/2":  # Draw
            players_data[black_player]["black_games"][eco_code]["results"][
                "num_draws"
            ] += 1

        # Update score percentage
        wins = players_data[black_player]["black_games"][eco_code]["results"][
            "num_wins"
        ]
        draws = players_data[black_player]["black_games"][eco_code]["results"][
            "num_draws"
        ]
        total = players_data[black_player]["black_games"][eco_code]["results"][
            "num_games"
        ]
        score = (wins + (draws * 0.5)) / total * 100 if total > 0 else 0
        players_data[black_player]["black_games"][eco_code]["results"][
            "score_percentage_with_opening"
        ] = round(score, 1)

        timing.end("update_player_data")
        timing.end("process_single_game")

    timing.end("process_chunk")
    return players_data


def merge_player_stats(
    data1: Dict[str, PlayerStats], data2: Dict[str, PlayerStats]
) -> Dict[str, PlayerStats]:
    """
    Merge two player statistics dictionaries.

    Args:
        data1: First player statistics dictionary
        data2: Second player statistics dictionary

    Returns:
        Merged player statistics
    """
    timing.start("merge_stats")
    merged_data: Dict[str, PlayerStats] = data1.copy()

    for player, stats in data2.items():
        if player not in merged_data:
            merged_data[player] = stats
        else:
            # Update total game count
            merged_data[player]["num_games_total"] += stats["num_games_total"]

            # Update white games
            for eco, opening_data in stats["white_games"].items():
                if eco not in merged_data[player]["white_games"]:
                    merged_data[player]["white_games"][eco] = opening_data
                else:
                    # Update results for this opening
                    merged_data[player]["white_games"][eco]["results"][
                        "num_games"
                    ] += opening_data["results"]["num_games"]
                    merged_data[player]["white_games"][eco]["results"][
                        "num_wins"
                    ] += opening_data["results"]["num_wins"]
                    merged_data[player]["white_games"][eco]["results"][
                        "num_losses"
                    ] += opening_data["results"]["num_losses"]
                    merged_data[player]["white_games"][eco]["results"][
                        "num_draws"
                    ] += opening_data["results"]["num_draws"]

                    # Recalculate score percentage
                    wins = merged_data[player]["white_games"][eco]["results"][
                        "num_wins"
                    ]
                    draws = merged_data[player]["white_games"][eco]["results"][
                        "num_draws"
                    ]
                    total = merged_data[player]["white_games"][eco]["results"][
                        "num_games"
                    ]
                    score = (wins + (draws * 0.5)) / total * 100 if total > 0 else 0
                    merged_data[player]["white_games"][eco]["results"][
                        "score_percentage_with_opening"
                    ] = round(score, 1)

            # Update black games
            for eco, opening_data in stats["black_games"].items():
                if eco not in merged_data[player]["black_games"]:
                    merged_data[player]["black_games"][eco] = opening_data
                else:
                    # Update results for this opening
                    merged_data[player]["black_games"][eco]["results"][
                        "num_games"
                    ] += opening_data["results"]["num_games"]
                    merged_data[player]["black_games"][eco]["results"][
                        "num_wins"
                    ] += opening_data["results"]["num_wins"]
                    merged_data[player]["black_games"][eco]["results"][
                        "num_losses"
                    ] += opening_data["results"]["num_losses"]
                    merged_data[player]["black_games"][eco]["results"][
                        "num_draws"
                    ] += opening_data["results"]["num_draws"]

                    # Recalculate score percentage
                    wins = merged_data[player]["black_games"][eco]["results"][
                        "num_wins"
                    ]
                    draws = merged_data[player]["black_games"][eco]["results"][
                        "num_draws"
                    ]
                    total = merged_data[player]["black_games"][eco]["results"][
                        "num_games"
                    ]
                    score = (wins + (draws * 0.5)) / total * 100 if total > 0 else 0
                    merged_data[player]["black_games"][eco]["results"][
                        "score_percentage_with_opening"
                    ] = round(score, 1)

    timing.end("merge_stats")
    return merged_data


# ========== File Operations ==========


def save_progress(
    players_data: Dict[str, PlayerStats], chunk_num: int, config: ProcessingConfig
) -> None:
    """
    Save current progress to disk.

    Args:
        players_data: Current player statistics
        chunk_num: Current chunk number
        config: Processing configuration
    """
    timing.start("save_progress")

    # Create save directory if it doesn't exist
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save player data
    player_data_path = save_dir / config.player_data_file

    # For large datasets, pickle can be more efficient than JSON
    with open(player_data_path, "wb") as f:
        pickle.dump(players_data, f)

    # Save progress information
    progress_path = save_dir / config.progress_file
    progress_info = {
        "last_chunk_processed": chunk_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_players": len(players_data),
        "config": {
            k: v
            for k, v in config.__dict__.items()
            if not k.startswith("_") and not callable(v)
        },
        "timing_summary": timing.get_summary() if config.enable_detailed_timing else {},
    }

    with open(progress_path, "w") as f:
        json.dump(progress_info, f, indent=2)

    print(
        f"Saved progress after chunk {chunk_num}. "
        + f"Current data includes {len(players_data):,} players."
    )

    timing.end("save_progress")


def load_progress(config: ProcessingConfig) -> Tuple[Dict[str, PlayerStats], int]:
    """
    Load previous progress from disk.

    Args:
        config: Processing configuration

    Returns:
        Tuple of (player_data, last_chunk_processed)
    """
    timing.start("load_progress")

    player_data_path = Path(config.save_dir) / config.player_data_file
    progress_path = Path(config.save_dir) / config.progress_file

    # Default values if no saved progress
    players_data: Dict[str, PlayerStats] = {}
    last_chunk = 0

    # Load player data if it exists
    if player_data_path.exists():
        try:
            with open(player_data_path, "rb") as f:
                players_data = pickle.load(f)
            print(f"Loaded player data with {len(players_data):,} players.")
        except Exception as e:
            print(f"Error loading player data: {e}")
            players_data = {}

    # Load progress info if it exists
    if progress_path.exists():
        try:
            with open(progress_path, "r") as f:
                progress_info = json.load(f)
                last_chunk = progress_info.get("last_chunk_processed", 0)
            print(f"Resuming from chunk {last_chunk}.")

            # Load previous timing stats if available
            if config.enable_detailed_timing and "timing_summary" in progress_info:
                print("Previous timing statistics available in progress file.")
        except Exception as e:
            print(f"Error loading progress info: {e}")
            last_chunk = 0

    timing.end("load_progress")
    return players_data, last_chunk


# ========== Main Processing Function ==========


def process_games_with_executor(
    chunk: List[chess.pgn.Game], config: ProcessingConfig
) -> Dict[str, PlayerStats]:
    """
    Process games using a process pool executor for better resource control.

    Args:
        chunk: A list of chess games to process
        config: Processing configuration

    Returns:
        Dictionary mapping player usernames to their statistics
    """
    timing.start("parallel_processing")

    # If the chunk is small, process it directly
    if len(chunk) < 1000 or not config.use_parallel:
        result = process_game_chunk(chunk)
        timing.end("parallel_processing")
        return result

    # Divide chunk into batches
    batch_size = config.batch_size
    batches = [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]

    # Use ProcessPoolExecutor for better control
    results = []

    # Set the start method for multiprocessing
    # 'fork' is fastest on macOS/Linux but less stable, 'spawn' is slower but more stable
    ctx = multiprocessing.get_context(config.process_start_method)

    with ProcessPoolExecutor(
        max_workers=config.num_processes, mp_context=ctx
    ) as executor:
        # Submit all batches and collect futures
        futures = [executor.submit(process_game_chunk, batch) for batch in batches]

        # Process results as they complete (this helps with memory usage)
        for i, future in enumerate(futures):
            batch_result = future.result()
            results.append(batch_result)

            # Periodically report progress
            if (i + 1) % 10 == 0 or i == len(futures) - 1:
                print(f"Completed {i + 1}/{len(futures)} batches")

    # Merge all results
    merged_result = {}
    for result in results:
        merged_result = merge_player_stats(merged_result, result)

    timing.end("parallel_processing")
    return merged_result


def optimized_process_pgn_file(
    file_path: str, config: ProcessingConfig
) -> Dict[str, PlayerStats]:
    """
    Enhanced version of the PGN processing function with better resource utilization.

    Args:
        file_path: Path to the PGN file
        config: Processing configuration

    Returns:
        Player statistics dictionary
    """
    # Start overall timing
    timing.start("total_processing")

    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"PGN file not found: {file_path}")

    # Load previous progress if any
    players_data, start_chunk = load_progress(config)

    # Create game reader with optimized settings
    game_gen = optimized_game_reader(file_path, config)

    # Skip chunks that were already processed
    if start_chunk > 0:
        timing.start("skip_chunks")
        print(
            f"Resuming from chunk {start_chunk}. Skipping {start_chunk * config.chunk_size:,} games..."
        )
        games_skipped = 0
        skip_start_time = time.time()

        for _ in range(start_chunk * config.chunk_size):
            try:
                next(game_gen)
                games_skipped += 1

                # Show progress while skipping
                if games_skipped % 50_000 == 0:
                    elapsed = time.time() - skip_start_time
                    rate = games_skipped / elapsed if elapsed > 0 else 0
                    print(
                        f"Skipped {games_skipped:,} games ({games_skipped/(start_chunk * config.chunk_size)*100:.1f}%) "
                        + f"at {rate:.1f} games/sec"
                    )

            except StopIteration:
                print("Reached end of file while skipping chunks.")
                timing.end("skip_chunks")
                timing.end("total_processing")
                return players_data

        timing.end("skip_chunks")

    # Process chunks with improved tracking
    chunk_num = start_chunk
    total_start_time = time.time()

    while True:
        if config.max_chunks and chunk_num >= start_chunk + config.max_chunks:
            print(f"Reached maximum number of chunks ({config.max_chunks}). Stopping.")
            break

        # Collect a chunk of games
        timing.start("collect_chunk")
        chunk = []
        chunk_collection_start = time.time()

        for _ in range(config.chunk_size):
            try:
                game = next(game_gen)
                chunk.append(game)

                # Show progress for large chunks
                if len(chunk) % 50_000 == 0:
                    elapsed = time.time() - chunk_collection_start
                    print(
                        f"Collected {len(chunk):,}/{config.chunk_size:,} games for chunk {chunk_num + 1} "
                        + f"in {elapsed:.1f} sec"
                    )

            except StopIteration:
                print("Reached end of file.")
                break

        timing.end("collect_chunk")

        if not chunk:
            print("No more games to process.")
            break

        chunk_size = len(chunk)
        print(f"Processing chunk {chunk_num + 1} with {chunk_size:,} games...")
        chunk_start_time = time.time()

        # Process chunk with advanced parallelism
        if config.profile_critical_sections:
            # Profile the processing
            profiler = cProfile.Profile()
            profiler.enable()
            chunk_data = process_games_with_executor(chunk, config)
            profiler.disable()

            # Print profile info
            stats = pstats.Stats(profiler).sort_stats("cumtime")
            stats.print_stats(20)  # Print top 20 functions by cumulative time
        else:
            # Normal processing
            chunk_data = process_games_with_executor(chunk, config)

        # Merge with existing data
        timing.start("merge_with_existing")
        players_data = merge_player_stats(players_data, chunk_data)
        timing.end("merge_with_existing")

        # Calculate and display detailed stats
        chunk_end_time = time.time()
        chunk_time = chunk_end_time - chunk_start_time
        games_per_second = chunk_size / chunk_time if chunk_time > 0 else 0

        print(
            f"Processed chunk {chunk_num + 1} in {chunk_time:.2f} seconds "
            + f"({games_per_second:.1f} games/sec)"
        )

        # Overall progress
        total_elapsed = chunk_end_time - total_start_time
        total_chunks_done = chunk_num - start_chunk + 1
        avg_chunk_time = total_elapsed / total_chunks_done

        print(
            f"Overall progress: {total_chunks_done} chunks in {total_elapsed/60:.1f} minutes "
            + f"(avg: {avg_chunk_time:.1f} sec/chunk)"
        )

        # Save progress periodically
        chunk_num += 1
        if chunk_num % config.save_interval == 0:
            save_progress(players_data, chunk_num, config)

            # Report memory usage after saving
            mem = psutil.virtual_memory()
            print(
                f"Memory usage: {mem.percent}% (Used: {mem.used/1024**3:.1f}GB, "
                + f"Available: {mem.available/1024**3:.1f}GB)"
            )

            # Force garbage collection
            gc.collect()

    # Save final progress
    save_progress(players_data, chunk_num, config)

    # Final statistics
    total_time = time.time() - total_start_time
    print(f"\nProcessing complete in {total_time/60:.1f} minutes")
    print(f"Total players: {len(players_data):,}")

    # Print timing statistics
    if config.enable_detailed_timing:
        timing.print_summary()

    timing.end("total_processing")
    return players_data


# ========== System Info and Optimization ==========


def get_system_info() -> Dict[str, Any]:
    """
    Get detailed information about the system's hardware resources.

    Returns:
        Dictionary with system information
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

    # Get current CPU usage
    info["cpu_usage_percent"] = psutil.cpu_percent(interval=1)

    # Get available memory
    mem = psutil.virtual_memory()
    info["available_memory_gb"] = round(mem.available / (1024**3), 2)
    info["memory_usage_percent"] = mem.percent

    return info


def optimize_chunk_size(available_memory_gb: float) -> int:
    """
    Calculate an optimal chunk size based on available system memory.

    A single game object is approximately 5-10KB in memory.
    We aim to use about 25% of available memory for a chunk.

    Args:
        available_memory_gb: Available memory in GB

    Returns:
        Recommended chunk size
    """
    # Conservative estimate of memory per game (in bytes)
    memory_per_game = 10 * 1024  # 10KB

    # Use 25% of available memory for chunk
    memory_for_chunk = available_memory_gb * 0.25 * 1024**3

    # Calculate chunk size
    chunk_size = int(memory_for_chunk / memory_per_game)

    # Round to nearest 10,000
    chunk_size = max(1000, round(chunk_size / 10000) * 10000)

    return chunk_size


def optimize_parallelism() -> Dict[str, int]:
    """
    Calculate optimal parallelism settings based on the system.

    Returns:
        Dictionary with recommended settings
    """
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or 1

    # On Mac M-series chips, all cores are efficient
    # On Intel Macs, we might want to leave 1 core free for system
    is_apple_silicon = platform.processor() == "arm"

    recommendations = {
        # Use all cores for Apple Silicon, leave 1 free for Intel
        "num_processes": (
            logical_cores if is_apple_silicon else max(1, logical_cores - 1)
        ),
        # For work distribution
        "optimal_batch_size": max(1, (logical_cores * 4) // physical_cores),
        # For memory considerations
        "max_parallel_chunks": max(1, physical_cores // 2),
    }

    return recommendations


# ========== Main Execution ==========


def main():
    """Main execution function."""
    # Get system information
    system_info = get_system_info()
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    # Get optimization recommendations
    parallel_settings = optimize_parallelism()
    print("\nRecommended Parallelism Settings:")
    for key, value in parallel_settings.items():
        print(f"  {key}: {value}")

    # Recommend chunk size based on memory
    recommended_chunk_size = optimize_chunk_size(system_info["available_memory_gb"])
    print(
        f"\nRecommended chunk size based on available memory: {recommended_chunk_size:,}"
    )

    # Path to the compressed PGN file
    pgn_path = "/Users/a/Documents/personalprojects/chess-opening-recommender/data/raw/lichess_db_standard_rated_2025-07.pgn.zst"

    # Check if the file exists
    if not Path(pgn_path).exists():
        print(f"Error: PGN file not found at {pgn_path}")
        return

    # Check if we're on Apple Silicon
    is_apple_silicon = platform.processor() == "arm"

    # Create optimized configuration
    optimized_config = ProcessingConfig(
        # Filtering parameters
        min_elo=1200,  # Only include games with players above this Elo
        exclude_time_controls={"bullet", "hyperbullet", "ultrabullet"},
        min_moves=None,  # Disable move filtering for better performance
        # Processing parameters - use memory-optimized chunk size
        chunk_size=recommended_chunk_size,
        max_chunks=5,  # Process a maximum of 5 chunks - change as needed
        save_interval=1,  # Save after each chunk
        # File paths
        save_dir="../data/processed",
        # Advanced parallelization settings
        use_parallel=True,
        num_processes=parallel_settings[
            "num_processes"
        ],  # Use optimal number of processes
        # Mac-specific optimizations
        process_start_method="fork",  # Fastest on Mac, use 'spawn' if you get errors
        buffer_size_mb=128,  # Larger buffer for faster decompression
        batch_size=parallel_settings["optimal_batch_size"],  # Optimal batch size
        use_process_pool=True,  # Better control over resources
        # Performance tracking
        enable_detailed_timing=True,
        profile_critical_sections=False,  # Set to True to enable profiling
    )

    print(
        f"\nOptimized for your Mac with {system_info['physical_cores']} physical cores, "
        + f"{system_info['logical_cores']} logical cores"
    )
    print(f"Using {optimized_config.num_processes} processes for parallel execution")
    print(f"Memory-optimized chunk size: {optimized_config.chunk_size:,} games")

    # Run the optimized processing pipeline
    print("\nStarting processing...")
    players_data = optimized_process_pgn_file(pgn_path, optimized_config)

    # Show some statistics
    print(f"\nFinal Results:")
    print(f"Total number of players: {len(players_data):,}")

    # Show a sample player
    if players_data:
        import random

        sample_player = random.choice(list(players_data.keys()))
        print(f"\nSample stats for player: {sample_player}")
        print(f"Rating: {players_data[sample_player]['rating']}")
        print(f"Total games: {players_data[sample_player]['num_games_total']}")

        # Show a few white openings
        white_openings = list(players_data[sample_player]["white_games"].items())
        if white_openings:
            print("\nSample white openings:")
            for eco, data in white_openings[:3]:  # Show just 3 examples
                print(
                    f"  {eco} - {data['opening_name']}: "
                    + f"{data['results']['score_percentage_with_opening']}% score in "
                    + f"{data['results']['num_games']} games"
                )

        # Show a few black openings
        black_openings = list(players_data[sample_player]["black_games"].items())
        if black_openings:
            print("\nSample black openings:")
            for eco, data in black_openings[:3]:  # Show just 3 examples
                print(
                    f"  {eco} - {data['opening_name']}: "
                    + f"{data['results']['score_percentage_with_opening']}% score in "
                    + f"{data['results']['num_games']} games"
                )


if __name__ == "__main__":
    main()
