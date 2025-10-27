# Chess Opening Recommender

Welcome to my chess opening recommender!

This will be an AI model that analyzes a chess player's online games history.

It will recommend opening sequences that the player is likely to enjoy and score well with.

## Data

### Data Collection

- First, I determined the 50,000 most active Lichess players from 2023 through August 2025 that met the following criteria:

  - Their account was still open
  - Their account hadn't been banned for cheating or other violations
  - I only used blitz, rapid and classical games to determine this level of activity.

- Then, I collected three years of Lichess.org games data from HuggingFace's API.
  - https://huggingface.co/datasets/Lichess/standard-chess-games/viewer/default/
  - I downloaded parquet files of game data from the past three years.
  - Then, one by one, I saved relevant games to a local database.

### Valid games

- I only collected games that met the following criteria:
  - One of the players was in my list of the 50,000 most active players
  - Rated games
  - Time control of Blitz, Rapid or Classical
  - Played between January 2023 and August 2025, inclusive
  - The two players were within 100 rating points of each other
  - The opening had a name and ECO code.

## Plans:

### Data Sanitization

- Remove openings where one side gets obliterated
  - Otherwise the model will erroneously predict a high (or low) win rate for everyone
