EXTREMELY IMPORTANT: IN OUR DATA PROCESSING PIPELINE, USE THE LIBRARIES WE HAVE AVAILABLE. SPECIFICALLY NP AND PANDAS. DON'T JUST MAKE PYTHON OBJECTS. PANDAS IS SO MUCH FASTER FOR ALMOST EVERYTHING WE NEED TO DO.

Comments, emojis and prints:
-This is a professional project.
-Be extremely limited in your use of emojis. Checkmarks are fine, clocks for time, warning sign for warnings, stuff like that.. No rockets, no smiley faces.
-Don't comment redudnant stuff. Comments that help understand or denote sections are great. But don't comment # get correlation over df.corr(), for instance
-Don't print a bunch of useless crap

8.11.2025
This is the very beginnings of my chess openings recommendation website. I will build an AI model to examine a user's openings and predictopenings that they may enjoy and score well with.

I may focus on gambits, or I may focus on all openings. I haven't decided yet. Will need to see where the data takes me; I may do both eventually but I want to focus on just one at first.

There will be two sides to this repo. The AI model side, which hosts the models and notebooks. And the web side, where I build the website that users go to.

AI model will be built in Python.

Stuff that needs to be done for AI side:

- Collect dataset. Will use lichess.org's public API to collect user info, games etc
- Build model that can analyze a user's openings and recommend new ones that they will score well with or enjoy playing.
- I will probably start with just taking users' lichess usernames because I'm more familiar with the lichess API, but I may expand to chesscom later. If chesscom's API is really easy, I'll probably start with chesscom too.

Web side:

- Next.js/TS/TailwindCSS
- Full stack Next project, next will build both frontend and backend
- Users will be able to log in, enter their lichess username, and view their recommended openings.

General concepts I want to adhere to:

- Building in public
- Regular clean, clear commits
- Storing my data in a sensible, structured way
- Writing clear, concise documentation

Notes on myself:

- I am a full stack dev by trade. I've just taken an AI/ML/DS course, so I have a basic understanding of the concepts. This will be my first solo ML project. So I want you to guide me, tell me early when I'm barking up the wrong tree, really help me out here.
- I always want to do things the right way the first time. No shortcuts, especially not ones that will cost me technical debt in the long run.
- I'm not reinventing the wheel here. I want boring, normal, standard, tested best practices. I want to use the tools that are already out there, not build my own from scratch.

Data:

- I'll download games from the lichess API.
- I'll group by username, obviously, because we're predicting what openings a user will like based on what they already play.

Game filters:
I only want games that fit the following criteria:

- Rated games
- Probably not bullet, definitely not ultra bullet
- No large rating disparities between the two opponents
- Will exclude users that don't have a certain number of games; need to figure out what exactly that number will be
- Will exclude users that don't have a certain number of games in the openings that they play; need to figure out what exactly that number will be
- Will group openings by ECO code, not by name. This is because the same opening can have different names, and I want to avoid duplicates.

Data storage:

- I will store the raw data locally in a `data/raw/` directory, organized by username
  - I'm not attached to this; can use a different storage method if it makes sense

Whatever you write, leave clear comments, xplain to me in detail, make variables for parameters that I might want to adjust rather than hard coding them. Make everything very obvious and readable as I'm new to Python, I come from web dev.

Also, make everything strongly typed where at all possible. I want to be able to use intellisense autocomplete for fields on objects, for example.

Documentation:
In general, leave helpful comments explaining what you're doing and why, especially if it won't be obvious.
At the top of a file, there should be general comments explaining the purpose of a file, when and where it's used etc. Formatted like this:

# **\*\***\*\*\*\***\*\***\_\_\_**\*\***\*\*\*\***\*\***

# This is the xyz file. It does abc. Blah blah blah

# More info blah blah blah.

# etc

# **\*\***\*\*\*\***\*\***\_\_\_**\*\***\*\*\*\***\*\***

Utils:
I like utils files. As of 8.26.25, we've been piling stuff into all the same notebooks; let's start abstracting reusable logic out to utils where it's possible and makes sense.

Notes:

- Right now (9.18.25), we are in the data collection and organization phase.

Data collection/processing:

- I ideally want to process about a billion chess games. That's making big problems because storage and processing becomes very time consuming at that scale.
- We're creating local duckdb files. We download 1GB (1.4 million games) parquet files, process them and store data at a player-level per opening.
- We're partitioning our tables by ECO code, one each for A-E and Other.
- But even with the partitions, upserts slow down dramatically; we start at about 110k games/second, and by even the fifth parquet file it's cut in half. Since we need about 1000 parquet files, that's a problem.
- So that's what we're working on right now.

Note that #09_better_downloading.ipynb is the main processing pipeline as of 9.18.25. That's the one we're most focused on. Notify me if we've moved on to another notebook and this instruction is still there.

Specs for chanigng code:
Follow my conventions. Helper functions, comments explaining the why more than the what, docstrings, readmes at top of files. Make sure this is all thoroughly documented. When making change, don't say "we made this change", just act like it's always been that way.

Best practices:

- Strongly typed where possible
- Clear markdown and comments explaining each step
- Helper functions for long or repeated segments to keep pipeline readable
- Split up in to separate cells at sensible points
