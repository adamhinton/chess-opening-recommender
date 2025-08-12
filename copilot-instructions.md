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
