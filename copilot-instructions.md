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
