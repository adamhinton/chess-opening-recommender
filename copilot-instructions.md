8.11.2025
This is the very beginnings of my chess openings recommendation website. I will build an AI model to examine a user's openings and recommend new openings that they may enjoy and score well with.

There will be two sides to this repo. The AI model side, which hosts the models and notebooks. And the web side, where I build the website that users go to.

AI model will be built in Python.

Stuff that needs to be done for AI side:

- Collect dataset. Will use lichess.org's public API to collect user info, games etc
- Build model that can analyze a user's openings and recommend new ones that they will score well with or enjoy playing.

Web side:

- Next.js/TS/TailwindCSS
- Full stack Next project, next will build both frontend and backend
- Users will be able to log in, enter their lichess username, and view their recommended openings.
