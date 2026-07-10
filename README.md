# Movie Recommender

A chat-based movie recommendation system built as a graduation project. The app combines collaborative filtering, movie metadata, and user feedback so it can recommend movies through a natural conversation instead of a search-by-ID flow.

## What The App Does

- Lets users sign up and log in
- Recommends movies from a chat message
- Matches genre, vibe, and age-rating preferences
- Collects a 1 to 5 star rating plus a written review
- Learns from saved feedback to improve later recommendations
- Shows anonymous community reviews
- Stores data in SQLite locally and PostgreSQL in production

If the user gives a high rating but writes a negative review, the written review is treated as the stronger signal for sentiment and future learning.

## How It Works

1. The user writes a request like a genre, vibe, or mood.
2. The app extracts movie preferences from the message.
3. The recommender ranks candidate movies using the saved model and metadata.
4. The app shows one recommendation and waits for feedback.
5. The user rates the movie and writes how they felt.
6. The feedback is stored in the database and used in later recommendations.
7. The review text is also used for NLP sentiment analysis.

## Model Components

- `netflix_recommendation_model.joblib`
  - Collaborative filtering / SVD recommendation model
- `netflix_review_nlp_model.joblib`
  - NLP sentiment model for positive, neutral, and negative reviews
- `movie_metadata.csv`
  - Movie metadata used for genre, vibe, and maturity matching
- `aclImdb/`
  - Local training dataset used for the NLP notebook

## Training Notebook

The notebook is [`model_training.ipynb`](model_training.ipynb). It includes:

- Data cleaning
- Exploratory data analysis
- Rating and popularity charts
- Genre and maturity analysis
- SVD-based recommendation training
- Review sentiment training with ACL IMDb

## Tech Stack

- Python
- Flask
- Jinja2 templates
- HTML, CSS, JavaScript
- pandas, NumPy, scikit-learn, joblib
- SQLite for local development
- PostgreSQL for deployed environments
- Railway / Vercel deployment

## Local Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure these files are in the project root:

```text
app.py
netflix_recommendation_model.joblib
netflix_review_nlp_model.joblib
movie_metadata.csv
```

3. Run the app:

```bash
python app.py
```

4. Open the site:

```text
http://127.0.0.1:5000
```

## Database

The app uses SQLite by default and creates `netflix_app.db` automatically during local development.

If `DATABASE_URL` is set, the app switches to PostgreSQL. That is the recommended setup for Railway and other persistent deployments.

Tables are created on startup, so the app can remember:

- user accounts
- chat messages
- movie recommendations
- feedback reviews and ratings

## Deployment

Start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

`main.py` is included as a compatibility entrypoint for platforms that expect `main:app`.

Vercel can run the app, but persistent feedback storage should use PostgreSQL through `DATABASE_URL`.

## Project Structure

```text
app.py
main.py
model_training.ipynb
MODEL_REPORT.md
movie_metadata.csv
netflix_recommendation_model.joblib
netflix_review_nlp_model.joblib
templates/
static/
requirements.txt
Procfile
railway.json
```

## Notes

- The raw `aclImdb/` dataset is not meant to be committed to GitHub.
- Do not commit `netflix_app.db` or virtual environment folders.
- The trained `.joblib` files are what the app loads at runtime.
