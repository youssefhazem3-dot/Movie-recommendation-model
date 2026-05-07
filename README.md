# Movie Recommender Website

Recommended stack:

- Backend language: Python
- Web framework: Flask
- Frontend: HTML, CSS, Jinja templates
- Local database: SQLite
- Deployment database: PostgreSQL

Python is the right backend choice here because the trained model is saved as a `.joblib` file.

## Setup

1. Put `netflix_recommendation_model.joblib` inside this folder.
2. Optional but recommended: put `netflix_review_nlp_model.joblib` inside this folder after training the ACL IMDb NLP model.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the site:

```bash
python app.py
```

5. Open:

```text
http://127.0.0.1:5000
```

For local development, the SQLite file `netflix_app.db` is created automatically.

## Database

The app uses PostgreSQL automatically when this environment variable exists:

```text
DATABASE_URL
```

On Railway, add a PostgreSQL service to the same project. Railway provides `DATABASE_URL` automatically, and the app creates the required tables on startup.

If `DATABASE_URL` is not set, the app falls back to local SQLite using `netflix_app.db`.

## App Flow

1. A user creates an account or logs in.
2. The user writes a chat message asking for a movie recommendation.
3. The app loads `netflix_recommendation_model.joblib` and recommends one movie.
4. The user rates the movie from 1 to 5 and writes how they felt while watching.
5. The rating updates future recommendations for that user.
6. The written review is saved and used to retrain the NLP sentiment model.
7. Other users can read reviews anonymously from the community reviews page.

## Model Direction

The notebook trains a hybrid feedback-aware recommender:

- SVD / Matrix Factorization as the main recommendation engine
- User feedback stored as ratings and reviews
- Feedback re-ranking on top of SVD recommendations
- ACL IMDb TF-IDF NLP model for positive/negative review sentiment

The final saved files are:

- `netflix_recommendation_model.joblib`
- `netflix_review_nlp_model.joblib`

The raw `aclImdb/` dataset folder is not committed to GitHub. Download/extract it locally before rerunning the NLP training notebook cells.

## GitHub Notes

Do commit:

- `app.py`
- `requirements.txt`
- `README.md`
- `templates/`
- `static/`

Do not commit:

- `netflix_app.db`
- `__pycache__/`
- virtual environment folders

You can commit `netflix_recommendation_model.joblib` only if it is small enough for GitHub. If it is large, upload it using Git LFS or keep it outside the repository and add it before running the app.

## How The Model Remembers Feedback

Every user review is saved in the `feedback` table:

- app user id
- Netflix user id
- movie id
- title
- rating
- review text
- sentiment label

The recommendation system reads old feedback from the database every time it recommends movies. The NLP model retrains from saved `review_text` and `sentiment_label` rows once enough feedback exists.

The `chat_messages` table stores the visible conversation for each logged-in user.

On Railway, this data is stored in PostgreSQL through `DATABASE_URL`.
