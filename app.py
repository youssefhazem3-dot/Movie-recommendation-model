from datetime import datetime
from functools import wraps
from pathlib import Path
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parent
IS_VERCEL = bool(os.environ.get("VERCEL"))
DB_PATH = Path(os.environ.get(
    "SQLITE_DB_PATH",
    "/tmp/netflix_app.db" if IS_VERCEL else str(BASE_DIR / "netflix_app.db"),
))
DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL)
RECOMMENDER_PATH = BASE_DIR / "netflix_recommendation_model.joblib"
NLP_MODEL_PATH = BASE_DIR / "netflix_review_nlp_model.joblib"
MOVIE_METADATA_PATH = BASE_DIR / "Movies 67.csv"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")

if USE_POSTGRES:
    import psycopg
    from psycopg.rows import dict_row


def get_db():
    if USE_POSTGRES:
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def prepare_sql(query):
    if USE_POSTGRES:
        return query.replace("?", "%s")
    return query


def db_execute(conn, query, params=()):
    return conn.execute(prepare_sql(query), params)


def init_db():
    sqlite_schema = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        app_user_id INTEGER NOT NULL,
        netflix_user_id INTEGER NOT NULL,
        movie_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        rating INTEGER NOT NULL,
        review_text TEXT NOT NULL,
        sentiment_label TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (app_user_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        app_user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        movie_id INTEGER,
        title TEXT,
        predicted_rating REAL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (app_user_id) REFERENCES users(id)
    );
    """

    postgres_schema = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY,
        app_user_id INTEGER NOT NULL REFERENCES users(id),
        netflix_user_id INTEGER NOT NULL,
        movie_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        rating INTEGER NOT NULL,
        review_text TEXT NOT NULL,
        sentiment_label TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS chat_messages (
        id SERIAL PRIMARY KEY,
        app_user_id INTEGER NOT NULL REFERENCES users(id),
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        movie_id INTEGER,
        title TEXT,
        predicted_rating DOUBLE PRECISION,
        created_at TEXT NOT NULL
    );
    """

    schema = postgres_schema if USE_POSTGRES else sqlite_schema

    conn = get_db()
    for statement in schema.strip().split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)
    conn.commit()
    conn.close()


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("login"))
        return func(*args, **kwargs)

    return wrapper


def load_recommender_model():
    if not RECOMMENDER_PATH.exists():
        return None
    model_objects = joblib.load(RECOMMENDER_PATH)
    model_objects["movie_titles"] = pd.DataFrame(model_objects["movie_titles"])
    return model_objects


recommender_model = load_recommender_model()


def load_movie_metadata():
    if not MOVIE_METADATA_PATH.exists():
        return pd.DataFrame(columns=["movie_id", "title", "genres", "age_rating"])

    metadata_df = pd.read_csv(MOVIE_METADATA_PATH)
    if "age_rating" not in metadata_df.columns:
        metadata_df["age_rating"] = "Teen"

    metadata_df = metadata_df[["movie_id", "title", "genres", "age_rating"]].copy()
    metadata_df["movie_id"] = metadata_df["movie_id"].astype(int)
    metadata_df["genres"] = metadata_df["genres"].fillna("").astype(str)
    metadata_df["age_rating"] = metadata_df["age_rating"].fillna("Teen").astype(str)

    return metadata_df


movie_metadata = load_movie_metadata()

GENRE_VIBE_KEYWORDS = {
    "action": ["action", "fight", "fighting", "explosive", "battle", "chase", "fast paced", "fast-paced", "adrenaline", "exciting"],
    "adventure": ["adventure", "journey", "quest", "explore", "exploration", "epic", "treasure"],
    "animation": ["animation", "animated", "cartoon", "anime"],
    "children": ["children", "child", "kids", "kid friendly", "for kids"],
    "comedy": ["comedy", "funny", "laugh", "hilarious", "light", "lighthearted", "feel good", "feel-good", "comfort", "silly"],
    "crime": ["crime", "criminal", "mafia", "gangster", "detective", "mystery", "noir", "investigation"],
    "documentary": ["documentary", "real", "true story", "educational", "informative", "factual"],
    "drama": ["drama", "serious", "emotional", "deep", "sad", "touching", "character driven", "character-driven"],
    "family": ["family", "kids", "children", "wholesome"],
    "fantasy": ["fantasy", "magic", "magical", "myth", "mythical", "fairy tale", "fairytale"],
    "health & fitness": ["fitness", "workout", "exercise", "yoga", "pilates"],
    "history": ["history", "historical", "period", "ancient"],
    "horror": ["horror", "scary", "terrifying", "dark", "creepy", "fear", "supernatural", "haunted"],
    "international": ["international", "foreign", "foreign language", "subtitles", "bollywood", "french", "indian"],
    "martial arts": ["martial arts", "kung fu", "karate", "samurai", "bruce lee", "jackie chan"],
    "music": ["music", "musical", "song", "dance", "concert", "singer", "band"],
    "romance": ["romance", "romantic", "love", "relationship", "date night", "love story"],
    "science fiction": ["sci-fi", "scifi", "science fiction", "space", "future", "alien", "robot", "futuristic", "dystopian", "mind bending", "mind-bending", "time travel"],
    "sports": ["sports", "sport", "football", "basketball", "boxing", "racing", "wrestling", "golf"],
    "thriller": ["thriller", "suspense", "tense", "intense", "twist", "plot twist", "suspenseful"],
    "travel": ["travel", "world", "places", "nature trip"],
    "tv show": ["tv show", "series", "season", "episodes", "sitcom"],
    "war": ["war", "military", "soldier", "army"],
    "western": ["western", "cowboy"],
}

MATURITY_LEVELS = {
    "3+": 1,
    "7+": 2,
    "Teen": 3,
    "16+": 4,
    "18+": 5,
}


def rating_to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    return "Positive"


def predict_review_sentiment(review_text, rating):
    if NLP_MODEL_PATH.exists():
        try:
            nlp_model = joblib.load(NLP_MODEL_PATH)
            return nlp_model.predict([review_text])[0]
        except Exception:
            pass

    return rating_to_sentiment(rating)


def build_feedback_response(title, rating, review_text, sentiment_label):
    review_lower = review_text.lower()

    if sentiment_label == "Positive":
        if rating >= 4:
            return (
                f"It sounds like {title} worked well for you. "
                "I will look for more movies with a similar feel next time."
            )
        return (
            f"I picked up a positive reaction to {title}, even though the rating was moderate. "
            "I will treat it as a partial match and tune future picks carefully."
        )

    if sentiment_label == "Negative":
        if "slow" in review_lower or "boring" in review_lower or "bored" in review_lower:
            return (
                f"Got it, {title} felt too slow for you. "
                "I will avoid slower picks and lean toward movies with stronger pace."
            )
        if "ending" in review_lower:
            return (
                f"That ending clearly did not land for you. "
                "I will be more careful with similar movies in future recommendations."
            )
        return (
            f"Thanks for being direct about {title}. "
            "I will move away from movies with a similar pattern for you."
        )

    if rating >= 4:
        return (
            f"Your review sounds mixed, but the rating says {title} still worked for you. "
            "I will keep similar movies in the pool, just not too aggressively."
        )

    if rating <= 2:
        return (
            f"Your review sounds balanced, but the low rating tells me {title} was not a good fit. "
            "I will reduce similar recommendations."
        )

    return (
        f"Sounds like {title} was somewhere in the middle for you. "
        "I will use that as a neutral signal and keep exploring better matches."
    )


def get_feedback_df():
    conn = get_db()
    rows = db_execute(
        conn,
        """
        SELECT app_user_id AS user_id, movie_id, title, rating,
               review_text, sentiment_label
        FROM feedback
        """
    ).fetchall()
    conn.close()
    return pd.DataFrame([dict(row) for row in rows])


def safe_series_get(series, key, default=0):
    if series is None:
        return default
    value = series.get(key, default)
    if value == default:
        value = series.get(str(key), default)
    return value


def build_feedback_biases(global_mean):
    feedback_df = get_feedback_df()
    if feedback_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    user_mean = feedback_df.groupby("user_id")["rating"].mean()
    movie_mean = feedback_df.groupby("movie_id")["rating"].mean()

    return user_mean - global_mean, movie_mean - global_mean


def build_user_feedback_profile(user_id, feedback_df):
    if recommender_model is None or feedback_df.empty:
        return None

    movie_factors = recommender_model.get("movie_factors")
    movie_id_to_idx = recommender_model.get("movie_id_to_idx")

    if movie_factors is None or movie_id_to_idx is None:
        return None

    global_mean = recommender_model["global_mean"]
    user_feedback = feedback_df[feedback_df["user_id"] == user_id]

    if user_feedback.empty:
        return None

    weighted_vectors = []
    total_weight = 0

    for row in user_feedback.itertuples(index=False):
        movie_idx = movie_id_to_idx.get(int(row.movie_id))
        if movie_idx is None:
            movie_idx = movie_id_to_idx.get(str(row.movie_id))
        if movie_idx is None:
            continue

        weight = float(row.rating) - global_mean
        if abs(weight) < 0.1:
            continue

        weighted_vectors.append(weight * movie_factors[movie_idx])
        total_weight += abs(weight)

    if not weighted_vectors or total_weight == 0:
        return None

    return np.sum(weighted_vectors, axis=0) / total_weight


def predict_rating(user_id, movie_id, feedback_user_bias, feedback_movie_bias, user_profile=None):
    global_mean = recommender_model["global_mean"]
    base_movie_bias = recommender_model.get("movie_bias", pd.Series(dtype=float))
    movie_factors = recommender_model.get("movie_factors")
    movie_id_to_idx = recommender_model.get("movie_id_to_idx")

    base_pred = None

    if user_profile is not None and movie_factors is not None and movie_id_to_idx is not None:
        movie_idx = movie_id_to_idx.get(int(movie_id))
        if movie_idx is None:
            movie_idx = movie_id_to_idx.get(str(movie_id))
        if movie_idx is not None:
            base_pred = global_mean + float(np.dot(user_profile, movie_factors[movie_idx]))

    if base_pred is None:
        # App users are new users, so we learn their taste from website feedback.
        u_bias = safe_series_get(feedback_user_bias, user_id, 0)
        m_bias = safe_series_get(base_movie_bias, movie_id, 0)
        base_pred = global_mean + u_bias + m_bias

    movie_feedback_adjustment = safe_series_get(feedback_movie_bias, movie_id, 0)
    pred = base_pred + 0.20 * movie_feedback_adjustment
    return float(np.clip(pred, 1, 5))


def extract_request_keywords(message):
    stop_words = {
        "a", "an", "and", "are", "for", "i", "in", "is", "me", "movie",
        "of", "please", "recommend", "show", "something", "that", "the",
        "to", "want", "watch", "with", "film", "genre", "vibe"
    }
    words = [
        word.strip(".,!?;:()[]{}\"'").lower()
        for word in message.split()
    ]
    return [word for word in words if len(word) > 2 and word not in stop_words]


def extract_requested_genres(message):
    message = message.lower()
    requested_genres = []

    for genre, keywords in GENRE_VIBE_KEYWORDS.items():
        if any(keyword in message for keyword in keywords):
            requested_genres.append(genre)

    return requested_genres


def genre_match_score(genres, requested_genres):
    if not requested_genres:
        return 0

    genres = str(genres).lower()
    return sum(genre in genres for genre in requested_genres)


def extract_maturity_preference(message):
    message = message.lower()

    if any(phrase in message for phrase in ["not adult", "no adult", "not 18", "without adult"]):
        return {"target": "Teen", "max_level": 3}

    if any(phrase in message for phrase in ["18+", "adult", "mature audience", "for adults", "grown up"]):
        return {"target": "18+", "min_level": 5}

    if any(phrase in message for phrase in ["16+", "older teen", "older teens", "older than teen", "older than teens"]):
        return {"target": "16+", "min_level": 4}

    if any(phrase in message for phrase in ["teen", "teenager", "13+", "pg-13"]):
        return {"target": "Teen", "max_level": 3}

    if any(phrase in message for phrase in ["7+", "older kids", "for kids", "kid friendly"]):
        return {"target": "7+", "max_level": 2}

    if any(phrase in message for phrase in ["3+", "all ages", "for everyone", "family friendly", "family-friendly"]):
        return {"target": "3+", "max_level": 1}

    return None


def get_maturity_level(age_rating):
    return MATURITY_LEVELS.get(str(age_rating).strip(), MATURITY_LEVELS["Teen"])


def maturity_allowed(age_rating, preference):
    if not preference:
        return True

    level = get_maturity_level(age_rating)
    if "max_level" in preference:
        return level <= preference["max_level"]
    if "min_level" in preference:
        return level >= preference["min_level"]
    return True


def maturity_match_score(age_rating, preference):
    if not preference:
        return 0

    level = get_maturity_level(age_rating)
    target_level = get_maturity_level(preference["target"])

    if level == target_level:
        return 1.0
    if maturity_allowed(age_rating, preference):
        return 0.35
    return -3.0


def recommend_movies(user_id, user_message="", top_n=1):
    if recommender_model is None:
        return []

    movie_titles = recommender_model["movie_titles"][["movie_id", "title"]].drop_duplicates()
    if not movie_metadata.empty:
        movie_titles = movie_titles.merge(
            movie_metadata[["movie_id", "genres", "age_rating"]],
            on="movie_id",
            how="left",
        )
    else:
        movie_titles["genres"] = ""
        movie_titles["age_rating"] = "Teen"

    movie_titles["genres"] = movie_titles["genres"].fillna("")
    movie_titles["age_rating"] = movie_titles["age_rating"].fillna("Teen")

    feedback_user_bias, feedback_movie_bias = build_feedback_biases(
        recommender_model["global_mean"]
    )

    feedback_df = get_feedback_df()
    user_profile = build_user_feedback_profile(user_id, feedback_df)

    watched = []
    if not feedback_df.empty:
        watched = feedback_df[feedback_df["user_id"] == user_id]["movie_id"].unique()

    unseen_movies = movie_titles[~movie_titles["movie_id"].isin(watched)].copy()
    unseen_movies["predicted_rating"] = [
        predict_rating(
            user_id,
            movie_id,
            feedback_user_bias,
            feedback_movie_bias,
            user_profile=user_profile,
        )
        for movie_id in unseen_movies["movie_id"]
    ]

    keywords = extract_request_keywords(user_message)
    if keywords:
        unseen_movies["title_match_score"] = unseen_movies["title"].str.lower().apply(
            lambda title: sum(keyword in title for keyword in keywords)
        )
    else:
        unseen_movies["title_match_score"] = 0

    requested_genres = extract_requested_genres(user_message)
    unseen_movies["genre_match_score"] = unseen_movies["genres"].apply(
        lambda genres: genre_match_score(genres, requested_genres)
    )

    maturity_preference = extract_maturity_preference(user_message)
    unseen_movies["maturity_match_score"] = unseen_movies["age_rating"].apply(
        lambda age_rating: maturity_match_score(age_rating, maturity_preference)
    )

    if maturity_preference:
        maturity_filtered_movies = unseen_movies[
            unseen_movies["age_rating"].apply(
                lambda age_rating: maturity_allowed(age_rating, maturity_preference)
            )
        ]
        if not maturity_filtered_movies.empty:
            unseen_movies = maturity_filtered_movies

    unseen_movies["final_score"] = (
        unseen_movies["predicted_rating"] + unseen_movies["title_match_score"] * 0.35
        + unseen_movies["genre_match_score"] * 0.75
        + unseen_movies["maturity_match_score"] * 0.90
    )

    recommendations = unseen_movies.sort_values(
        by=["final_score", "predicted_rating"], ascending=False
    ).head(top_n)

    records = []
    for movie in recommendations.to_dict("records"):
        records.append(
            {
                "movie_id": int(movie["movie_id"]),
                "title": str(movie["title"]),
                "predicted_rating": float(movie["predicted_rating"]),
                "genres": str(movie.get("genres", "")),
                "age_rating": str(movie.get("age_rating", "Teen")),
            }
        )

    return records


def save_chat_message(user_id, role, message, movie=None):
    movie = movie or {}
    conn = get_db()
    db_execute(
        conn,
        """
        INSERT INTO chat_messages (
            app_user_id, role, message, movie_id, title,
            predicted_rating, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            role,
            message,
            movie.get("movie_id"),
            movie.get("title"),
            movie.get("predicted_rating"),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_chat_history(user_id):
    conn = get_db()
    rows = db_execute(
        conn,
        """
        SELECT * FROM chat_messages
        WHERE app_user_id = ?
        ORDER BY created_at ASC
        """,
        (user_id,),
    ).fetchall()
    conn.close()
    return rows


def get_public_reviews(limit=24):
    conn = get_db()
    rows = db_execute(
        conn,
        """
        SELECT title, rating, review_text, sentiment_label, created_at
        FROM feedback
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return rows


def train_nlp_model_from_feedback():
    feedback_df = get_feedback_df()
    if feedback_df.empty:
        return False, "No feedback data yet."

    nlp_df = feedback_df.dropna(subset=["review_text", "sentiment_label"]).copy()
    if nlp_df.shape[0] < 50:
        return False, "Need at least 50 feedback reviews before training NLP."
    if nlp_df["sentiment_label"].nunique() < 2:
        return False, "Need at least 2 sentiment classes before training NLP."

    nlp_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("lgModel", LogisticRegression(max_iter=1000)),
        ]
    )

    nlp_model.fit(nlp_df["review_text"], nlp_df["sentiment_label"])
    joblib.dump(nlp_model, NLP_MODEL_PATH)

    return True, "NLP model trained and saved."


def create_recommendation_response(user_id, user_message):
    save_chat_message(user_id, "user", user_message)
    recommendations = recommend_movies(user_id, user_message=user_message, top_n=1)

    if not recommendations:
        assistant_message = "I could not find a recommendation because the model file is missing."
        save_chat_message(user_id, "assistant", assistant_message)
        return {
            "ok": False,
            "error": assistant_message,
            "user_message": user_message,
            "assistant_message": assistant_message,
        }

    movie = recommendations[0]
    genres_text = movie.get("genres", "").strip()
    genre_sentence = f" It matches this kind of request through: {genres_text}." if genres_text else ""
    age_rating = movie.get("age_rating", "Teen")
    maturity_sentence = f" Estimated maturity rating: {age_rating}."
    assistant_message = (
        f"I recommend {movie['title']}. "
        f"My predicted rating for you is {movie['predicted_rating']:.2f}/5. "
        f"{genre_sentence}{maturity_sentence} "
        "Watch it, then tell me how you felt and rate it."
    )

    save_chat_message(user_id, "assistant", assistant_message, movie)
    session["pending_movie"] = movie

    return {
        "ok": True,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "movie": movie,
    }


def save_feedback_response(user_id, movie_id, title, rating, review_text):
    sentiment_label = predict_review_sentiment(review_text, rating)

    conn = get_db()
    db_execute(
        conn,
        """
        INSERT INTO feedback (
            app_user_id, netflix_user_id, movie_id, title, rating,
            review_text, sentiment_label, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            user_id,
            movie_id,
            title,
            rating,
            review_text,
            sentiment_label,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    train_nlp_model_from_feedback()

    user_message = f"I rated {title} {rating}/5. {review_text}"
    assistant_message = build_feedback_response(title, rating, review_text, sentiment_label)

    save_chat_message(user_id, "user", user_message)
    save_chat_message(user_id, "assistant", assistant_message)
    session.pop("pending_movie", None)

    return {
        "ok": True,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "review": {
            "title": title,
            "rating": rating,
            "review_text": review_text,
            "sentiment_label": sentiment_label,
        },
    }


@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        if not username or not password:
            flash("Username and password are required.", "danger")
            return redirect(url_for("signup"))

        conn = get_db()
        try:
            db_execute(
                conn,
                """
                INSERT INTO users (username, password_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (username, generate_password_hash(password), datetime.now().isoformat()),
            )
            conn.commit()
            flash("Account created. You can login now.", "success")
            return redirect(url_for("login"))
        except Exception:
            conn.rollback()
            flash("This username already exists.", "danger")
        finally:
            conn.close()

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        conn = get_db()
        user = db_execute(
            conn,
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


@app.route("/api/recommend", methods=["POST"])
@login_required
def api_recommend():
    data = request.get_json(silent=True) or request.form
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"ok": False, "error": "Write a message first."}), 400

    return jsonify(create_recommendation_response(session["user_id"], user_message))


@app.route("/api/feedback", methods=["POST"])
@login_required
def api_feedback():
    data = request.get_json(silent=True) or request.form

    try:
        movie_id = int(data.get("movie_id"))
        rating = int(data.get("rating"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid movie or rating."}), 400

    title = data.get("title", "").strip()
    review_text = data.get("review_text", "").strip()

    if not title or not review_text:
        return jsonify({"ok": False, "error": "Review text is required."}), 400

    return jsonify(
        save_feedback_response(
            session["user_id"],
            movie_id,
            title,
            rating,
            review_text,
        )
    )


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    pending_movie = session.get("pending_movie")

    if request.method == "POST":
        user_message = request.form["message"].strip()
        if not user_message:
            flash("Write a message first.", "warning")
            return redirect(url_for("dashboard"))

        result = create_recommendation_response(session["user_id"], user_message)
        if result["ok"]:
            pending_movie = result["movie"]
        else:
            flash(result["assistant_message"], "danger")

    return render_template(
        "dashboard.html",
        chat_history=get_chat_history(session["user_id"]),
        recent_reviews=get_public_reviews(limit=6),
        pending_movie=pending_movie,
        model_ready=recommender_model is not None,
    )


@app.route("/feedback", methods=["POST"])
@login_required
def feedback():
    movie_id = int(request.form["movie_id"])
    title = request.form["title"]
    rating = int(request.form["rating"])
    review_text = request.form["review_text"].strip()
    save_feedback_response(session["user_id"], movie_id, title, rating, review_text)

    return redirect(url_for("dashboard"))


@app.route("/my-feedback")
@login_required
def my_feedback():
    conn = get_db()
    rows = db_execute(
        conn,
        """
        SELECT * FROM feedback
        WHERE app_user_id = ?
        ORDER BY created_at DESC
        """,
        (session["user_id"],),
    ).fetchall()
    conn.close()
    return render_template("my_feedback.html", rows=rows)


@app.route("/reviews")
@login_required
def reviews():
    return render_template("reviews.html", rows=get_public_reviews(limit=60))


init_db()


if __name__ == "__main__":
    app.run(debug=True)
