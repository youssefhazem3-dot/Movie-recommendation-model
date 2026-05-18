# Movie Recommender Model Report

## 1. Project Overview

This project is a movie recommendation website that combines collaborative filtering, user feedback, and NLP sentiment analysis. The user signs up, chats with the system, asks for a movie based on a genre or vibe, receives one recommendation, then gives a rating from 1 to 5 and writes a review about how they felt while watching the movie.

The system learns from two types of information:

- Historical Netflix rating data: `user_id`, `movie_id`, `rating`, `date`, `year`, and `title`.
- Website feedback data: the logged-in user's rating and written review after watching a recommended movie.

The final deployed system is a hybrid feedback-aware recommender:

```text
SVD Matrix Factorization
        +
User Feedback Reranking
        +
NLP Sentiment Classification
        +
Genre, Vibe, and Age Rating Matching
```

The goal is not only to recommend popular or highly rated movies, but to adapt future recommendations based on what each website user says and rates.

## 2. Datasets Used

### 2.1 Netflix Ratings Dataset

The original Netflix dataset was converted from the raw combined text files into a clean CSV file called `netflix_cleaned.csv`.

Original cleaned shape:

```text
Shape: (100480507, 6)
Columns: ['user_id', 'movie_id', 'rating', 'date', 'year', 'title']
```

The important columns for recommendation are:

| Column | Meaning |
|---|---|
| `user_id` | The original Netflix user identifier |
| `movie_id` | The movie identifier |
| `rating` | User rating from 1 to 5 |
| `title` | Movie title |

The `date` and `year` columns were dropped for the main model because the current recommender focuses on user-movie preference patterns, not time-based recommendation.

### 2.2 Movie Metadata Dataset

The file `Movies 67.csv` is used as movie metadata for the deployed website.

Current columns:

```text
movie_id, year, title, genres, age_rating
```

This file helps the chat system understand requests like:

- "recommend a funny movie"
- "I want sci-fi"
- "give me a family friendly movie"
- "I want something 18+"
- "show me a movie for older teens"

Missing genres were generated using a metadata enrichment script. The script filled all missing genre values.

Genre enrichment result:

```text
Original missing genres: 3760
Remaining missing genres: 0
Rule-based fills: 2715
Existing title matches: 161
Fallback Drama: 884
```

The `age_rating` column is an inferred maturity bucket, not an official movie certificate. The generated categories are:

```text
3+
7+
Teen
16+
18+
```

Age rating distribution:

```text
Teen    8138
16+     5261
7+      3391
3+       808
18+      175
```

### 2.3 ACL IMDb Review Dataset

The NLP sentiment model was trained using the ACL IMDb review dataset.

Training sample used in the notebook:

```text
Positive reviews: 5000
Negative reviews: 5000
Total training reviews: 10000
```

Testing sample:

```text
Positive reviews: 2000
Negative reviews: 2000
Total testing reviews: 4000
```

The IMDb dataset teaches the NLP model how to classify written reviews as positive or negative.

## 3. Data Preprocessing

### 3.1 Netflix Data Conversion

The original Netflix files were stored in this format:

```text
movie_id:
user_id,rating,date
user_id,rating,date
```

The notebook converted the four combined text files into a clean table:

```text
user_id,movie_id,rating,date
```

Then movie titles were merged using the movie titles file.

### 3.2 Sampling

Because the full Netflix dataset contains more than 100 million rows, the notebook uses sampling so training remains practical on a normal laptop.

Baseline sample:

```text
1,000,000 rows
```

SVD training sample:

```text
200,000 rows
```

Sampling is done with:

```python
random_state=42
```

This makes the experiment reproducible.

### 3.3 Column Cleaning

For modeling, unnecessary columns were dropped:

```text
Dropped: date, year
Kept: user_id, movie_id, rating, title
```

The `year` column was not used in the recommender because it does not directly describe whether a user liked a movie. It can be added later for decade-based filtering if needed.

## 4. Baseline Recommendation Model

Before building the final hybrid recommender, a simple baseline model was created.

The baseline uses:

- Global average rating
- User rating bias
- Movie rating bias

The formula is:

```text
predicted_rating = global_mean + user_bias + movie_bias
```

Where:

```text
user_bias = user's average rating - global_mean
movie_bias = movie's average rating - global_mean
```

Predictions are clipped between 1 and 5:

```text
1 <= predicted_rating <= 5
```

Baseline train/test split:

```text
X_train: 800000 rows
X_test: 200000 rows
test_size = 0.20
random_state = 42
```

Baseline evaluation:

```text
RMSE: 1.0946
MAE: 0.8455
```

This baseline is useful because it gives a simple comparison point before using SVD.

## 5. SVD Matrix Factorization Model

The main recommendation model is based on SVD matrix factorization.

### 5.1 Why SVD Was Used

The Netflix dataset is perfect for matrix factorization because it contains user-movie-rating interactions.

SVD learns hidden preference patterns, also called latent factors. These factors can represent abstract movie taste dimensions, such as:

- comedy vs. drama preference
- action/adventure preference
- family movie preference
- serious vs. light movie preference

The model does not manually define these factors. It learns them from rating patterns.

### 5.2 SVD Training Data

The SVD model was trained on:

```text
SVD dataframe shape: (200000, 4)
Unique users: 127825
Unique movies: 11466
```

The columns used were:

```text
user_id, movie_id, rating, title
```

### 5.3 Encoding Users and Movies

SVD requires matrix row and column indexes, not raw IDs. Therefore:

- `user_id` was encoded into `user_idx`
- `movie_id` was encoded into `movie_idx`

This created:

```text
Encoded users: 127825
Encoded movies: 11466
```

### 5.4 Sparse User-Movie Matrix

The model builds a sparse matrix:

```text
rows = users
columns = movies
values = rating - global_mean
```

The ratings are centered around the global mean before training:

```text
centered_rating = rating - global_mean
```

Centering helps the SVD model focus on preference differences instead of only learning that most ratings are around the average.

### 5.5 SVD Algorithm

The model uses:

```python
TruncatedSVD(n_components=30, random_state=42)
```

This means each user and movie is represented by a 30-dimensional latent vector.

Training output:

```text
User factors shape: (127825, 30)
Movie factors shape: (11466, 30)
Explained variance: 0.0561
```

### 5.6 SVD Prediction Formula

For a known user and movie:

```text
predicted_rating = global_mean + dot(user_vector, movie_vector)
```

Then the prediction is clipped:

```text
1 <= predicted_rating <= 5
```

For unknown users or unknown movies, the notebook returns the global mean rating as a fallback.

### 5.7 SVD Evaluation

SVD test evaluation:

```text
RMSE: 1.0896
MAE: 0.9139
```

The RMSE is slightly better than the baseline model. The MAE is higher, which shows that the SVD model is useful but not perfect. This is one reason the final project uses SVD together with feedback reranking instead of relying only on raw SVD predictions.

## 6. Hybrid Feedback-Aware Recommender

The deployed website does not use plain SVD alone. It uses a hybrid recommendation system.

The saved model file is:

```text
netflix_recommendation_model.joblib
```

Saved model type:

```text
hybrid_svd_feedback_reranker
```

The saved object includes:

| Object | Purpose |
|---|---|
| `global_mean` | Default average rating |
| `movie_titles` | Movie IDs and titles |
| `movie_bias` | Movie popularity/preference bias |
| `user_bias` | Original Netflix user bias |
| `svd_model` | Trained SVD object |
| `movie_factors` | Learned movie latent vectors |
| `movie_id_to_idx` | Lookup dictionary from movie ID to vector index |
| `n_components` | Number of SVD latent factors |

### 6.1 Why Feedback Reranking Is Needed

The website users are new users. They are not the same as the original Netflix users, so the deployed app cannot always use a historical Netflix `user_id` directly.

Instead, the app learns each website user's profile from feedback:

```text
rating + written review
```

After the user rates movies, the system builds a preference profile based on the movie vectors of movies they reviewed.

### 6.2 User Feedback Profile

For each reviewed movie, the system looks up the movie's SVD vector.

The weight is:

```text
weight = user_rating - global_mean
```

If the user gives a high rating, the movie vector pulls the profile toward similar movies.

If the user gives a low rating, the movie vector pushes the profile away from similar movies.

The profile is calculated as:

```text
user_profile = sum(weight * movie_vector) / sum(abs(weight))
```

This lets the website create a personalized vector for a new user without retraining the full SVD model every time.

### 6.3 Prediction in the Website

If the user has enough feedback to build a profile:

```text
base_prediction = global_mean + dot(user_profile, movie_vector)
```

If the user does not have enough feedback yet:

```text
base_prediction = global_mean + website_user_feedback_bias + movie_bias
```

Then the model also applies a small movie-level feedback adjustment:

```text
prediction = base_prediction + 0.20 * movie_feedback_adjustment
```

Finally:

```text
prediction = clip(prediction, 1, 5)
```

### 6.4 Excluding Already Reviewed Movies

The system excludes movies already reviewed by the current website user. This prevents the model from recommending the same movie repeatedly.

## 7. Chat-Based Recommendation Layer

The user does not type a Netflix user ID anymore. Instead, the user chats naturally with the system.

Example prompts:

```text
I want a funny family movie
Recommend a teen sci-fi movie
I want something scary and 18+
Give me a movie for all ages
I want an emotional drama
```

The app extracts:

- title keywords
- requested genres
- requested vibe
- requested age/maturity level

Then it reranks candidate movies.

### 7.1 Genre and Vibe Matching

The app has a dictionary called `GENRE_VIBE_KEYWORDS`.

Examples:

| User words | Matched genre/vibe |
|---|---|
| funny, laugh, lighthearted | comedy |
| scary, creepy, haunted | horror |
| sci-fi, space, alien, future | science fiction |
| detective, mafia, mystery | crime |
| romantic, love story | romance |
| anime, animated, cartoon | animation |
| kung fu, karate, samurai | martial arts |

The score is:

```text
genre_match_score = number of requested genres found in movie genres
```

### 7.2 Age Rating Matching

The system supports these maturity levels:

```text
3+ < 7+ < Teen < 16+ < 18+
```

The app maps them internally as:

```text
3+   = 1
7+   = 2
Teen = 3
16+  = 4
18+  = 5
```

Examples:

| User request | System interpretation |
|---|---|
| family friendly, all ages | `3+` |
| for kids, older kids | `7+` |
| teen, PG-13 | `Teen` |
| older teens, 16+ | `16+` |
| adult, mature, 18+ | `18+` |
| not adult | `Teen` or below |

If a maturity preference is detected, the app filters out movies that do not fit when possible.

### 7.3 Final Ranking Formula

The final website recommendation score is:

```text
final_score =
    predicted_rating
    + 0.35 * title_match_score
    + 0.75 * genre_match_score
    + 0.90 * maturity_match_score
```

This means the model still respects predicted rating, but it can move a movie higher if it matches the user's requested vibe or age category.

## 8. NLP Sentiment Model

The NLP part of the project classifies written user reviews.

The saved NLP model file is:

```text
netflix_review_nlp_model.joblib
```

### 8.1 NLP Preprocessing

The model uses TF-IDF vectorization.

TF-IDF means Term Frequency-Inverse Document Frequency. It converts text into numerical features by giving higher importance to words or phrases that are meaningful in reviews.

The notebook NLP vectorizer uses:

```python
TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)
```

This means:

- English stop words are removed.
- Single words and two-word phrases are used.
- Very rare terms are ignored.
- The vocabulary is limited to 50,000 features.

### 8.2 NLP Classifier

The sentiment classifier is Logistic Regression:

```python
LogisticRegression(max_iter=1000, random_state=42)
```

The full NLP pipeline is:

```text
Review text
    -> TF-IDF tokenization/vectorization
    -> Logistic Regression
    -> Positive or Negative sentiment
```

### 8.3 NLP Evaluation

The model was tested on 4,000 IMDb reviews.

Evaluation:

```text
Accuracy: 0.8852

Negative precision: 0.88
Negative recall:    0.89
Negative f1-score:  0.89

Positive precision: 0.89
Positive recall:    0.88
Positive f1-score:  0.88
```

This shows that the NLP model learned useful positive/negative sentiment patterns.

## 9. Website Feedback Learning

When the user submits feedback, they provide:

```text
movie_id
title
rating from 1 to 5
written review
```

The system predicts the review sentiment using the NLP model.

If the saved NLP model is not available, the app falls back to the numeric rating:

```text
rating 1-2 -> Negative
rating 3   -> Neutral
rating 4-5 -> Positive
```

The feedback is saved in the database and used for future recommendations.

### 9.1 Online NLP Retraining

The website can retrain the NLP model from user feedback after enough reviews are collected.

Retraining conditions:

```text
At least 50 feedback reviews
At least 2 sentiment classes
```

The online retraining pipeline uses:

```python
Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('lgModel', LogisticRegression(max_iter=1000))
])
```

This lets the website gradually adapt to the language used by its real users.

The minimum was set to 50 reviews instead of a very small number such as 5 because online NLP retraining can overfit badly when the dataset is tiny. With only a few reviews, the classifier may memorize specific words from those reviews instead of learning general sentiment patterns.

## 10. Database Design

The deployed website uses PostgreSQL on Railway.

The app reads:

```text
DATABASE_URL
```

If `DATABASE_URL` exists, the app uses PostgreSQL. If it does not exist, the app falls back to SQLite for local development.

### 10.1 Tables

#### users

Stores website accounts.

| Column | Purpose |
|---|---|
| `id` | User primary key |
| `username` | Unique username |
| `password_hash` | Secure hashed password |
| `created_at` | Account creation time |

#### feedback

Stores user ratings and reviews.

| Column | Purpose |
|---|---|
| `id` | Feedback primary key |
| `app_user_id` | Website user ID |
| `netflix_user_id` | Kept for compatibility with earlier model design |
| `movie_id` | Recommended movie ID |
| `title` | Movie title |
| `rating` | User rating from 1 to 5 |
| `review_text` | Written user review |
| `sentiment_label` | NLP sentiment result |
| `created_at` | Feedback time |

#### chat_messages

Stores the chat conversation.

| Column | Purpose |
|---|---|
| `id` | Message primary key |
| `app_user_id` | Website user ID |
| `role` | `user` or `assistant` |
| `message` | Chat message |
| `movie_id` | Recommended movie ID when applicable |
| `title` | Recommended movie title when applicable |
| `predicted_rating` | Model's predicted rating |
| `created_at` | Message time |

## 11. Deployment Architecture

The project is deployed on Railway.

Main deployment files:

| File | Purpose |
|---|---|
| `app.py` | Flask application |
| `main.py` | Imports the Flask app |
| `Procfile` | Start command for deployment |
| `railway.json` | Railway start command |
| `requirements.txt` | Python dependencies |
| `templates/` | HTML pages |
| `static/` | CSS and JavaScript |
| `netflix_recommendation_model.joblib` | Saved recommender |
| `netflix_review_nlp_model.joblib` | Saved NLP model |
| `Movies 67.csv` | Movie metadata with genres and age ratings |

Railway start command:

```text
gunicorn app:app --bind 0.0.0.0:$PORT
```

## 12. Full Recommendation Flow

The system works like this:

1. User signs up or logs in.
2. User writes a chat request.
3. The app extracts keywords, genre/vibe signals, and maturity preference.
4. The recommender scores unseen movies.
5. The app returns the best movie recommendation.
6. The feedback form appears.
7. User rates the movie from 1 to 5 and writes a review.
8. NLP model classifies the review sentiment.
9. Feedback is saved in PostgreSQL.
10. Future recommendations use the user's feedback profile.
11. After enough feedback, the NLP model can retrain from website reviews.

## 13. Strengths of the Model

- Uses real historical Netflix rating data.
- Uses SVD matrix factorization, which is appropriate for user-item recommendation.
- Supports new website users through feedback-based profiling.
- Uses NLP to understand written user reviews.
- Stores user feedback in PostgreSQL, so learning is remembered after redeploys.
- Supports chat-based recommendation instead of requiring a Netflix user ID.
- Uses genre, vibe, and maturity preferences from natural language.
- Can show anonymous public reviews to make the site feel active and user-centered.

## 14. Limitations

- The maturity ratings are inferred, not official certificates.
- Some generated genres are rule-based and may not be perfectly accurate.
- The SVD model was trained on a sample, not the full 100 million rows, to keep training practical.
- The online NLP retraining is simple and may overfit if the feedback dataset is still small.
- The current model recommends one movie at a time.
- The model does not yet use movie posters, descriptions, actors, directors, or release decade as ranking features.
- The recommender does not retrain the full SVD model after each review. Instead, it updates the user's feedback profile and reranks recommendations.

## 15. Future Improvements

Possible improvements:

- Train SVD on a larger sample or the full dataset using a stronger machine.
- Add movie descriptions and use semantic embeddings for better chat understanding.
- Add official age certificates from TMDB, IMDb, or another metadata provider.
- Add poster images and movie detail pages.
- Recommend multiple movies and let the user choose.
- Add a scheduled retraining job that periodically retrains SVD with collected website feedback.
- Add LightGCN or neural collaborative filtering.
- Explore reinforcement learning as future work, where user feedback acts as reward.
- Improve sentiment from Positive/Negative into emotion categories such as excited, bored, sad, confused, or satisfied.

## 16. Conclusion

This project implements a practical hybrid movie recommender. The core model is SVD matrix factorization trained on Netflix user ratings. The deployed website improves the base recommendation using feedback-aware reranking, NLP sentiment analysis, genre/vibe matching, and inferred maturity filtering.

The most important part of the system is the feedback loop. The user does not only receive a recommendation; they rate it and describe how they felt. That feedback is saved in PostgreSQL and used to shape future recommendations, making the system more personalized over time.
