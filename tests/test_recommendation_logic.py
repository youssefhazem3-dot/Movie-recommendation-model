import unittest

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import app


class RecommendationLogicTests(unittest.TestCase):
    def setUp(self):
        self.original_model = app.recommender_model
        self.original_metadata = app.movie_metadata
        self.original_biases = app.build_feedback_biases
        self.original_feedback = app.get_feedback_df

        app.recommender_model = {
            "movie_titles": pd.DataFrame({
                "movie_id": [1, 2],
                "title": ["Action One", "Comedy Two"],
            }),
            "global_mean": 3.5,
            "movie_bias": pd.Series({1: 0.1, 2: 0.2}),
            "movie_factors": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "movie_id_to_idx": {1: 0, 2: 1},
        }
        app.movie_metadata = pd.DataFrame({
            "movie_id": [1, 2],
            "genres": ["Action", "Comedy"],
            "age_rating": ["Teen", "Teen"],
        })
        app.build_feedback_biases = lambda _: (pd.Series(dtype=float), pd.Series(dtype=float))
        app.get_feedback_df = lambda: pd.DataFrame()

    def tearDown(self):
        app.recommender_model = self.original_model
        app.movie_metadata = self.original_metadata
        app.build_feedback_biases = self.original_biases
        app.get_feedback_df = self.original_feedback

    def test_recommendation_without_reference_title_does_not_crash(self):
        recommendations = app.recommend_movies(1, "recommend a funny movie")

        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]["title"], "Comedy Two")

    def test_unknown_review_vocabulary_is_neutral(self):
        model = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("lgModel", LogisticRegression(max_iter=200)),
        ])
        model.fit(
            ["great enjoyable", "awful boring", "excellent movie", "terrible film"],
            ["Positive", "Negative", "Positive", "Negative"],
        )

        self.assertEqual(
            app.get_nlp_prediction(model, "qzxv blorf snargle"),
            "Neutral",
        )


if __name__ == "__main__":
    unittest.main()