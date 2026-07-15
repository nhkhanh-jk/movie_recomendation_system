import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class ContentBasedRecommender:
    """
    Content-Based Filtering using TF-IDF on movie genres.

    Each movie is represented as a TF-IDF vector of its genre tokens plus
    a decade tag (e.g. 'decade_1990'). Cosine similarity is computed between
    all pairs, giving a (n_movies x n_movies) matrix.

    For user recommendations, similarity scores are aggregated across all
    movies the user has rated highly, then unseen movies are ranked.
    """

    def __init__(self) -> None:
        self.movies = None
        self.sim_matrix = None
        self.movie_idx: dict[str, int] = {}
        self.id_to_idx: dict[int, int] = {}
        self._fitted = False

    def fit(self, movies: pd.DataFrame) -> "ContentBasedRecommender":
        self.movies = movies.reset_index(drop=True)

        features = self.movies.apply(self._build_feature_string, axis=1)

        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1)
        tfidf_matrix = vectorizer.fit_transform(features)

        self.sim_matrix = cosine_similarity(tfidf_matrix)

        self.movie_idx = {title: idx for idx, title in enumerate(self.movies["title"])}
        self.id_to_idx = {mid: idx for idx, mid in enumerate(self.movies["movie_id"])}

        self._fitted = True
        print(f"Content-Based: fitted on {len(self.movies)} movies")
        return self

    @staticmethod
    def _build_feature_string(row: pd.Series) -> str:
        parts = [row["genres"]]
        if "year" in row and pd.notna(row["year"]):
            decade = int(row["year"] // 10 * 10)
            parts.append(f"decade_{decade}")
        return " ".join(parts)

    def recommend_similar_movies(self, title: str, top_k: int = 10) -> pd.DataFrame:
        """Return top-K movies most similar to the given title."""
        self._check_fitted()

        if title not in self.movie_idx:
            raise ValueError(f"Movie '{title}' not found. Use get_movie_titles() to check.")

        idx = self.movie_idx[title]
        sim_scores = sorted(enumerate(self.sim_matrix[idx]), key=lambda x: x[1], reverse=True)
        sim_scores = [(i, s) for i, s in sim_scores if i != idx]
        sim_scores = sim_scores[:top_k]

        indices = [i for i, _ in sim_scores]
        scores  = [s for _, s in sim_scores]

        result = self.movies.iloc[indices][["title", "genres", "year"]].copy()
        result["similarity_score"] = np.round(scores, 4)
        return result.reset_index(drop=True)

    def recommend_for_user(
        self,
        user_id: int,
        ratings: pd.DataFrame,
        top_k: int = 10,
        rating_threshold: float = 3.5,
    ) -> pd.DataFrame:
        """Recommend unseen movies for a user based on their rating history.

        Aggregates similarity scores from all movies the user rated >= threshold,
        weighted by (rating - threshold), then returns the top-K unseen results.
        """
        self._check_fitted()

        user_ratings = ratings[ratings["user_id"] == user_id]
        if user_ratings.empty:
            raise ValueError(f"User {user_id} not found in ratings.")

        liked = user_ratings[user_ratings["rating"] >= rating_threshold]
        if liked.empty:
            liked = user_ratings

        agg_scores = np.zeros(len(self.movies))
        for _, row in liked.iterrows():
            mid = row["movie_id"]
            if mid not in self.id_to_idx:
                continue
            weight = max(row["rating"] - rating_threshold, 0.1)
            agg_scores += weight * self.sim_matrix[self.id_to_idx[mid]]

        seen_ids = set(user_ratings["movie_id"].tolist())
        for mid in seen_ids:
            if mid in self.id_to_idx:
                agg_scores[self.id_to_idx[mid]] = 0.0

        top_indices = np.argsort(agg_scores)[::-1][:top_k]
        result = self.movies.iloc[top_indices][["title", "genres", "year"]].copy()
        result["score"] = np.round(agg_scores[top_indices], 4)
        return result.reset_index(drop=True)

    def get_movie_titles(self) -> list[str]:
        self._check_fitted()
        return self.movies["title"].tolist()

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit(movies) before making recommendations.")

    def save(self, path: str | None = None) -> str:
        self._check_fitted()
        path = path or os.path.join(MODEL_DIR, "content_based.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | None = None) -> "ContentBasedRecommender":
        path = path or os.path.join(MODEL_DIR, "content_based.pkl")
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model


if __name__ == "__main__":
    from src.data_loader import load_movies, load_ratings

    movies  = load_movies()
    ratings = load_ratings()

    cb = ContentBasedRecommender()
    cb.fit(movies)
    cb.save()

    print("\nMovies similar to 'Toy Story (1995)':")
    print(cb.recommend_similar_movies("Toy Story (1995)", top_k=10).to_string(index=False))

    print("\nRecommendations for User 1:")
    print(cb.recommend_for_user(1, ratings, top_k=10).to_string(index=False))
