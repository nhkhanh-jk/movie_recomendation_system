import os
import numpy as np
import pandas as pd
import joblib

from src.content_based import ContentBasedRecommender
from src.matrix_factorization import MatrixFactorizationRecommender


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MIN_RATINGS_FOR_MF = 10   # users below this are treated as cold-start
DEFAULT_ALPHA_WARM = 0.3  # CB weight for warm users (MF dominates)
DEFAULT_ALPHA_COLD = 0.85 # CB weight for cold-start users


class HybridRecommender:
    """
    Weighted hybrid of Content-Based Filtering and Matrix Factorization.

    Final score: hybrid = alpha * cb_score + (1 - alpha) * mf_score

    Alpha is chosen automatically based on how many ratings the user has:
    - fewer than MIN_RATINGS_FOR_MF -> cold-start, rely more on CB
    - otherwise -> warm user, rely more on MF
    """

    def __init__(
        self,
        alpha_warm: float = DEFAULT_ALPHA_WARM,
        alpha_cold: float = DEFAULT_ALPHA_COLD,
        mf_algo: str = "svd",
    ) -> None:
        self.alpha_warm = alpha_warm
        self.alpha_cold = alpha_cold
        self.mf_algo = mf_algo

        self._cb = None
        self._mf = None
        self._fitted = False

    def fit(self, movies: pd.DataFrame, train_ratings: pd.DataFrame) -> "HybridRecommender":
        """Train both sub-models."""
        print("Training Content-Based model...")
        self._cb = ContentBasedRecommender()
        self._cb.fit(movies)

        print(f"Training {self.mf_algo.upper()} model...")
        self._mf = MatrixFactorizationRecommender(algorithm=self.mf_algo)
        self._mf.fit(train_ratings, all_movie_ids=movies["movie_id"].tolist())

        self._movies = movies
        self._fitted = True
        return self

    def recommend_for_user(
        self,
        user_id: int,
        ratings: pd.DataFrame,
        top_k: int = 10,
        alpha: float | None = None,
    ) -> pd.DataFrame:
        """Return top-K hybrid recommendations for a user.

        Both CB and MF scores are min-max normalized to [0, 1] before
        combining, so neither scale dominates the other.
        """
        self._check_fitted()

        user_ratings = ratings[ratings["user_id"] == user_id]
        n = len(user_ratings)

        if alpha is None:
            alpha = self.alpha_cold if n < MIN_RATINGS_FOR_MF else self.alpha_warm
            mode  = "cold-start" if n < MIN_RATINGS_FOR_MF else "warm"
            print(f"User {user_id} | {n} ratings | {mode} | alpha={alpha:.2f}")

        seen_ids      = set(user_ratings["movie_id"].tolist())
        unseen_movies = self._movies[~self._movies["movie_id"].isin(seen_ids)].copy()

        if unseen_movies.empty:
            return pd.DataFrame()

        cb_norm = self._minmax_norm(self._get_cb_scores(user_id, ratings, unseen_movies))
        mf_norm = self._minmax_norm(self._get_mf_scores(user_id, unseen_movies))
        hybrid  = alpha * cb_norm + (1 - alpha) * mf_norm

        result = unseen_movies[["title", "genres", "year"]].copy().reset_index(drop=True)
        result["cb_score"]     = np.round(cb_norm, 4)
        result["mf_score"]     = np.round(mf_norm, 4)
        result["hybrid_score"] = np.round(hybrid, 4)

        return result.sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)

    def _get_cb_scores(self, user_id: int, ratings: pd.DataFrame, unseen_movies: pd.DataFrame) -> np.ndarray:
        user_ratings = ratings[ratings["user_id"] == user_id]
        liked = user_ratings[user_ratings["rating"] >= 3.5]
        if liked.empty:
            liked = user_ratings

        agg        = np.zeros(len(unseen_movies))
        unseen_ids = unseen_movies["movie_id"].values

        for _, row in liked.iterrows():
            mid = row["movie_id"]
            if mid not in self._cb.id_to_idx:
                continue
            src_idx = self._cb.id_to_idx[mid]
            weight  = max(row["rating"] - 3.5, 0.1)

            for i, umid in enumerate(unseen_ids):
                if umid in self._cb.id_to_idx:
                    agg[i] += weight * self._cb.sim_matrix[src_idx, self._cb.id_to_idx[umid]]

        return agg

    def _get_mf_scores(self, user_id: int, unseen_movies: pd.DataFrame) -> np.ndarray:
        return np.array([
            self._mf._algo.predict(int(user_id), int(mid)).est
            for mid in unseen_movies["movie_id"]
        ])

    @staticmethod
    def _minmax_norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.zeros_like(arr, dtype=float)
        return (arr - lo) / (hi - lo)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")

    def save(self, path: str | None = None) -> str:
        self._check_fitted()
        path = path or os.path.join(MODEL_DIR, "hybrid.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | None = None) -> "HybridRecommender":
        path = path or os.path.join(MODEL_DIR, "hybrid.pkl")
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model


if __name__ == "__main__":
    from src.data_loader import load_movies, load_ratings, train_test_split

    movies  = load_movies()
    ratings = load_ratings()
    train, _ = train_test_split(ratings)

    hybrid = HybridRecommender(mf_algo="svd")
    hybrid.fit(movies, train)
    hybrid.save()

    print("\nHybrid Recommendations for User 1:")
    print(hybrid.recommend_for_user(1, ratings, top_k=10).to_string(index=False))
