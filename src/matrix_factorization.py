import os
import time
import numpy as np
import pandas as pd
import joblib

from surprise import SVD, SVDpp, NMF, Dataset, Reader
from surprise.model_selection import cross_validate


MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
RATING_SCALE = (1, 5)


class MatrixFactorizationRecommender:
    """
    Collaborative Filtering via Matrix Factorization (SVD / SVD++ / NMF).

    Wraps scikit-surprise algorithms. Learns latent user and item factors
    from the rating matrix so it can predict how a user would rate unseen movies.

    Args:
        algorithm : 'svd', 'svdpp', or 'nmf'. Default 'svd'.
        n_factors : Number of latent factors. Default 100.
        n_epochs  : Training epochs. Default 20.
        lr_all    : Learning rate (SVD/SVD++ only). Default 0.005.
        reg_all   : Regularization strength. Default 0.02.
    """

    ALGO_MAP = {"svd": SVD, "svdpp": SVDpp, "nmf": NMF}

    def __init__(
        self,
        algorithm: str = "svd",
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
    ) -> None:
        if algorithm not in self.ALGO_MAP:
            raise ValueError(f"algorithm must be one of {list(self.ALGO_MAP)}")

        self.algorithm_name = algorithm
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr_all    = lr_all
        self.reg_all   = reg_all

        self._algo      = None
        self._trainset  = None
        self._movie_ids: list[int] = []
        self._fitted    = False

    @staticmethod
    def _to_surprise_dataset(ratings: pd.DataFrame) -> Dataset:
        reader = Reader(rating_scale=RATING_SCALE)
        return Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)

    def _make_algo(self):
        AlgoClass = self.ALGO_MAP[self.algorithm_name]
        kwargs = dict(n_factors=self.n_factors, n_epochs=self.n_epochs)

        if self.algorithm_name in ("svd", "svdpp"):
            kwargs["lr_all"]  = self.lr_all
            kwargs["reg_all"] = self.reg_all
        else:
            kwargs["reg_pu"] = self.reg_all
            kwargs["reg_qi"] = self.reg_all

        return AlgoClass(**kwargs)

    def fit(
        self,
        train_ratings: pd.DataFrame,
        all_movie_ids: list[int] | None = None,
    ) -> "MatrixFactorizationRecommender":
        """Train on train_ratings. Pass all_movie_ids to enable recommending
        movies that may not appear in the training split."""
        self._movie_ids = all_movie_ids or train_ratings["movie_id"].unique().tolist()

        data     = self._to_surprise_dataset(train_ratings)
        trainset = data.build_full_trainset()
        self._algo = self._make_algo()

        print(f"Training {self.algorithm_name.upper()} on {len(train_ratings):,} ratings...")
        t0 = time.time()
        self._algo.fit(trainset)
        self._trainset = trainset
        self._fitted   = True
        print(f"Done in {time.time() - t0:.1f}s")
        return self

    def cross_validate(self, ratings: pd.DataFrame, cv: int = 5) -> dict[str, float]:
        """Run k-fold CV and return mean RMSE and MAE."""
        data  = self._to_surprise_dataset(ratings)
        algo  = self._make_algo()
        results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=cv, verbose=False)
        rmse = float(np.mean(results["test_rmse"]))
        mae  = float(np.mean(results["test_mae"]))
        print(f"{self.algorithm_name.upper()} {cv}-fold CV — RMSE={rmse:.4f}, MAE={mae:.4f}")
        return {"rmse": rmse, "mae": mae}

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict the rating a user would give a movie.

        Note: IDs must be passed as int (same type used during training from DataFrame).
        """
        self._check_fitted()
        return round(self._algo.predict(int(user_id), int(movie_id)).est, 4)

    def recommend_for_user(
        self,
        user_id: int,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Predict ratings for all unseen movies and return top-K."""
        self._check_fitted()

        seen_ids   = set(ratings[ratings["user_id"] == user_id]["movie_id"].tolist())
        unseen_ids = [mid for mid in self._movie_ids if mid not in seen_ids]

        predictions = [
            (mid, self._algo.predict(int(user_id), int(mid)).est)
            for mid in unseen_ids
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        top = predictions[:top_k]

        top_ids    = [mid for mid, _ in top]
        top_scores = [s for _, s in top]

        id_to_info = movies.set_index("movie_id")[["title", "genres", "year"]]
        result = id_to_info.reindex(top_ids).reset_index()
        result["predicted_rating"] = np.round(top_scores, 4)
        return result[["title", "genres", "year", "predicted_rating"]]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit() before making predictions.")

    def save(self, path: str | None = None) -> str:
        self._check_fitted()
        path = path or os.path.join(MODEL_DIR, f"mf_{self.algorithm_name}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, algorithm: str = "svd", path: str | None = None) -> "MatrixFactorizationRecommender":
        path = path or os.path.join(MODEL_DIR, f"mf_{algorithm}.pkl")
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model


if __name__ == "__main__":
    from src.data_loader import load_movies, load_ratings, train_test_split

    movies  = load_movies()
    ratings = load_ratings()
    train, _ = train_test_split(ratings)
    all_ids  = movies["movie_id"].tolist()

    for algo in ("svd", "svdpp", "nmf"):
        mf = MatrixFactorizationRecommender(algorithm=algo, n_epochs=20)
        mf.fit(train, all_movie_ids=all_ids)
        mf.save()

    svd = MatrixFactorizationRecommender.load("svd")
    svd._movies = movies
    print("\nSVD Recommendations for User 1:")
    print(svd.recommend_for_user(1, ratings, movies, top_k=10).to_string(index=False))
