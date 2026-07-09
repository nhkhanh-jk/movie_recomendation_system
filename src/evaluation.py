import numpy as np
import pandas as pd
from tabulate import tabulate


RELEVANCE_THRESHOLD = 4.0
TOP_K_VALUES = [5, 10, 20]


class Evaluator:
    """
    Evaluates recommendation models using standard IR metrics.

    Rating metrics (MF models): RMSE, MAE
    Ranking metrics (all models): Precision@K, Recall@K, NDCG@K, F1@K

    A movie is "relevant" if the user's true rating >= RELEVANCE_THRESHOLD.

    Important: train_ratings is used to determine which movies are already
    seen by the user, so test-set movies remain as recommendation candidates.
    This avoids artificially zeroing out the items we want to evaluate on.
    """

    def __init__(
        self,
        test_ratings: pd.DataFrame,
        train_ratings: pd.DataFrame,
        movies: pd.DataFrame,
        k_values: list[int] = None,
    ) -> None:
        self.test_ratings  = test_ratings
        self.train_ratings = train_ratings
        self.movies        = movies
        self.k_values      = k_values or TOP_K_VALUES

        # ground truth per user: movies they actually liked in the test split
        self._user_relevant: dict[int, set[int]] = (
            test_ratings[test_ratings["rating"] >= RELEVANCE_THRESHOLD]
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )
        self._results: list[dict] = []

    def rating_metrics(self, predicted: list[float], actual: list[float]) -> dict:
        pred = np.array(predicted)
        true = np.array(actual)
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        mae  = float(np.mean(np.abs(pred - true)))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4)}

    def evaluate_mf_ratings(self, mf_model, label: str) -> dict:
        """Compute RMSE and MAE on the held-out test set."""
        preds, actuals = [], []
        for _, row in self.test_ratings.iterrows():
            preds.append(mf_model.predict_rating(int(row["user_id"]), int(row["movie_id"])))
            actuals.append(row["rating"])

        metrics = self.rating_metrics(preds, actuals)
        metrics["model"] = label
        print(f"{label:12s} — RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
        return metrics

    @staticmethod
    def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
        hits = sum(1 for mid in recommended[:k] if mid in relevant)
        return hits / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
        if not relevant:
            return 0.0
        hits = sum(1 for mid in recommended[:k] if mid in relevant)
        return hits / len(relevant)

    @staticmethod
    def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
        def dcg(items):
            return sum(
                (1.0 if mid in relevant else 0.0) / np.log2(rank + 2)
                for rank, mid in enumerate(items)
            )
        actual_dcg = dcg(recommended[:k])
        ideal_dcg  = dcg(list(relevant)[:k])
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    @staticmethod
    def f1_at_k(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def evaluate_recommender(
        self,
        recommender,
        label: str,
        n_users: int = 200,
        top_k_max: int = 20,
    ) -> dict:
        """Evaluate ranking quality on a sample of users.

        Recommendations are generated using train_ratings as the "seen" set,
        so items from the test split are still available as candidates.
        """
        train_ratings = self.train_ratings

        eval_users = [
            uid for uid in self._user_relevant
            if uid in train_ratings["user_id"].values
        ]
        if n_users and len(eval_users) > n_users:
            rng = np.random.default_rng(42)
            eval_users = rng.choice(eval_users, size=n_users, replace=False).tolist()

        metrics_per_k: dict[int, list] = {k: [] for k in self.k_values}

        for user_id in eval_users:
            try:
                recs = self._get_recommendations(recommender, user_id, train_ratings, top_k_max)
            except Exception:
                continue

            relevant = self._user_relevant.get(user_id, set())
            if not relevant:
                continue

            for k in self.k_values:
                p  = self.precision_at_k(recs, relevant, k)
                r  = self.recall_at_k(recs, relevant, k)
                nd = self.ndcg_at_k(recs, relevant, k)
                metrics_per_k[k].append({"p": p, "r": r, "ndcg": nd, "f1": self.f1_at_k(p, r)})

        result = {"model": label, "rmse": "—", "mae": "—"}
        for k in self.k_values:
            data = metrics_per_k[k]
            if data:
                result[f"P@{k}"]    = round(np.mean([d["p"]    for d in data]), 4)
                result[f"R@{k}"]    = round(np.mean([d["r"]    for d in data]), 4)
                result[f"NDCG@{k}"] = round(np.mean([d["ndcg"] for d in data]), 4)
                result[f"F1@{k}"]   = round(np.mean([d["f1"]   for d in data]), 4)

        self._results.append(result)
        print(f"{label:12s} — " + " | ".join(
            f"P@{k}={result.get(f'P@{k}', 0):.4f}" for k in self.k_values
        ))
        return result

    @staticmethod
    def _get_recommendations(recommender, user_id: int, ratings: pd.DataFrame, top_k: int) -> list[int]:
        from src.content_based import ContentBasedRecommender
        from src.matrix_factorization import MatrixFactorizationRecommender
        from src.hybrid import HybridRecommender

        if isinstance(recommender, ContentBasedRecommender):
            df = recommender.recommend_for_user(user_id, ratings, top_k=top_k)
        elif isinstance(recommender, MatrixFactorizationRecommender):
            df = recommender.recommend_for_user(user_id, ratings, recommender._movies, top_k=top_k)
        elif isinstance(recommender, HybridRecommender):
            df = recommender.recommend_for_user(user_id, ratings, top_k=top_k)
        else:
            raise TypeError(f"Unknown recommender type: {type(recommender)}")

        if df is None or df.empty:
            return []

        if "movie_id" in df.columns:
            return df["movie_id"].tolist()

        title_to_id = recommender._movies.set_index("title")["movie_id"].to_dict()
        return [title_to_id[t] for t in df["title"] if t in title_to_id]

    def print_report(self, k: int = 10) -> None:
        if not self._results:
            print("No results yet. Run evaluate_recommender() first.")
            return

        headers = ["Model", "RMSE", "MAE", f"P@{k}", f"R@{k}", f"NDCG@{k}", f"F1@{k}"]
        rows = [
            [
                r.get("model"),
                r.get("rmse", "—"),
                r.get("mae", "—"),
                r.get(f"P@{k}", "—"),
                r.get(f"R@{k}", "—"),
                r.get(f"NDCG@{k}", "—"),
                r.get(f"F1@{k}", "—"),
            ]
            for r in self._results
        ]

        print("\n" + "=" * 65)
        print("  Results")
        print("=" * 65)
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        print(f"\n  Relevance threshold: rating >= {RELEVANCE_THRESHOLD}")
        print(f"  Dataset: MovieLens 1M\n")

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._results)
