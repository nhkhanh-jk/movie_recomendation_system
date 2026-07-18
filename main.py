import mlflow
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_all, train_test_split, dataset_stats
from src.content_based import ContentBasedRecommender
from src.matrix_factorization import MatrixFactorizationRecommender
from src.hybrid import HybridRecommender
from src.evaluation import Evaluator

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def train(args) -> None:
    movies, ratings, _ = load_all()
    train_ratings, test_ratings = train_test_split(ratings)
    all_ids                     = movies["movie_id"].tolist()

    # Đảm bảo MLflow luôn ghi vào file mlflow.db tại thư mục gốc của dự án
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    mlflow.set_experiment("movie-recommendation")

    # ── Content-Based (không có hyperparams để log) ─────────────────
    print("\n[1/4] Content-Based...")
    cb = ContentBasedRecommender()
    cb.fit(movies)
    cb.save()

    # ── SVD ─────────────────────────────────────────────────────────
    print("\n[2/4] SVD...")
    with mlflow.start_run(run_name="SVD"):
        n_epochs  = 20
        n_factors = 100
        mlflow.log_param("algorithm", "svd")
        mlflow.log_param("n_epochs",  n_epochs)
        mlflow.log_param("n_factors", n_factors)

        svd = MatrixFactorizationRecommender(
            algorithm="svd", n_epochs=n_epochs, n_factors=n_factors
        )
        svd.fit(train_ratings, all_ids)
        svd._movies = movies
        svd.save()

        ev = Evaluator(test_ratings, train_ratings, movies)
        mf_metrics = ev.evaluate_mf_ratings(svd, label="SVD")
        mlflow.log_metric("rmse", round(mf_metrics["rmse"], 4))
        mlflow.log_metric("mae",  round(mf_metrics["mae"],  4))

    # ── SVD++ (optional) ────────────────────────────────────────────
    if args.full:
        print("\n[3/4] SVD++...")
        with mlflow.start_run(run_name="SVD++"):
            mlflow.log_param("algorithm", "svdpp")
            mlflow.log_param("n_epochs",  20)

            svdpp = MatrixFactorizationRecommender(algorithm="svdpp", n_epochs=20)
            svdpp.fit(train_ratings, all_ids)
            svdpp._movies = movies
            svdpp.save()

            ev = Evaluator(test_ratings, train_ratings, movies)
            mf_metrics = ev.evaluate_mf_ratings(svdpp, label="SVD++")
            mlflow.log_metric("rmse", round(mf_metrics["rmse"], 4))
            mlflow.log_metric("mae",  round(mf_metrics["mae"],  4))
    else:
        print("\n[3/4] Skipping SVD++ (pass --full to include it)")

    # ── Hybrid ──────────────────────────────────────────────────────
    print("\n[4/4] Hybrid (CB + SVD)...")
    with mlflow.start_run(run_name="Hybrid"):
        alpha_warm = 0.3
        alpha_cold = 0.85
        mlflow.log_param("mf_algo",    "svd")
        mlflow.log_param("alpha_warm", alpha_warm)
        mlflow.log_param("alpha_cold", alpha_cold)

        hybrid = HybridRecommender(mf_algo="svd")
        hybrid.fit(movies, train_ratings)
        hybrid.save()

    print("\nDone. Models saved to models/\n")


def evaluate(args) -> None:
    movies, ratings, _ = load_all()
    train_ratings, test_ratings = train_test_split(ratings)

    ev = Evaluator(test_ratings, train_ratings, movies, k_values=[5, 10, 20])

    try:
        cb = ContentBasedRecommender.load()
        cb._movies = movies
        ev.evaluate_recommender(cb, label="Content-Based", n_users=args.n_users)
    except FileNotFoundError:
        print("Content-Based model not found. Run --mode train first.")

    try:
        svd = MatrixFactorizationRecommender.load("svd")
        svd._movies = movies
        rmse_m = ev.evaluate_mf_ratings(svd, label="SVD")
        ev.evaluate_recommender(svd, label="SVD", n_users=args.n_users)
        if ev._results:
            ev._results[-1].update(rmse_m)
    except FileNotFoundError:
        print("SVD model not found. Run --mode train first.")

    svdpp_path = os.path.join(MODEL_DIR, "mf_svdpp.pkl")
    if os.path.exists(svdpp_path):
        svdpp = MatrixFactorizationRecommender.load("svdpp")
        svdpp._movies = movies
        rmse_m = ev.evaluate_mf_ratings(svdpp, label="SVD++")
        ev.evaluate_recommender(svdpp, label="SVD++", n_users=args.n_users)
        if ev._results:
            ev._results[-1].update(rmse_m)

    try:
        hybrid = HybridRecommender.load()
        hybrid._movies = movies
        ev.evaluate_recommender(hybrid, label="Hybrid", n_users=args.n_users)
    except FileNotFoundError:
        print("Hybrid model not found. Run --mode train first.")

    ev.print_report(k=10)


def recommend(args) -> None:
    movies, ratings, _ = load_all()

    if args.type == "item":
        if not args.title:
            print("--title is required for --type item")
            sys.exit(1)

        print(f"\nMovies similar to '{args.title}':\n")
        try:
            cb = ContentBasedRecommender.load()
        except FileNotFoundError:
            cb = ContentBasedRecommender()
            cb.fit(movies)
            cb.save()

        print(cb.recommend_similar_movies(args.title, top_k=args.top_k).to_string(index=False))

    elif args.type == "user":
        if args.user_id is None:
            print("--user_id is required for --type user")
            sys.exit(1)

        user_id = args.user_id
        model   = args.model.lower()
        print(f"\nRecommendations for User {user_id} [{model.upper()}]:\n")

        if model in ("cb", "content-based"):
            try:
                rec = ContentBasedRecommender.load()
            except FileNotFoundError:
                rec = ContentBasedRecommender()
                rec.fit(movies)
                rec.save()
            print(rec.recommend_for_user(user_id, ratings, top_k=args.top_k).to_string(index=False))

        elif model in ("svd", "svdpp", "nmf"):
            try:
                rec = MatrixFactorizationRecommender.load(model)
            except FileNotFoundError:
                train_r, _ = train_test_split(ratings)
                rec = MatrixFactorizationRecommender(algorithm=model)
                rec.fit(train_r, movies["movie_id"].tolist())
                rec.save()
            rec._movies = movies
            print(rec.recommend_for_user(user_id, ratings, movies, top_k=args.top_k).to_string(index=False))

        elif model == "hybrid":
            try:
                rec = HybridRecommender.load()
                rec._movies = movies
            except FileNotFoundError:
                train_r, _ = train_test_split(ratings)
                rec = HybridRecommender(mf_algo="svd")
                rec.fit(movies, train_r)
                rec.save()
                rec._movies = movies
            print(rec.recommend_for_user(user_id, ratings, top_k=args.top_k).to_string(index=False))

        else:
            print(f"Unknown model '{model}'. Choose from: cb, svd, svdpp, nmf, hybrid")
            sys.exit(1)


def stats(args) -> None:
    movies, ratings, users = load_all()
    dataset_stats(movies, ratings, users)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Movie Recommendation System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "recommend", "stats"],
        required=True,
    )
    parser.add_argument("--full",    action="store_true", help="Also train SVD++ (slow).")
    parser.add_argument("--type",    choices=["item", "user"], default="user")
    parser.add_argument("--title",   type=str, default=None)
    parser.add_argument("--user_id", type=int, default=None)
    parser.add_argument("--model",   type=str, default="hybrid",
                        help="cb | svd | svdpp | nmf | hybrid")
    parser.add_argument("--top_k",   type=int, default=10)
    parser.add_argument("--n_users", type=int, default=200,
                        help="Users to sample during evaluation")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    {"train": train, "evaluate": evaluate, "recommend": recommend, "stats": stats}[args.mode](args)
