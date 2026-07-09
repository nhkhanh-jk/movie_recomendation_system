import os
import pandas as pd
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "movie_data")

MOVIES_PATH  = os.path.join(DATA_DIR, "movies.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
USERS_PATH   = os.path.join(DATA_DIR, "users.csv")


def load_movies(path: str = MOVIES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="latin1")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])

    # join genres into a single space-separated string for TF-IDF
    df["genres"] = (
        df["genres"]
        .fillna("Unknown")
        .str.replace("|", " ", regex=False)
        .str.replace("-", "", regex=False)
    )

    # extract release year from title, e.g. "Toy Story (1995)" -> 1995
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype(float)

    return df.reset_index(drop=True)


def load_ratings(path: str = RATINGS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="latin1")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])

    keep = ["user_id", "movie_id", "rating", "timestamp"]
    df = df[[c for c in keep if c in df.columns]]

    return df.reset_index(drop=True)


def load_users(path: str = USERS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="latin1")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
    return df.reset_index(drop=True)


def load_all(
    movies_path: str  = MOVIES_PATH,
    ratings_path: str = RATINGS_PATH,
    users_path: str   = USERS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies  = load_movies(movies_path)
    ratings = load_ratings(ratings_path)
    users   = load_users(users_path)
    return movies, ratings, users


def train_test_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    min_ratings_per_user: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal train/test split.

    For each user, sort ratings by timestamp and hold out the last
    `test_ratio` fraction as test. This avoids data leakage — the model
    never sees future interactions during training.

    Users with fewer than `min_ratings_per_user` ratings go entirely to train.
    """
    ratings = ratings.sort_values(["user_id", "timestamp"])

    train_rows, test_rows = [], []

    for _, group in ratings.groupby("user_id"):
        n = len(group)
        if n < min_ratings_per_user:
            train_rows.append(group)
            continue

        split_idx = int(n * (1 - test_ratio))
        train_rows.append(group.iloc[:split_idx])
        test_rows.append(group.iloc[split_idx:])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)

    print(f"Train: {len(train_df):,} | Test: {len(test_df):,} "
          f"({len(test_df)/len(ratings):.1%})")

    return train_df, test_df


def dataset_stats(movies: pd.DataFrame, ratings: pd.DataFrame, users: pd.DataFrame) -> None:
    print("=" * 45)
    print("  MovieLens 1M — Dataset Summary")
    print("=" * 45)
    print(f"  Movies  : {len(movies):,}")
    print(f"  Users   : {len(users):,}")
    print(f"  Ratings : {len(ratings):,}")
    print(f"  Sparsity: {1 - len(ratings) / (len(movies) * len(users)):.2%}")
    print(f"  Rating range : {ratings['rating'].min()} – {ratings['rating'].max()}")
    print(f"  Avg rating   : {ratings['rating'].mean():.2f}")
    top_genres = movies["genres"].str.split().explode().value_counts()
    print(f"  Top genres   : {', '.join(top_genres.head(5).index.tolist())}")
    print("=" * 45)


if __name__ == "__main__":
    movies, ratings, users = load_all()
    dataset_stats(movies, ratings, users)
    train_test_split(ratings)
