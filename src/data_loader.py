import pandas as pd
from sqlalchemy import create_engine
import os

POSTGRES_URL = (
    f"postgresql+psycopg2://mluser:mlpassword@" f"{os.getenv('POSTGRES_HOST', 'localhost')}:5432/movielens_olap"
)

def get_engine():
    return create_engine(POSTGRES_URL)

def load_movies() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM movie_features", get_engine())

def load_ratings() -> pd.DataFrame:
    return pd.read_sql("SELECT user_id, movie_id, rating, timestamp FROM ratings_clean", get_engine())

def load_users() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM users_clean", get_engine())


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    movies  = load_movies()
    ratings = load_ratings()
    users   = load_users()
    return movies, ratings, users


def train_test_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    min_ratings_per_user: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
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
