import os
import pandas as pd
from sqlalchemy import create_engine

DB_USER = "mluser"
DB_PASSWORD = "mlpassword"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "movielens_oltp"

CONNECTION_STRING = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONNECTION_STRING)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "movie_data")
MOVIES_PATH  = os.path.join(DATA_DIR, "movies.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
USERS_PATH   = os.path.join(DATA_DIR, "users.csv")

def import_movies(engine):
    print("Importing movies...")

    df = pd.read_csv(MOVIES_PATH, sep="\t", encoding="latin1")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
    df = df[["movie_id", "title", "genres"]]
    df.to_sql(
        name="movie_raw",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000,
        method="multi"
    )
    print(f"✓ Done: {len(df)} movies imported")

def import_ratings(engine):
    print("Importing ratings (1M rows, may take a while)...")

    df = pd.read_csv(RATINGS_PATH, sep="\t", encoding="latin1")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
    df = df[["user_id", "movie_id", "rating", "timestamp"]]

    df.to_sql(
        name="ratings_raw",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000,
        method="multi"
    )
    print(f"✓ Done: {len(df)} ratings imported")


def import_users(engine):
    print("Importing users...")

    df = pd.read_csv(USERS_PATH, sep="\t", encoding="latin1")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
    df = df[["user_id", "gender", "age", "occupation", "zipcode"]]

    df.to_sql(
        name="users_raw",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000,
        method="multi"
    )
    print(f"✓ Done: {len(df)} users imported")


if __name__ == "__main__":
    import_movies(engine)
    import_ratings(engine)
    import_users(engine)
    print("\nAll data imported successfully!")
