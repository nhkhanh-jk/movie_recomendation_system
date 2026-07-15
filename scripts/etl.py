import os
import pandas as pd
from sqlalchemy import create_engine


MYSQL_URL    = "mysql+pymysql://mluser:mlpassword@localhost:3306/movielens_oltp"
POSTGRES_URL = "postgresql+psycopg2://mluser:mlpassword@localhost:5432/movielens_olap"

mysql_engine    = create_engine(MYSQL_URL)
postgres_engine = create_engine(POSTGRES_URL)


# Transform movies 
def etl_movies():
    print("ETL movies...")

    df = pd.read_sql("SELECT * FROM movie_raw", mysql_engine)

    # Genres: "Action|Crime|Thriller" → "Action Crime Thriller"
    df["genres"] = ( 
        df["genres"]
        .fillna("Unknown")
        .str.replace("|", " ", regex=False)
        .str.replace("-", "", regex=False)
    )

    # Trích năm từ title: "Toy Story (1995)" → 1995
    df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype(float)

    # Tính decade: 1995 → 1990
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")  # Int64 hỗ trợ NaN

    df[["movie_id", "title", "genres", "year", "decade"]].to_sql(
        name="movie_features",
        con=postgres_engine,
        if_exists="append",
        index=False,
        chunksize=1000,
        method="multi"
    )
    print(f"✓ Done: {len(df)} movies → PostgreSQL")



def etl_ratings():
    print("ETL ratings (1M rows)...")

    df = pd.read_sql("SELECT user_id, movie_id, rating, timestamp FROM ratings_raw", mysql_engine)

    df.to_sql(
        name="ratings_clean",
        con=postgres_engine,
        if_exists="append",
        index=False,
        chunksize=5000,
        method="multi"
    )
    print(f"✓ Done: {len(df)} ratings → PostgreSQL")



def etl_users():
    print("ETL users...")

    df = pd.read_sql("SELECT * FROM users_raw", mysql_engine)

    df.to_sql(
        name="users_clean",
        con=postgres_engine,
        if_exists="append",
        index=False,
        chunksize=1000,
        method="multi"
    )
    print(f"✓ Done: {len(df)} users → PostgreSQL")


if __name__ == "__main__":
    etl_movies()
    etl_ratings()
    etl_users()
    print("\nETL completed!")
