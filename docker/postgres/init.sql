-- PostgreSQL OLAP schema

CREATE TABLE IF NOT EXISTS movie_features (
    movie_id   INTEGER PRIMARY KEY,
    title      VARCHAR(500) NOT NULL,
    genres     VARCHAR(500),       
    year       INTEGER,             
    decade     INTEGER             
);


CREATE TABLE IF NOT EXISTS ratings_clean (
    id        SERIAL PRIMARY KEY,
    user_id   INTEGER NOT NULL,
    movie_id  INTEGER NOT NULL,
    rating    FLOAT NOT NULL,
    timestamp BIGINT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ratings_user  ON ratings_clean(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings_clean(movie_id);

CREATE TABLE IF NOT EXISTS users_clean (
    user_id    INTEGER PRIMARY KEY,
    gender     VARCHAR(1),
    age        INTEGER,
    occupation INTEGER,
    zip_code   VARCHAR(20)
);
