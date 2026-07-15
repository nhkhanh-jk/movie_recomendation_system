-- MySQL OLTP schema 

CREATE DATABASE IF NOT EXISTS movielens_oltp;
USE movielens_oltp;

-- Movie Table
CREATE TABLE IF NOT EXISTS movie_raw (
    movie_id INT PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    genres VARCHAR(500)
);

-- Ratings Table
CREATE TABLE IF NOT EXISTS ratings_raw (
    id        INT AUTO_INCREMENT PRIMARY KEY,
    user_id   INT NOT NULL,
    movie_id  INT NOT NULL,
    rating    FLOAT NOT NULL,
    timestamp BIGINT NOT NULL,      
    INDEX idx_user_id (user_id), 
    INDEX idx_movie_id (movie_id)
);

-- Users Table
CREATE TABLE IF NOT EXISTS users_raw (
    user_id    INT PRIMARY KEY,
    gender     VARCHAR(1),          
    age        INT,
    occupation INT,
    zip_code   VARCHAR(20)
);
