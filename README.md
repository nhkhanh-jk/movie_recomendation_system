# Movie Recommendation System

Built on the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset (~1M ratings, 3,883 movies, 6,040 users).

Three approaches are implemented and compared:
- **Content-Based Filtering** — TF-IDF on genres + decade bucketing, Cosine Similarity
- **Matrix Factorization** — SVD, SVD++, NMF via [scikit-surprise](https://surpriselib.com/)
- **Hybrid** — weighted combination of CB and MF with automatic cold-start handling

---

## Project structure

```
movie_recomendation/
├── movie_data/
│   ├── movies.csv
│   ├── ratings.csv
│   └── users.csv
├── src/
│   ├── data_loader.py          # loading, cleaning, temporal train/test split
│   ├── content_based.py        # TF-IDF + Cosine Similarity recommender
│   ├── matrix_factorization.py # SVD / SVD++ / NMF recommender
│   ├── hybrid.py               # weighted hybrid model
│   └── evaluation.py           # RMSE, MAE, Precision@K, Recall@K, NDCG@K
├── models/                     # saved model files (.pkl)
├── main.py                     # CLI entry point
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Download the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) and place the `.csv` files into `movie_data/`.

---

## Usage

**Train all models:**
```bash
python main.py --mode train

# include SVD++ (takes ~5–10 min extra)
python main.py --mode train --full
```

**Evaluate and compare models:**
```bash
python main.py --mode evaluate --n_users 200
```

Sample output:
```
╭───────────────┬────────┬────────┬────────┬────────┬───────────┬─────────╮
│ Model         │  RMSE  │  MAE   │  P@10  │  R@10  │  NDCG@10  │  F1@10  │
├───────────────┼────────┼────────┼────────┼────────┼───────────┼─────────┤
│ Content-Based │   —    │   —    │ 0.011  │ 0.019  │   0.015   │  0.012  │
│ SVD           │ 0.8906 │ 0.6979 │ 0.050  │ 0.030  │   0.057   │  0.034  │
│ Hybrid        │   —    │   —    │ 0.056  │ 0.058  │   0.071   │  0.042  │
╰───────────────┴────────┴────────┴────────┴────────┴───────────┴─────────╯
```

**Get recommendations:**
```bash
# movies similar to a title
python main.py --mode recommend --type item --title "Toy Story (1995)" --top_k 10

# personalized for a user (hybrid by default)
python main.py --mode recommend --type user --user_id 42 --top_k 10

# pick a specific model
python main.py --mode recommend --type user --user_id 42 --model svd --top_k 10
python main.py --mode recommend --type user --user_id 42 --model cb  --top_k 10
```

**Dataset stats:**
```bash
python main.py --mode stats
```

---

## Models

### Content-Based Filtering

Each movie is converted to a TF-IDF vector of its genre tokens plus a decade tag (e.g. `decade_1990`). Cosine similarity is computed between all pairs, producing a 3883×3883 similarity matrix.

For user recommendations, similarity scores are aggregated across all movies the user liked (rating ≥ 3.5), weighted by `(rating - threshold)`, and unseen movies are ranked.

### Matrix Factorization (SVD / SVD++ / NMF)

Learns latent user and item factor vectors that minimize rating prediction error. Training uses an 80/20 temporal split — for each user, the last 20% of ratings by timestamp go to the test set. This mirrors real deployment: train on the past, predict the future.

SVD++ extends SVD by also encoding which items a user has rated (implicit feedback), not just the ratings themselves.

### Hybrid

```
score(u, i) = alpha * cb_score(u, i) + (1 - alpha) * mf_score(u, i)
```

Both scores are min-max normalized to [0, 1] before combining so neither scale dominates.

Alpha is chosen automatically:
- **cold-start user** (< 10 ratings): alpha = 0.85 — lean on CB since MF needs more history
- **warm user** (≥ 10 ratings): alpha = 0.30 — lean on MF for better personalization

---

## Evaluation

Train/test split is temporal per user to avoid data leakage. A movie is "relevant" if the user's true rating ≥ 4.0.

| Metric | Description |
|--------|------------|
| RMSE / MAE | Rating prediction error (MF models only) |
| Precision@K | Fraction of top-K recommendations that are relevant |
| Recall@K | Fraction of relevant items found in top-K |
| NDCG@K | Ranking quality — rewards placing relevant items higher |
| F1@K | Harmonic mean of Precision and Recall |

---

## Stack

- `pandas`, `numpy` — data handling
- `scikit-learn` — TF-IDF, cosine similarity
- `scikit-surprise` — SVD, SVD++, NMF
- `joblib` — model serialization
- `tabulate` — evaluation output formatting

---

## Dataset

MovieLens 1M — GroupLens Research, University of Minnesota.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS 5, 4: 29:1–29:19.
