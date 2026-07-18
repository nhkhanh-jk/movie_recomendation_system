from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import os, sys
import pandas as pd

sys.path.insert(0, "/app")

from src.content_based import ContentBasedRecommender
from src.matrix_factorization import MatrixFactorizationRecommender
from src.hybrid import HybridRecommender
from src.data_loader import load_all

models = {}   # cache models trong RAM
data   = {}   # cache movies + ratings

@asynccontextmanager
async def lifespan(app):
    # Chạy 1 lần khi API start
    data["movies"], data["ratings"], _ = load_all()
    models["cb"]     = ContentBasedRecommender.load()
    models["svd"]    = MatrixFactorizationRecommender.load("svd")
    models["svd"]._movies = data["movies"]
    models["hybrid"] = HybridRecommender.load()
    models["hybrid"]._movies = data["movies"]
    print("✓ Models loaded!")
    yield
    models.clear()

app = FastAPI(title="Movie Recommendation API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return{
        "status": "healthy",
        "models_loaded": list(models.keys())
    }

@app.get("/recommend/user/{user_id}")
def recommend_for_user(user_id: int, model: str = "hybrid", top_k: int = 10):
    model = model.lower()
    if model not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' không hỗ trợ. Hãy chọn trong: {list(models.keys())}"
        )
    try:
        rec_model = models[model]
        if model == "svd":
            df_rec = rec_model.recommend_for_user(
                user_id, data["ratings"], data["movies"], top_k=top_k
            )
        else:
            df_rec = rec_model.recommend_for_user(user_id, data["ratings"], top_k=top_k)
        return df_rec.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"lỗi: {str(e)}"
        )

@app.get("/recommend/similar")
def recommend_similar(title: str, top_k: int=10):
    try:
        cb_model = models["cb"]
        df_rec = cb_model.recommend_similar_movies(title, top_k=top_k)
        return df_rec.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code = 500,
        detail=f"Lỗi: {str(e)}")