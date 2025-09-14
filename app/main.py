from fastapi import FastAPI
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.metrics import DistanceMetric
from app.functions import returnSearchResultIndexes
import os

# define model info
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ✅ load from Hugging Face hub
VIDEO_INDEX_PATH = os.getenv("VIDEO_INDEX_PATH", "app/data/video-index.parquet")

# load model dynamically at runtime (downloads from Hugging Face on first run, cached afterwards)
model = SentenceTransformer(MODEL_NAME)

# load video index
df = pl.scan_parquet(VIDEO_INDEX_PATH)

# create distance metric object
dist = DistanceMetric.get_metric("manhattan")

# create FastAPI app
app = FastAPI()

@app.get("/")
def health_check():
    return {"health_check": "OK"}

@app.get("/info")
def info():
    return {
        "name": "yt-search",
        "description": "Search API for exploring indexed YouTube videos using semantic similarity."
    }

@app.get("/search")
def search(query: str):
    idx_result = returnSearchResultIndexes(query, df, model, dist)
    return df.select(["title", "video_id"]).collect()[idx_result].to_dict(as_series=False)
