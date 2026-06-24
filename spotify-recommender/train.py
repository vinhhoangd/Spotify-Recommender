"""
Offline training pipeline.

Trains two models and saves artifacts to ./models:

1. Collaborative filtering — ALS matrix factorization (implicit feedback)
   on the Last.fm 360K user-artist play-count dataset.
2. Content-based — scaled audio-feature matrix over the 114k-track
   Spotify Tracks dataset for cosine-similarity track recommendations.

Run once:  python train.py
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

import implicit
from implicit.nearest_neighbours import bm25_weight
from implicit.evaluation import train_test_split, ranking_metrics_at_k
from implicit.datasets.lastfm import get_lastfm
from sklearn.preprocessing import MinMaxScaler

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

CONTENT_CSV_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]


# ── Collaborative: ALS on Last.fm ─────────────────────────────────────────────

def train_collaborative(factors=64, iterations=15, regularization=0.05):
    print("Loading Last.fm 360K dataset (auto-downloads on first run)…")
    artists, users, plays = get_lastfm()  # plays: item(artist) x user CSR

    # Weight raw play counts with BM25 to dampen power users / popular artists.
    plays = bm25_weight(plays, K1=100, B=0.8).tocsr()
    user_plays = plays.T.tocsr()  # ALS expects user x item

    print("Splitting train/test for offline evaluation…")
    train, test = train_test_split(user_plays, train_percentage=0.9)

    print(f"Training ALS (factors={factors}, iterations={iterations})…")
    model = implicit.als.AlternatingLeastSquares(
        factors=factors, iterations=iterations,
        regularization=regularization, use_gpu=False,
    )
    model.fit(train)

    print("Evaluating ranking metrics @ 10…")
    metrics = ranking_metrics_at_k(model, train, test, K=10, num_threads=1)
    print("  ", metrics)

    # Refit on full data for the served model.
    print("Refitting on full dataset for serving…")
    final = implicit.als.AlternatingLeastSquares(
        factors=factors, iterations=iterations,
        regularization=regularization, use_gpu=False,
    )
    final.fit(user_plays)

    np.save(MODELS_DIR / "artist_factors.npy", final.item_factors.astype(np.float32))
    np.save(MODELS_DIR / "artist_names.npy", artists)
    with open(MODELS_DIR / "collab_meta.json", "w") as f:
        json.dump({"factors": factors, "iterations": iterations,
                   "regularization": regularization,
                   "n_artists": int(len(artists)), "n_users": int(user_plays.shape[0]),
                   "metrics_at_10": {k: float(v) for k, v in metrics.items()}}, f, indent=2)
    print(f"Saved collaborative artifacts ({len(artists)} artists).")
    return metrics


# ── Content: audio-feature matrix on Spotify Tracks ───────────────────────────

def train_content():
    print("Loading Spotify Tracks dataset…")
    df = pd.read_csv(CONTENT_CSV_URL)
    df = df.dropna(subset=["track_name", "artists"]).drop_duplicates("track_id").reset_index(drop=True)

    scaler = MinMaxScaler()
    feature_matrix = scaler.fit_transform(df[AUDIO_FEATURES].fillna(0).values).astype(np.float32)

    np.save(MODELS_DIR / "content_features.npy", feature_matrix)
    meta = df[["track_id", "track_name", "artists", "album_name",
               "track_genre", "popularity"]].copy()
    meta.to_parquet(MODELS_DIR / "content_meta.parquet")
    with open(MODELS_DIR / "content_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved content artifacts ({len(df)} tracks, {feature_matrix.shape[1]} features).")


if __name__ == "__main__":
    metrics = train_collaborative()
    train_content()
    print("\nDone. Artifacts in ./models")
    print("ALS ranking metrics @10:", metrics)
