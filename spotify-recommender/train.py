"""
Offline training pipeline.

Trains two models and saves artifacts to ./models:

1. Collaborative filtering — ALS matrix factorization (implicit feedback) on the
   Spotify Million Playlist Dataset (MPD, 2018). Playlists are treated as "users"
   and artists as "items"; an artist's weight in a playlist is how many of its
   tracks appear there. Covers ~mainstream artists through 2018 (incl. The Weeknd).
2. Content-based — scaled audio-feature matrix over the ~114k-track Spotify Tracks
   dataset for cosine-similarity track recommendations.

Run:
    python train.py                 # train both (default 50 MPD slices = 50k playlists)
    python train.py --slices 80     # more playlists -> better long-tail coverage
    python train.py --content-only  # skip collaborative
    python train.py --collab-only   # skip content
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from scipy.sparse import csr_matrix

import implicit
from implicit.nearest_neighbours import bm25_weight
from implicit.evaluation import train_test_split, ranking_metrics_at_k
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
DATA_DIR = BASE / "data" / "mpd"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

MPD_REPO = "jaxliu/Spotify_Million_Playlist_Dataset_Challenge"
MPD_BASE = f"https://huggingface.co/datasets/{MPD_REPO}/resolve/main"
MPD_MANIFEST = f"https://huggingface.co/api/datasets/{MPD_REPO}"
CONTENT_PARQUET_URL = ("https://huggingface.co/datasets/kevinanjalo/"
                       "spotify_audio_features/resolve/main/data/"
                       "spotify_audio_features_0.parquet")
CONTENT_TOP_N = 500_000  # keep the most popular tracks (search speed + memory)
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]


# ── MPD download + parse ──────────────────────────────────────────────────────

def list_available_slices() -> list[str]:
    """Return the slice file paths the mirror actually hosts (not all are present)."""
    d = requests.get(MPD_MANIFEST, timeout=60).json()
    files = [s["rfilename"] for s in d.get("siblings", [])
             if s["rfilename"].startswith("data/mpd.slice")]
    # Sort numerically by the starting index so subsets are deterministic.
    return sorted(files, key=lambda f: int(f.split(".")[2].split("-")[0]))


def download_slices(n_slices: int) -> list[Path]:
    available = list_available_slices()
    if not available:
        raise RuntimeError("No MPD slice files found on the mirror.")
    chosen = available[:min(n_slices, len(available))]
    if n_slices > len(available):
        print(f"  note: mirror hosts {len(available)} slices; using all of them.")
    paths = []
    for i, rpath in enumerate(chosen):
        fname = rpath.split("/")[-1]
        dest = DATA_DIR / fname
        if not dest.exists():
            print(f"  downloading {fname} ({i + 1}/{len(chosen)})…")
            r = requests.get(f"{MPD_BASE}/{rpath}", timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
        paths.append(dest)
    return paths


def build_interactions(paths: list[Path]):
    """Returns (artist_names, user_item CSR[playlist x artist] of weighted counts)."""
    artist_idx: dict[str, int] = {}
    artist_names: list[str] = []
    rows, cols, data = [], [], []
    playlist_id = 0

    for p in paths:
        slc = json.loads(p.read_text())
        for pl in slc["playlists"]:
            counts = defaultdict(int)
            for t in pl["tracks"]:
                counts[t["artist_name"]] += 1
            for name, c in counts.items():
                if name not in artist_idx:
                    artist_idx[name] = len(artist_names)
                    artist_names.append(name)
                rows.append(playlist_id)
                cols.append(artist_idx[name])
                data.append(c)
            playlist_id += 1

    mat = csr_matrix((data, (rows, cols)), shape=(playlist_id, len(artist_names)))
    return np.array(artist_names, dtype=object), mat


# ── Collaborative: ALS on MPD ─────────────────────────────────────────────────

def train_collaborative(n_slices=50, factors=96, iterations=20, regularization=0.05,
                        min_artist_playlists=3):
    print(f"Downloading {n_slices} MPD slices ({n_slices * 1000:,} playlists)…")
    paths = download_slices(n_slices)
    print("Building playlist × artist interaction matrix…")
    artist_names, user_item = build_interactions(paths)
    print(f"  {user_item.shape[0]:,} playlists × {user_item.shape[1]:,} artists, "
          f"{user_item.nnz:,} interactions")

    # Drop ultra-rare artists (appear in < min_artist_playlists playlists): too
    # little signal to learn a meaningful vector, and they bloat the index.
    artist_pl_counts = np.asarray((user_item > 0).sum(axis=0)).ravel()
    keep = artist_pl_counts >= min_artist_playlists
    user_item = user_item[:, keep].tocsr()
    artist_names = artist_names[keep]
    print(f"  kept {keep.sum():,} artists appearing in ≥{min_artist_playlists} playlists")

    # BM25-weight to dampen power playlists / ubiquitous artists.
    weighted = bm25_weight(user_item.T, K1=100, B=0.8).T.tocsr()

    print("Splitting train/test for offline evaluation…")
    train, test = train_test_split(weighted, train_percentage=0.9)

    print(f"Training ALS (factors={factors}, iterations={iterations})…")
    model = implicit.als.AlternatingLeastSquares(
        factors=factors, iterations=iterations,
        regularization=regularization, use_gpu=False)
    model.fit(train)

    print("Evaluating ranking metrics @10…")
    metrics = ranking_metrics_at_k(model, train, test, K=10, num_threads=1)
    print("  ", metrics)

    print("Refitting on full data for serving…")
    final = implicit.als.AlternatingLeastSquares(
        factors=factors, iterations=iterations,
        regularization=regularization, use_gpu=False)
    final.fit(weighted)

    # final.item_factors is indexed by artist column.
    np.save(MODELS_DIR / "artist_factors.npy", final.item_factors.astype(np.float32))
    np.save(MODELS_DIR / "artist_names.npy", artist_names)
    with open(MODELS_DIR / "collab_meta.json", "w") as f:
        json.dump({"source": "Spotify Million Playlist Dataset (2018)",
                   "n_slices": n_slices, "factors": factors, "iterations": iterations,
                   "regularization": regularization,
                   "n_artists": int(len(artist_names)),
                   "n_users": int(user_item.shape[0]),
                   "min_artist_playlists": min_artist_playlists,
                   "metrics_at_10": {k: float(v) for k, v in metrics.items()}}, f, indent=2)
    print(f"Saved collaborative artifacts ({len(artist_names):,} artists).")
    return metrics


# ── Content: audio-feature matrix on Spotify Tracks ───────────────────────────

def train_content():
    cols = ["id", "name", "popularity", "null_response"] + AUDIO_FEATURES
    dest = DATA_DIR.parent / "content" / "spotify_audio_features_0.parquet"
    dest.parent.mkdir(exist_ok=True, parents=True)
    if not dest.exists():
        print("Downloading Spotify audio-features dataset (~1.2GB)…")
        with requests.get(CONTENT_PARQUET_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

    print("Loading audio features (reading needed columns only)…")
    df = pd.read_parquet(dest, columns=cols)

    # Drop failed API rows, missing features/names, dupes.
    if "null_response" in df.columns:
        df = df[df["null_response"] != True]  # noqa: E712
    df = df.dropna(subset=["name"] + AUDIO_FEATURES).drop_duplicates("id")

    # Keep the most popular tracks: searchable, memory-light, real songs.
    df = df.sort_values("popularity", ascending=False).head(CONTENT_TOP_N).reset_index(drop=True)
    print(f"  kept top {len(df):,} tracks by popularity")

    scaler = MinMaxScaler()
    feature_matrix = scaler.fit_transform(df[AUDIO_FEATURES].values).astype(np.float32)

    np.save(MODELS_DIR / "content_features.npy", feature_matrix)
    meta = df[["id", "name", "popularity"]].rename(
        columns={"id": "track_id", "name": "track_name"})
    meta.to_parquet(MODELS_DIR / "content_meta.parquet")
    with open(MODELS_DIR / "content_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved content artifacts ({len(df):,} tracks; artist/genre fetched live via API).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=int, default=50, help="number of MPD slices (×1000 playlists)")
    ap.add_argument("--content-only", action="store_true")
    ap.add_argument("--collab-only", action="store_true")
    args = ap.parse_args()

    metrics = None
    if not args.content_only:
        metrics = train_collaborative(n_slices=args.slices)
    if not args.collab_only:
        train_content()
    print("\nDone. Artifacts in ./models")
    if metrics:
        print("ALS ranking metrics @10:", metrics)
