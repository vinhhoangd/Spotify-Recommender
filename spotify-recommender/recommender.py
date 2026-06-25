"""
Serving-time recommenders that load the artifacts produced by train.py.

- CollaborativeRecommender: artist→artist recommendations from ALS latent
  factors (cosine similarity in the learned embedding space). Supports
  multi-seed queries by averaging seed artist vectors.
- ContentRecommender: track→track recommendations via cosine similarity
  over scaled Spotify audio features.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"


def artifacts_exist() -> bool:
    required = ["artist_factors.npy", "artist_names.npy",
                "content_features.npy", "content_meta.parquet"]
    return all((MODELS_DIR / f).exists() for f in required)


class CollaborativeRecommender:
    """Artist recommendations from ALS latent factors (Last.fm-trained)."""

    def __init__(self):
        self.factors = np.load(MODELS_DIR / "artist_factors.npy")
        self.names = np.load(MODELS_DIR / "artist_names.npy", allow_pickle=True)
        # Normalise factors once so dot product == cosine similarity.
        norms = np.linalg.norm(self.factors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self._unit = self.factors / norms
        self._lower = {n.lower(): i for i, n in enumerate(self.names)}
        with open(MODELS_DIR / "collab_meta.json") as f:
            self.meta = json.load(f)

    def search_artists(self, query: str, limit: int = 10) -> list[str]:
        q = query.lower().strip()
        exact = [n for n in self.names if n.lower() == q]
        prefix = [n for n in self.names if n.lower().startswith(q) and n.lower() != q]
        contains = [n for n in self.names if q in n.lower()
                    and not n.lower().startswith(q)]
        seen, out = set(), []
        for n in exact + prefix + contains:
            if n not in seen:
                seen.add(n)
                out.append(n)
            if len(out) >= limit:
                break
        return out

    def recommend(self, seed_artists: list[str], n: int = 20) -> pd.DataFrame:
        idxs = [self._lower[a.lower()] for a in seed_artists if a.lower() in self._lower]
        if not idxs:
            return pd.DataFrame(columns=["artist", "score"])

        seed_vec = self._unit[idxs].mean(axis=0, keepdims=True)
        sims = (self._unit @ seed_vec.T).ravel()
        sims[idxs] = -np.inf  # exclude seeds

        top = np.argpartition(-sims, n)[:n]
        top = top[np.argsort(-sims[top])]
        return pd.DataFrame({"artist": self.names[top], "score": sims[top]}).reset_index(drop=True)


class ContentRecommender:
    """Track recommendations via cosine similarity on audio features."""

    def __init__(self):
        self.features = np.load(MODELS_DIR / "content_features.npy")
        self.meta = pd.read_parquet(MODELS_DIR / "content_meta.parquet").reset_index(drop=True)
        # Pre-normalise for fast cosine via dot product.
        norms = np.linalg.norm(self.features, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self._unit = self.features / norms
        self._id_to_idx = {tid: i for i, tid in enumerate(self.meta["track_id"].values)}
        self._name_lower = self.meta["track_name"].astype(str).str.lower()

    def search_tracks(self, query: str, limit: int = 15) -> pd.DataFrame:
        # This dataset has no artist/genre columns — search by track name only.
        q = query.lower().strip()
        mask = self._name_lower.str.contains(q, na=False, regex=False).values
        hits = self.meta[mask].copy()
        return hits.sort_values("popularity", ascending=False).head(limit)

    def recommend(self, seed_track_ids: list[str], n: int = 20) -> pd.DataFrame:
        idxs = [self._id_to_idx[t] for t in seed_track_ids if t in self._id_to_idx]
        if not idxs:
            return pd.DataFrame()

        seed_vec = self._unit[idxs].mean(axis=0, keepdims=True)
        sims = (self._unit @ seed_vec.T).ravel()
        sims[idxs] = -np.inf

        top = np.argpartition(-sims, min(n, len(sims) - 1))[:n]
        top = top[np.argsort(-sims[top])]
        result = self.meta.iloc[top].copy()
        result["score"] = sims[top]
        return result.reset_index(drop=True)
