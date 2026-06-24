"""
Hybrid recommender combining:
1. Collaborative filtering via ALS (Alternating Least Squares) matrix factorization
2. Content-based filtering on Spotify audio features
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

try:
    import implicit
    HAS_IMPLICIT = True
except ImportError:
    HAS_IMPLICIT = False

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]


class ContentBasedRecommender:
    """Recommends tracks by cosine similarity on Spotify audio features."""

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_matrix: np.ndarray | None = None
        self.track_ids: list[str] = []
        self.track_meta: pd.DataFrame | None = None

    def fit(self, tracks_df: pd.DataFrame, features_df: pd.DataFrame):
        merged = tracks_df.merge(features_df, on="track_id", how="inner").drop_duplicates("track_id")
        self.track_meta = merged[["track_id", "name", "artist"]].reset_index(drop=True)
        self.track_ids = merged["track_id"].tolist()

        cols = [c for c in AUDIO_FEATURES if c in merged.columns]
        raw = merged[cols].fillna(0).values
        self.feature_matrix = self.scaler.fit_transform(raw)

    def recommend(self, seed_track_ids: list[str], n: int = 20) -> pd.DataFrame:
        if self.feature_matrix is None:
            raise RuntimeError("Call fit() first.")

        idx_map = {tid: i for i, tid in enumerate(self.track_ids)}
        seed_idxs = [idx_map[tid] for tid in seed_track_ids if tid in idx_map]
        if not seed_idxs:
            return pd.DataFrame()

        seed_vec = self.feature_matrix[seed_idxs].mean(axis=0, keepdims=True)
        sims = cosine_similarity(seed_vec, self.feature_matrix)[0]

        # Exclude seeds
        for i in seed_idxs:
            sims[i] = -1

        top_idxs = np.argsort(sims)[::-1][:n]
        result = self.track_meta.iloc[top_idxs].copy()
        result["score"] = sims[top_idxs]
        return result.reset_index(drop=True)


class CollaborativeRecommender:
    """
    Implicit ALS matrix factorization on user-item interaction matrix.
    Falls back to popularity-based ranking if `implicit` is not installed.
    """

    def __init__(self, factors: int = 64, iterations: int = 20, regularization: float = 0.1):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.model = None
        self.user_ids: list[str] = []
        self.track_ids: list[str] = []
        self.track_meta: pd.DataFrame | None = None
        self.item_user_matrix: csr_matrix | None = None

    def fit(self, interactions_df: pd.DataFrame, track_meta_df: pd.DataFrame):
        """
        interactions_df: columns [user_id, track_id, weight]
        track_meta_df:   columns [track_id, name, artist]
        """
        self.user_ids = list(interactions_df["user_id"].unique())
        self.track_ids = list(interactions_df["track_id"].unique())
        self.track_meta = track_meta_df.set_index("track_id")

        user_idx = {u: i for i, u in enumerate(self.user_ids)}
        item_idx = {t: i for i, t in enumerate(self.track_ids)}

        rows = interactions_df["track_id"].map(item_idx)
        cols = interactions_df["user_id"].map(user_idx)
        data = interactions_df["weight"].values

        self.item_user_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.track_ids), len(self.user_ids)),
        )

        if HAS_IMPLICIT:
            self.model = implicit.als.AlternatingLeastSquares(
                factors=self.factors,
                iterations=self.iterations,
                regularization=self.regularization,
                use_gpu=False,
            )
            self.model.fit(self.item_user_matrix)

    def recommend_for_user(self, user_id: str, n: int = 20) -> pd.DataFrame:
        if self.item_user_matrix is None:
            raise RuntimeError("Call fit() first.")

        if HAS_IMPLICIT and user_id in self.user_ids and len(self.user_ids) > 1:
            uid = self.user_ids.index(user_id)
            user_items = self.item_user_matrix.T.tocsr()
            capped_n = min(n, len(self.track_ids))
            try:
                item_ids, scores = self.model.recommend(
                    uid, user_items, N=capped_n, filter_already_liked_items=True
                )
                recs = []
                for iid, score in zip(item_ids, scores):
                    tid = self.track_ids[iid]
                    meta = self.track_meta.loc[tid] if tid in self.track_meta.index else {}
                    recs.append({
                        "track_id": tid,
                        "name": meta.get("name", "Unknown"),
                        "artist": meta.get("artist", "Unknown"),
                        "score": float(score),
                    })
                return pd.DataFrame(recs)
            except Exception:
                pass  # fall through to popularity fallback

        # Fallback: popularity within dataset
        counts = (
            self.item_user_matrix.sum(axis=1).A1
        )
        top_idxs = np.argsort(counts)[::-1][:n]
        recs = []
        for i in top_idxs:
            tid = self.track_ids[i]
            meta = self.track_meta.loc[tid] if tid in self.track_meta.index else {}
            recs.append({
                "track_id": tid,
                "name": meta.get("name", "Unknown"),
                "artist": meta.get("artist", "Unknown"),
                "score": float(counts[i]),
            })
        return pd.DataFrame(recs)


class HybridRecommender:
    """Blends content-based and collaborative signals."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha  # weight for collaborative score
        self.content = ContentBasedRecommender()
        self.collab = CollaborativeRecommender()
        self._collab_fitted = False
        self._content_fitted = False

    def fit_content(self, tracks_df: pd.DataFrame, features_df: pd.DataFrame):
        self.content.fit(tracks_df, features_df)
        self._content_fitted = True

    def fit_collab(self, interactions_df: pd.DataFrame, track_meta_df: pd.DataFrame):
        self.collab.fit(interactions_df, track_meta_df)
        self._collab_fitted = True

    def recommend(self, seed_track_ids: list[str], user_id: str | None = None,
                  n: int = 20) -> pd.DataFrame:
        results = {}

        if self._content_fitted:
            cb = self.content.recommend(seed_track_ids, n=n * 2)
            for _, row in cb.iterrows():
                results[row["track_id"]] = {
                    "track_id": row["track_id"],
                    "name": row["name"],
                    "artist": row["artist"],
                    "content_score": row["score"],
                    "collab_score": 0.0,
                }

        if self._collab_fitted and user_id:
            cf = self.collab.recommend_for_user(user_id, n=n * 2)
            for _, row in cf.iterrows():
                tid = row["track_id"]
                if tid in results:
                    results[tid]["collab_score"] = row["score"]
                else:
                    results[tid] = {
                        "track_id": tid,
                        "name": row["name"],
                        "artist": row["artist"],
                        "content_score": 0.0,
                        "collab_score": row["score"],
                    }

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results.values())
        # Normalise each score to [0,1] before blending
        for col in ["content_score", "collab_score"]:
            mx = df[col].max()
            if mx > 0:
                df[col] = df[col] / mx

        df["hybrid_score"] = (1 - self.alpha) * df["content_score"] + self.alpha * df["collab_score"]
        df = df[~df["track_id"].isin(seed_track_ids)]
        return df.sort_values("hybrid_score", ascending=False).head(n).reset_index(drop=True)
