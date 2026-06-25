import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from recommender import artifacts_exist, CollaborativeRecommender, ContentRecommender
import spotify_client as sc

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")

st.markdown("""
<style>
  .card {
    background: #181828; border-radius: 12px; padding: 12px 14px;
    margin-bottom: 10px; display: flex; align-items: center; gap: 14px;
  }
  .card:hover { background: #1f1f33; }
  .title { font-weight: 700; font-size: 15px; color: #fff; }
  .sub { font-size: 13px; color: #9aa; }
  .badge { background: #1db954; border-radius: 6px; padding: 3px 9px;
           font-size: 12px; font-weight: 700; color: #000; white-space: nowrap; }
  .rank { color: #555; font-weight: 700; width: 24px; text-align: right; }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = Path(__file__).parent / "models"


@st.cache_resource(show_spinner="Loading trained models…")
def load_models():
    return CollaborativeRecommender(), ContentRecommender()


def img_or_placeholder(url, size=56):
    if url:
        return f'<img src="{url}" width="{size}" height="{size}" style="border-radius:8px;object-fit:cover;">'
    return f'<div style="width:{size}px;height:{size}px;border-radius:8px;background:#333;display:flex;align-items:center;justify-content:center;">🎵</div>'


def render_card(rank, title, sub, score, image_url=None, link=None, preview=None):
    title_html = f'<a href="{link}" target="_blank" style="color:inherit;text-decoration:none;">{title}</a>' if link else title
    audio = f'<audio controls src="{preview}" style="height:30px;width:150px;"></audio>' if preview else ""
    st.markdown(f"""
    <div class="card">
      <div class="rank">{rank}</div>
      {img_or_placeholder(image_url)}
      <div style="flex:1;min-width:0;">
        <div class="title">{title_html}</div>
        <div class="sub">{sub}</div>
      </div>
      {audio}
      <span class="badge">{score:.3f}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Guard: artifacts present? ─────────────────────────────────────────────────
if not artifacts_exist():
    st.title("🎵 Music Recommender")
    st.error("Trained models not found. Run the training pipeline first:")
    st.code("python train.py", language="bash")
    st.caption("This downloads the Million Playlist Dataset and Spotify Tracks datasets "
               "and trains the ALS + content models (~2-3 min, one time).")
    st.stop()

collab, content = load_models()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎵 Music Recommender")
st.caption("ALS matrix factorization (Spotify Million Playlist Dataset) + content-based filtering "
           "(Spotify audio features). Live Spotify API used for cover art.")

with st.sidebar:
    st.header("⚙️ Settings")
    n_recs = st.slider("Number of recommendations", 5, 30, 12)
    enrich = st.checkbox("Fetch cover art from Spotify", value=sc.is_available(),
                         disabled=not sc.is_available(),
                         help="Requires SPOTIFY_CLIENT_ID/SECRET in .env" if not sc.is_available() else None)
    if not sc.is_available():
        st.caption("⚠️ No Spotify credentials — recommendations work, but without cover art.")
    st.divider()
    with st.expander("📊 Model stats"):
        try:
            cm = json.loads((MODELS_DIR / "collab_meta.json").read_text())
            st.metric("Artists (ALS)", f"{cm['n_artists']:,}")
            st.metric("Users trained on", f"{cm['n_users']:,}")
            st.metric("Latent factors", cm["factors"])
            m = cm.get("metrics_at_10", {})
            if m:
                st.caption("**Offline eval @10**")
                st.write({k: round(v, 4) for k, v in m.items()})
            st.metric("Tracks (content)", f"{len(content.meta):,}")
        except Exception:
            pass

tab_artist, tab_track, tab_about = st.tabs(
    ["🎤 Similar Artists (Collaborative)", "🎶 Similar Tracks (Content)", "ℹ️ How It Works"]
)

# ── Collaborative: artist → artists ──────────────────────────────────────────
with tab_artist:
    st.subheader("Find artists similar to ones you like")
    st.caption("Powered by ALS matrix factorization on 100k Spotify playlists (3.8M artist co-occurrences).")
    query = st.text_input("Search an artist", placeholder="e.g. Radiohead, Daft Punk, Kendrick Lamar",
                          key="artist_q")
    if query:
        matches = collab.search_artists(query, limit=8)
        if not matches:
            st.warning(
                f"No artist matching **'{query}'** in the dataset. "
                "The collaborative model is trained on the **2018** Million Playlist "
                "Dataset, so it covers mainstream artists through 2018 but not 2019+ "
                "debuts (e.g. Olivia Rodrigo, Ice Spice) or very obscure artists.\n\n"
                "👉 Try the **Similar Tracks (Content)** tab — its catalog is newer."
            )
        else:
            seeds = st.multiselect("Seed artist(s)", matches, default=matches[:1], key="artist_seeds")
            if seeds:
                recs = collab.recommend(seeds, n=n_recs)
                if recs.empty:
                    st.info("No recommendations found.")
                else:
                    for i, row in recs.iterrows():
                        meta = sc.enrich_artist(row["artist"]) if enrich else {}
                        genres = ", ".join(meta.get("genres", [])[:3])
                        followers = meta.get("followers")
                        sub = genres or (f"{followers:,} followers" if followers else "artist")
                        render_card(i + 1, row["artist"], sub, row["score"],
                                    image_url=meta.get("image_url"), link=meta.get("external_url"))

# ── Content: track → tracks ──────────────────────────────────────────────────
with tab_track:
    st.subheader("Find tracks that sound similar")
    st.caption("Cosine similarity over 9 normalised audio features across 114k tracks.")
    tquery = st.text_input("Search a track or artist", placeholder="e.g. Bohemian Rhapsody",
                           key="track_q")
    same_genre = st.checkbox("Restrict to same genre", value=False)
    if tquery:
        hits = content.search_tracks(tquery, limit=15)
        if hits.empty:
            st.warning(f"No track matching '{tquery}' in the catalog.")
        else:
            labels = [f"{r.track_name} — {r.artists} ({r.track_genre})" for r in hits.itertuples()]
            choice = st.selectbox("Pick the seed track", options=range(len(labels)),
                                  format_func=lambda i: labels[i], key="track_pick")
            seed_id = hits.iloc[choice]["track_id"]
            recs = content.recommend([seed_id], n=n_recs, same_genre=same_genre)
            if recs.empty:
                st.info("No recommendations found.")
            else:
                for i, row in recs.iterrows():
                    meta = sc.enrich_track(row["track_name"], row["artists"]) if enrich else {}
                    render_card(i + 1, row["track_name"],
                                f"{row['artists']} · {row['track_genre']}", row["score"],
                                image_url=meta.get("image_url"), link=meta.get("external_url"),
                                preview=meta.get("preview_url"))

with tab_about:
    st.markdown("""
## How It Works

This recommender is trained **offline** on two public datasets and served live.
The Spotify API is used only to fetch cover art — not for the ML — because Spotify
deprecated `/audio-features` and `/recommendations` for new apps in Nov 2024.

### 1. Collaborative Filtering — ALS Matrix Factorization
**Dataset:** Spotify Million Playlist Dataset (2018) — 100k playlists, 3.8M
artist co-occurrences, ~45k artists (kept those in ≥3 playlists).

Playlists are treated as "users" and artists as "items"; an artist's weight in
a playlist is how many of its tracks appear there. The matrix is weighted with
**BM25** (dampens huge playlists and ubiquitous artists), then factorized with
**Alternating Least Squares (ALS)** from the `implicit` library. Each artist is
embedded as a dense latent vector; similar artists are found by **cosine
similarity** in that learned space. Multi-seed queries average the seed vectors.

This is the same family of algorithm behind production music recommenders.
Model quality is measured offline with **precision@k / MAP / NDCG** on a 10%
held-out split (see *Model stats* in the sidebar).

### 2. Content-Based Filtering
**Dataset:** Spotify Tracks — 114k tracks across 125 genres with audio features.

Nine features (`danceability`, `energy`, `loudness`, `speechiness`,
`acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`) are
min-max scaled. Track-to-track similarity is **cosine similarity** over these
vectors, optionally constrained to the seed's genre.

### Architecture
```
Million Playlist Dataset ─► BM25 weight ─► ALS factorization ─► artist embeddings
Spotify Tracks ──────────► MinMax scale ─► audio-feature matrix ► cosine similarity
                                          │
                                   Streamlit UI ◄── Spotify API (cover art)
```

### Reproduce
```bash
pip install -r requirements.txt
python train.py        # downloads data, trains, saves to ./models
streamlit run app.py
```
    """)
