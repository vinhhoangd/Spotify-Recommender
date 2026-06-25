# 🎵 Spotify Recommender

A music recommendation system combining **collaborative filtering (ALS matrix factorization)** and **content-based filtering**, served through a **Streamlit** UI. Models are trained offline on public multi-user datasets; the Spotify Web API is used only to enrich results with album/artist cover art.

> **Why offline training?** Spotify deprecated the `/audio-features` and `/recommendations` endpoints for new apps in November 2024, and a single user's listening history cannot drive matrix factorization (which needs *many* users to learn taste patterns). So the ML is trained on public datasets with real multi-user signal, and the live API is reserved for what it still does well — fetching cover art.

---

## ✨ Features

- **Similar Artists** — ALS matrix factorization trained on the Spotify Million Playlist Dataset (100k playlists, 3.8M artist co-occurrences). Search any artist (or several) and get recommendations from the learned latent space.
- **Similar Tracks** — cosine similarity over normalized Spotify audio features across ~90k tracks, with an optional same-genre constraint.
- **Offline evaluation** — precision@k, MAP, and NDCG on a held-out split, surfaced live in the UI.
- **Cover art enrichment** — optional, via the Spotify Web API (Client Credentials flow — no user login).
- **Fully reproducible** — one `train.py` run downloads the data, trains, evaluates, and saves all artifacts.

---

## 📊 Model performance

ALS collaborative model, evaluated @10 on a 10% held-out split of the Million Playlist Dataset:

| Metric | Score |
|---|---|
| Precision@10 | **0.108** |
| MAP@10 | **0.059** |
| NDCG@10 | **0.097** |
| AUC | **0.559** |

Trained on **100,000 playlists × ~45,000 artists**, 96 latent factors, 20 iterations, BM25-weighted artist co-occurrences.

**Qualitative sanity check:**

| Seed artist | Top recommendations |
|---|---|
| The Weeknd | PARTYNEXTDOOR, Jeremih, Tory Lanez, Tinashe, Niykee Heaton |
| Drake | Big Sean, French Montana, Ty Dolla $ign, DJ Khaled, Desiigner |
| Kendrick Lamar | J. Cole, Joey Bada$$, A$AP Rocky, ScHoolboy Q, Vic Mensa |

---

## 🧠 How it works

### 1. Collaborative filtering — ALS matrix factorization
**Dataset:** [Spotify Million Playlist Dataset](https://huggingface.co/datasets/jaxliu/Spotify_Million_Playlist_Dataset_Challenge) (2018) — 100k playlists, 3.8M artist co-occurrences, ~45k artists (auto-downloaded from a HuggingFace mirror).

Playlists are treated as "users" and artists as "items"; an artist's weight in a playlist is how many of its tracks appear there. The matrix is weighted with **BM25** to dampen large playlists and ubiquitous artists, then factorized with **Alternating Least Squares (ALS)**. Each artist becomes a dense latent vector; similar artists are found by **cosine similarity** in that embedding space. Multi-seed queries average the seed vectors before searching.

### 2. Content-based filtering
**Dataset:** [Spotify audio-features dump](https://huggingface.co/datasets/kevinanjalo/spotify_audio_features) — the 500k most popular tracks (of 25.5M) with audio features. Artist names and cover art are fetched live from the Spotify API by track ID (the dataset stores track name + features only).

Nine features (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`) are min-max scaled. Track-to-track similarity is **cosine similarity** over these vectors.

> Audio features come from Spotify's `/audio-features` endpoint (deprecated for new apps in Nov 2024), so this catalog is capped at the pre-deprecation era — no 2025 releases, since the features to describe them can no longer be computed by anyone.

### Architecture

```
Million Playlist Dataset ──► BM25 weight  ──► ALS factorization ──► artist embeddings ┐
Spotify Tracks ───────────► MinMax scale ──► audio-feature matrix ─► cosine similarity ┤
                                                                                       ▼
                                                                Streamlit UI ◄── Spotify API (cover art)
```

---

## 🚀 Getting started

```bash
cd spotify-recommender
pip install -r requirements.txt

# Train models (downloads ~3.3GB of data, ~8-12 min one-time)
# Use fewer playlists for a faster/lighter run:  python train.py --slices 30
python train.py

# Launch the app
streamlit run app.py
```

Open http://127.0.0.1:8501.

### Optional: cover art
Recommendations work without any credentials. To show album/artist images, create a `.env` from the template and add [Spotify app credentials](https://developer.spotify.com/dashboard):

```bash
cp .env.example .env
# edit .env:
#   SPOTIFY_CLIENT_ID=...
#   SPOTIFY_CLIENT_SECRET=...
```

No redirect URI or user login is needed — the app uses the Client Credentials flow.

---

## 📁 Project structure

```
spotify-recommender/
├── train.py           # Offline pipeline: download → train → evaluate → save artifacts
├── recommender.py     # Serving: CollaborativeRecommender + ContentRecommender
├── spotify_client.py  # Spotify Web API (Client Credentials) — cover art enrichment
├── app.py             # Streamlit UI
├── requirements.txt
├── .env.example
└── models/            # Generated by train.py (gitignored)
```

---

## 🛠️ Tech stack

`Python` · `implicit` (ALS) · `scikit-learn` · `pandas` / `numpy` / `scipy` · `Streamlit` · `spotipy`

---

## 📌 Notes & limitations

- Collaborative recommendations are **artist-level**, aggregated from playlist co-occurrence.
- The Million Playlist Dataset is from **2018**, so the collaborative model covers mainstream artists through 2018 but not 2019+ debuts or the obscure long tail. The Content tab uses a newer (~2022) catalog.
- Content recommendations match acoustic profile, which can cross genres — use the same-genre toggle for tighter results.
- Model artifacts (`models/`) are gitignored; regenerate them with `python train.py`.
