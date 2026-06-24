import os
import time
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from spotify_client import (
    get_spotify_oauth,
    get_user_client,
    fetch_user_top_tracks,
    fetch_saved_tracks,
    fetch_recently_played,
    fetch_audio_features,
    fetch_recommendations,
)
from recommender import HybridRecommender

load_dotenv()

st.set_page_config(page_title="Spotify Recommender", page_icon="🎵", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  body { background: #0d0d0d; }
  .track-card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: transform .15s;
  }
  .track-card:hover { transform: scale(1.01); }
  .track-title { font-weight: 700; font-size: 15px; color: #fff; }
  .track-artist { font-size: 13px; color: #aaa; }
  .score-badge {
    background: #1db954;
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 12px;
    font-weight: 700;
    color: #000;
    white-space: nowrap;
  }
</style>
""", unsafe_allow_html=True)

# ── Auth helpers ─────────────────────────────────────────────────────────────

def auth_flow():
    sp_oauth = get_spotify_oauth()
    params = st.query_params
    if "code" in params:
        code = params["code"]
        token = sp_oauth.get_access_token(code, as_dict=True)
        st.session_state["token_info"] = token
        st.query_params.clear()
        st.rerun()

    if "token_info" not in st.session_state:
        auth_url = sp_oauth.get_authorize_url()
        st.title("🎵 Spotify Recommender")
        st.markdown("### Connect your Spotify account to get personalised recommendations.")
        st.markdown(f'<a href="{auth_url}" target="_self"><button style="background:#1db954;color:#000;border:none;padding:12px 28px;border-radius:50px;font-size:16px;font-weight:700;cursor:pointer;">Connect with Spotify</button></a>',
                    unsafe_allow_html=True)
        st.stop()

    token_info = st.session_state["token_info"]
    sp_oauth = get_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info["refresh_token"])
        st.session_state["token_info"] = token_info
    return get_user_client(token_info)


def track_card(track: dict, score: float | None = None):
    img = track.get("image_url", "")
    img_html = f'<img src="{img}" width="56" height="56" style="border-radius:6px;object-fit:cover;">' if img else ""
    score_html = f'<span class="score-badge">{score:.2f}</span>' if score is not None else ""
    ext = track.get("external_url", "#")
    preview = track.get("preview_url")
    audio_html = f'<audio controls src="{preview}" style="height:28px;width:160px;"></audio>' if preview else ""
    st.markdown(f"""
    <div class="track-card">
      {img_html}
      <div style="flex:1;min-width:0;">
        <div class="track-title"><a href="{ext}" target="_blank" style="color:inherit;text-decoration:none;">{track['name']}</a></div>
        <div class="track-artist">{track['artist']}</div>
      </div>
      {audio_html}
      {score_html}
    </div>
    """, unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.getenv("SPOTIFY_CLIENT_ID") or not os.getenv("SPOTIFY_CLIENT_SECRET"):
        st.error("Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET. Copy `.env.example` to `.env` and fill in your credentials.")
        st.stop()

    sp = auth_flow()

    st.title("🎵 Spotify Recommender")
    st.caption("Hybrid content-based + collaborative filtering recommendations powered by your listening history.")

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        n_recs = st.slider("Number of recommendations", 5, 50, 20)
        alpha = st.slider("Collaborative weight (α)", 0.0, 1.0, 0.4,
                          help="0 = pure content-based, 1 = pure collaborative")
        use_spotify_recs = st.checkbox("Also show Spotify's own recommendations", value=True)
        st.divider()
        if st.button("🔄 Refresh data", use_container_width=True):
            for key in ["top_tracks", "saved_tracks", "recent_tracks", "audio_features"]:
                st.session_state.pop(key, None)
        if st.button("🚪 Log out", use_container_width=True):
            st.session_state.clear()
            try:
                os.remove(".spotify_cache")
            except FileNotFoundError:
                pass
            st.rerun()

    # ── Fetch user data ──────────────────────────────────────────────────────
    with st.spinner("Loading your Spotify library…"):
        if "top_tracks" not in st.session_state:
            st.session_state["top_tracks"] = fetch_user_top_tracks(sp)
        if "saved_tracks" not in st.session_state:
            st.session_state["saved_tracks"] = fetch_saved_tracks(sp, limit=100)
        if "recent_tracks" not in st.session_state:
            st.session_state["recent_tracks"] = fetch_recently_played(sp)

    top_tracks = st.session_state["top_tracks"]
    saved_tracks = st.session_state["saved_tracks"]
    recent_tracks = st.session_state["recent_tracks"]

    # Build unified track list
    all_tracks = pd.DataFrame(top_tracks + saved_tracks).drop_duplicates("track_id")
    if all_tracks.empty:
        st.warning("No tracks found in your library. Play some music on Spotify first!")
        st.stop()

    # Fetch audio features
    if "audio_features" not in st.session_state:
        with st.spinner("Fetching audio features…"):
            feats = fetch_audio_features(sp, all_tracks["track_id"].tolist())
            st.session_state["audio_features"] = feats
    features_df = pd.DataFrame(st.session_state["audio_features"])

    # ── Build & run recommender ──────────────────────────────────────────────
    rec = HybridRecommender(alpha=alpha)
    rec.fit_content(all_tracks[["track_id", "name", "artist"]], features_df)

    # Build interaction matrix from weighted signals
    rows = []
    user_id = "me"
    for t in top_tracks:
        weight = 3 if t["time_range"] == "short_term" else (2 if t["time_range"] == "medium_term" else 1)
        rows.append({"user_id": user_id, "track_id": t["track_id"], "weight": weight})
    for t in saved_tracks:
        rows.append({"user_id": user_id, "track_id": t["track_id"], "weight": 2})
    for t in recent_tracks:
        rows.append({"user_id": user_id, "track_id": t["track_id"], "weight": 1})

    interactions_df = (
        pd.DataFrame(rows)
        .groupby(["user_id", "track_id"], as_index=False)["weight"].sum()
    )
    rec.fit_collab(interactions_df, all_tracks[["track_id", "name", "artist"]])

    seed_ids = [t["track_id"] for t in top_tracks[:10]]
    hybrid_recs = rec.recommend(seed_ids, user_id=user_id, n=n_recs)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_recs, tab_library, tab_features, tab_about = st.tabs(
        ["🎯 Recommendations", "📚 Your Library", "📊 Audio Analysis", "ℹ️ How It Works"]
    )

    with tab_recs:
        st.subheader("Hybrid Recommendations")
        if hybrid_recs.empty:
            st.info("Not enough data to generate recommendations yet. Try saving or listening to more tracks!")
        else:
            # Enrich with Spotify metadata for images / preview
            enriched_ids = hybrid_recs["track_id"].tolist()
            try:
                sp_tracks = sp.tracks(enriched_ids)["tracks"]
                meta_map = {
                    t["id"]: {
                        "image_url": t["album"]["images"][0]["url"] if t["album"]["images"] else None,
                        "preview_url": t["preview_url"],
                        "external_url": t["external_urls"]["spotify"],
                    }
                    for t in sp_tracks if t
                }
            except Exception:
                meta_map = {}

            for _, row in hybrid_recs.iterrows():
                meta = meta_map.get(row["track_id"], {})
                track_card({
                    "name": row["name"],
                    "artist": row["artist"],
                    **meta,
                }, score=row["hybrid_score"])

        if use_spotify_recs and seed_ids:
            st.subheader("Spotify's Recommendations (baseline)")
            try:
                spotify_recs = fetch_recommendations(sp, seed_ids[:5], n=n_recs)
                for t in spotify_recs:
                    track_card(t)
            except Exception as e:
                st.warning(f"Could not load Spotify recommendations: {e}")

    with tab_library:
        st.subheader("Your Top Tracks")
        top_df = pd.DataFrame(top_tracks).drop_duplicates("track_id")[["name", "artist", "popularity", "time_range"]]
        st.dataframe(top_df, use_container_width=True, hide_index=True)

        st.subheader("Recently Played")
        recent_df = pd.DataFrame(recent_tracks)[["name", "artist", "played_at"]]
        st.dataframe(recent_df, use_container_width=True, hide_index=True)

    with tab_features:
        if features_df.empty:
            st.info("No audio features available.")
        else:
            merged_feat = all_tracks.merge(features_df, on="track_id")
            st.subheader("Audio Feature Distribution")
            feat_cols = [c for c in ["danceability", "energy", "valence", "acousticness", "speechiness", "instrumentalness"] if c in merged_feat.columns]
            selected = st.multiselect("Features to display", feat_cols, default=feat_cols[:4])
            if selected:
                melted = merged_feat[["name"] + selected].melt(id_vars="name", var_name="feature", value_name="value")
                fig = px.violin(melted, x="feature", y="value", color="feature",
                                box=True, points=False,
                                color_discrete_sequence=px.colors.qualitative.Plotly)
                fig.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
                                  font_color="white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Tempo vs Energy")
            if "tempo" in merged_feat.columns and "energy" in merged_feat.columns:
                fig2 = px.scatter(merged_feat.drop_duplicates("track_id").head(200),
                                  x="tempo", y="energy", color="danceability" if "danceability" in merged_feat.columns else None,
                                  hover_data=["name", "artist"],
                                  color_continuous_scale="Viridis")
                fig2.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
                                   font_color="white")
                st.plotly_chart(fig2, use_container_width=True)

    with tab_about:
        st.markdown("""
## How the Recommender Works

### 1. Data Collection
Your top tracks (short / medium / long term), saved library, and recently played tracks are pulled from the Spotify Web API and assigned **interaction weights**:

| Signal | Weight |
|---|---|
| Top track (short-term) | 3 |
| Top track (medium-term) / saved | 2 |
| Top track (long-term) / recently played | 1 |

### 2. Content-Based Filtering
Spotify provides **audio features** for every track:
`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`.

These are normalised and used to build a feature matrix. The **cosine similarity** between the average of your seed tracks and every candidate track produces a content score.

### 3. Collaborative Filtering (ALS Matrix Factorization)
The user-item interaction matrix is decomposed using **Alternating Least Squares** (ALS) from the `implicit` library — the same algorithm used in large-scale production recommenders (e.g. Spotify Discover Weekly internally uses similar ideas).

Each user and item is embedded in a latent factor space. For a new user with limited history, the model leverages the implicit feedback signal (listen count proxy) to find items popular with similar taste profiles.

### 4. Hybrid Blending
The final score is a **weighted average**:

```
hybrid_score = (1 - α) × content_score + α × collaborative_score
```

Adjust **α** in the sidebar to favour content similarity (0) or collaborative signals (1).

### Architecture
```
Spotify API ──► Data Fetcher ──► Feature Extractor
                     │                    │
                     ▼                    ▼
             Interaction Matrix    Audio Feature Matrix
                     │                    │
                     ▼                    ▼
             ALS Collaborative    Cosine Content-Based
                     │                    │
                     └────── Hybrid ──────┘
                                  │
                             Streamlit UI
```
        """)


if __name__ == "__main__":
    main()
