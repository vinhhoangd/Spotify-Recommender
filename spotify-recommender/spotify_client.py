"""
Live Spotify access for *enrichment only* — cover art, preview URLs, and
canonical names. Uses the Client Credentials flow (no user login), so it
avoids both the OAuth dance and the endpoints Spotify deprecated for new
apps in Nov 2024 (/audio-features, /recommendations).

All functions degrade to {} / [] if credentials are missing or a call fails,
so the app still works offline from the trained models.
"""
import os
import functools
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()


def _read_cred(key: str) -> str | None:
    """Look for a credential in env vars first, then Streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return None


def creds_status() -> dict:
    """Non-secret diagnostic: which source each credential came from."""
    def src(key):
        if os.getenv(key):
            return "env"
        try:
            import streamlit as st
            if key in st.secrets:
                return "secrets"
        except Exception:
            pass
        return "missing"
    return {"SPOTIFY_CLIENT_ID": src("SPOTIFY_CLIENT_ID"),
            "SPOTIFY_CLIENT_SECRET": src("SPOTIFY_CLIENT_SECRET")}


@functools.lru_cache(maxsize=1)
def get_client():
    cid = _read_cred("SPOTIFY_CLIENT_ID")
    secret = _read_cred("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret or secret == "your_client_secret_here":
        return None
    try:
        auth = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        sp = spotipy.Spotify(auth_manager=auth, requests_timeout=10, retries=2)
        sp.search(q="test", type="track", limit=1)  # validate creds eagerly
        return sp
    except Exception:
        return None


def is_available() -> bool:
    return get_client() is not None


@functools.lru_cache(maxsize=2048)
def enrich_track(track_name: str, artist: str) -> dict:
    """Return cover art / preview / link for a track by name+artist."""
    sp = get_client()
    if sp is None:
        return {}
    try:
        q = f"track:{track_name} artist:{artist}"
        res = sp.search(q=q, type="track", limit=1)
        items = res.get("tracks", {}).get("items", [])
        if not items:
            return {}
        t = items[0]
        imgs = t["album"].get("images", [])
        return {
            "image_url": imgs[0]["url"] if imgs else None,
            "preview_url": t.get("preview_url"),
            "external_url": t["external_urls"].get("spotify"),
        }
    except Exception:
        return {}


@functools.lru_cache(maxsize=8192)
def _enrich_one_track(track_id: str) -> dict:
    """Single-track lookup. Spotify 403-blocks the batch /tracks endpoint for
    new apps, but the single /tracks/{id} endpoint works, so we fetch one at a
    time (results are cached, and only ~12-15 are shown per query)."""
    sp = get_client()
    if sp is None:
        return {}
    try:
        t = sp.track(track_id)
    except Exception:
        return {}
    imgs = t["album"].get("images", [])
    return {
        "artist": ", ".join(a["name"] for a in t.get("artists", [])) or "Unknown",
        "image_url": imgs[0]["url"] if imgs else None,
        "preview_url": t.get("preview_url"),
        "external_url": t["external_urls"].get("spotify"),
    }


def enrich_tracks_by_id(track_ids: tuple[str, ...]) -> dict:
    """Fetch artist name / cover art / preview / link for Spotify track IDs.
    Returns {track_id: {...}}. Used by the content recommender, whose dataset
    has no artist/genre columns."""
    if get_client() is None:
        return {}
    out = {}
    for tid in track_ids:
        meta = _enrich_one_track(tid)
        if meta:
            out[tid] = meta
    return out


@functools.lru_cache(maxsize=2048)
def enrich_artist(artist: str) -> dict:
    """Return image / link / genres / followers for an artist by name."""
    sp = get_client()
    if sp is None:
        return {}
    try:
        res = sp.search(q=f"artist:{artist}", type="artist", limit=1)
        items = res.get("artists", {}).get("items", [])
        if not items:
            return {}
        a = items[0]
        imgs = a.get("images", [])
        return {
            "image_url": imgs[0]["url"] if imgs else None,
            "external_url": a["external_urls"].get("spotify"),
            "genres": a.get("genres", []),
            "followers": a.get("followers", {}).get("total"),
        }
    except Exception:
        return {}
