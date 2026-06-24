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


@functools.lru_cache(maxsize=1)
def get_client():
    cid = os.getenv("SPOTIFY_CLIENT_ID")
    secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not secret or secret == "your_client_secret_here":
        return None
    try:
        auth = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        return spotipy.Spotify(auth_manager=auth, requests_timeout=10, retries=2)
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
