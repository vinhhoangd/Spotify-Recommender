import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

SCOPES = "user-top-read user-library-read playlist-read-private user-read-recently-played"


def get_spotify_oauth():
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8501"),
        scope=SCOPES,
        cache_path=".spotify_cache",
        open_browser=False,
    )


def get_client_credentials_client():
    auth = SpotifyClientCredentials(
        client_id=os.getenv("SPOTIFY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    )
    return spotipy.Spotify(auth_manager=auth)


def get_user_client(token_info: dict):
    return spotipy.Spotify(auth=token_info["access_token"])


def fetch_user_top_tracks(sp: spotipy.Spotify, limit: int = 50) -> list[dict]:
    results = []
    for term in ["short_term", "medium_term", "long_term"]:
        data = sp.current_user_top_tracks(limit=limit, time_range=term)
        for item in data["items"]:
            results.append({"track_id": item["id"], "name": item["name"],
                            "artist": item["artists"][0]["name"],
                            "popularity": item["popularity"], "time_range": term})
    return results


def fetch_saved_tracks(sp: spotipy.Spotify, limit: int = 50) -> list[dict]:
    results = []
    offset = 0
    while len(results) < limit:
        data = sp.current_user_saved_tracks(limit=min(50, limit - len(results)), offset=offset)
        if not data["items"]:
            break
        for item in data["items"]:
            t = item["track"]
            results.append({"track_id": t["id"], "name": t["name"],
                            "artist": t["artists"][0]["name"],
                            "popularity": t["popularity"]})
        offset += 50
    return results


def fetch_recently_played(sp: spotipy.Spotify, limit: int = 50) -> list[dict]:
    data = sp.current_user_recently_played(limit=limit)
    results = []
    for item in data["items"]:
        t = item["track"]
        results.append({"track_id": t["id"], "name": t["name"],
                        "artist": t["artists"][0]["name"],
                        "played_at": item["played_at"]})
    return results


def fetch_audio_features(sp: spotipy.Spotify, track_ids: list[str]) -> list[dict]:
    features = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i + 100]
        data = sp.audio_features(batch)
        features.extend([f for f in data if f])
    return features


def fetch_recommendations(sp: spotipy.Spotify, seed_tracks: list[str],
                           seed_artists: list[str] = None, limit: int = 20) -> list[dict]:
    seed_tracks = seed_tracks[:5]
    seed_artists = (seed_artists or [])[:max(0, 5 - len(seed_tracks))]
    data = sp.recommendations(seed_tracks=seed_tracks, seed_artists=seed_artists, limit=limit)
    return [{"track_id": t["id"], "name": t["name"],
             "artist": t["artists"][0]["name"],
             "album": t["album"]["name"],
             "preview_url": t["preview_url"],
             "external_url": t["external_urls"]["spotify"],
             "image_url": t["album"]["images"][0]["url"] if t["album"]["images"] else None}
            for t in data["tracks"]]
