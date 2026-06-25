"""
On deployment hosts (e.g. Streamlit Community Cloud) the trained ``models/``
folder isn't in the git repo. This module downloads the artifacts bundle from
a GitHub Release on first boot and extracts it next to the app. Locally, if the
models already exist, it's a no-op.
"""
import io
import os
import tarfile
from pathlib import Path

import requests

MODELS_DIR = Path(__file__).parent / "models"
REQUIRED = ["artist_factors.npy", "artist_names.npy", "collab_meta.json",
            "content_features.npy", "content_meta.parquet", "content_scaler.pkl"]

# GitHub Release asset holding models.tar.gz. Override with the MODELS_URL env
# var / Streamlit secret if you host the bundle elsewhere.
DEFAULT_MODELS_URL = ("https://github.com/vinhhoangd/Spotify-Recommender/"
                      "releases/download/models-v1/models.tar.gz")


def models_present() -> bool:
    return all((MODELS_DIR / f).exists() for f in REQUIRED)


def ensure_models() -> None:
    if models_present():
        return
    url = os.getenv("MODELS_URL", DEFAULT_MODELS_URL)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        # Bundle is created with `-C spotify-recommender models`, so members are
        # prefixed with "models/". Extract into the app directory.
        tar.extractall(path=MODELS_DIR.parent)
    if not models_present():
        raise RuntimeError(
            f"Model bundle from {url} did not contain the expected artifacts.")
