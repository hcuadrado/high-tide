# musicbrainz.py
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""MusicBrainz genre/tag enrichment for the taste corpus.

MusicBrainz enforces a hard 1 request/second rate limit and requires a
descriptive User-Agent, so every call goes through a module-level throttle and
results are cached to disk indefinitely (genres/tags are effectively stable).
This client is artist-level only for now; recording-level enrichment (per-track
genres via ISRC) can be layered on later using the same throttle and cache.

All calls must run on a worker thread — they block on the throttle.
"""

import json
import logging
import os
import threading
import time

import requests

from . import utils

logger = logging.getLogger(__name__)

_MB_BASE = "https://musicbrainz.org/ws/2"
_USER_AGENT = (
    "io.github.nokse22.high-tide/1.4.0 "
    "( https://github.com/hcuadrado/high-tide )"
)
# Stay comfortably above the 1 req/sec ceiling to avoid throttling/blocks.
_MIN_INTERVAL = 1.1
_TIMEOUT = 15
# Positives never expire; re-try artists we couldn't tag after this long, in
# case MusicBrainz coverage improved.
_NEGATIVE_TTL = 30 * 24 * 3600

_throttle_lock = threading.Lock()
_last_call = 0.0

_cache_lock = threading.Lock()
_cache: dict | None = None


#
#   DISK CACHE
#


def _cache_path() -> str:
    return os.path.join(utils.CACHE_DIR, "musicbrainz_cache.json")


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    try:
        with open(_cache_path()) as f:
            _cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    path = _cache_path()
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(_cache, f)
        os.replace(tmp_path, path)
    except Exception:
        logger.exception("musicbrainz: failed to write cache")


def _is_stale(entry: dict) -> bool:
    # Positive hits (we found genres or tags) are kept forever.
    if entry.get("genres") or entry.get("tags"):
        return False
    return (time.time() - entry.get("fetched_at", 0)) > _NEGATIVE_TTL


#
#   THROTTLED HTTP
#


def _throttled_get(path: str, params: dict, cancel_event=None) -> dict | None:
    global _last_call
    if cancel_event is not None and cancel_event.is_set():
        return None
    with _throttle_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
    params = {**params, "fmt": "json"}
    try:
        response = requests.get(
            f"{_MB_BASE}/{path}",
            params=params,
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
        )
        if response.status_code == 503:
            logger.warning("musicbrainz: rate limited (503) on %s", path)
            return None
        response.raise_for_status()
        return response.json()
    except Exception:
        logger.warning("musicbrainz: request failed for %s", path, exc_info=True)
        return None


#
#   RESOLUTION
#


def _extract_genres_tags(artist_json: dict) -> tuple[list, list]:
    def _names(items) -> list:
        ranked = sorted(
            (i for i in (items or []) if i.get("name")),
            key=lambda i: i.get("count", 0),
            reverse=True,
        )
        return [i["name"] for i in ranked]

    return _names(artist_json.get("genres")), _names(artist_json.get("tags"))


def _mbid_from_isrc(isrc: str, cancel_event=None) -> str | None:
    data = _throttled_get(
        "recording",
        {"query": f"isrc:{isrc}", "inc": "artist-credits", "limit": 1},
        cancel_event,
    )
    if not data:
        return None
    for rec in data.get("recordings", []):
        for credit in rec.get("artist-credit", []):
            artist = credit.get("artist") if isinstance(credit, dict) else None
            if artist and artist.get("id"):
                return artist["id"]
    return None


def _mbid_from_name(name: str, cancel_event=None) -> str | None:
    data = _throttled_get(
        "artist", {"query": f'artist:"{name}"', "limit": 1}, cancel_event
    )
    if not data:
        return None
    artists = data.get("artists", [])
    if artists and artists[0].get("id"):
        return artists[0]["id"]
    return None


def _fetch_artist_tags(mbid: str, cancel_event=None) -> tuple[list, list]:
    data = _throttled_get(
        f"artist/{mbid}", {"inc": "genres+tags"}, cancel_event
    )
    if not data:
        return [], []
    return _extract_genres_tags(data)


#
#   PUBLIC API
#


def get_cached(tidal_artist_id) -> dict | None:
    """Return the cached enrichment entry for an artist, or None if absent."""
    entry = _load_cache().get(str(tidal_artist_id))
    if entry and not _is_stale(entry):
        return entry
    return None


def enrich_artist(
    tidal_artist_id,
    name: str,
    isrc: str | None = None,
    cancel_event=None,
) -> dict:
    """Resolve and cache genres/tags for a TIDAL artist.

    Tries the (reliable) ISRC join first, falling back to a name search. The
    result — including negative hits — is written to the disk cache. Returns the
    cache entry: {"mbid", "genres", "tags", "fetched_at"}.
    """
    key = str(tidal_artist_id)
    cached = get_cached(tidal_artist_id)
    if cached is not None:
        return cached

    mbid = None
    if isrc:
        mbid = _mbid_from_isrc(isrc, cancel_event)
    if mbid is None and name:
        mbid = _mbid_from_name(name, cancel_event)

    genres: list = []
    tags: list = []
    if mbid is not None:
        genres, tags = _fetch_artist_tags(mbid, cancel_event)

    entry = {
        "mbid": mbid,
        "genres": genres,
        "tags": tags,
        "fetched_at": int(time.time()),
    }
    with _cache_lock:
        _load_cache()[key] = entry
        _save_cache()
    logger.debug(
        "musicbrainz: enriched artist %s (%r) -> mbid=%s genres=%s tags=%s",
        key, name, mbid, genres[:3], tags[:3],
    )
    return entry
