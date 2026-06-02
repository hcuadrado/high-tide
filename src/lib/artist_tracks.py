# artist_tracks.py
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Persisted artist top-tracks cache for AI Radio.

An artist's top tracks are stable, so they are cached to disk and reloaded
without hitting the network. The raw TIDAL JSON for each track is stored and
rebuilt into full :class:`Track` objects via ``session.parse_track`` — building
``Track(session, id)`` directly would issue one network request per track,
whereas ``parse_track`` parses an in-memory JSON object with no request.

All calls must run on a worker thread.
"""

import json
import logging
import os
import threading
import time

from . import utils

logger = logging.getLogger(__name__)

# Top tracks rarely change; positives live this long. Empty results (the artist
# has no top tracks, or a transient failure) are re-tried far sooner.
_POSITIVE_TTL = 30 * 24 * 3600
_NEGATIVE_TTL = 7 * 24 * 3600
_DEFAULT_LIMIT = 20

_cache_lock = threading.Lock()
_cache: dict | None = None


def _cache_path() -> str:
    return os.path.join(utils.CACHE_DIR, "artist_top_tracks.json")


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
        logger.exception("artist_tracks: failed to write cache")


def _is_fresh(entry: dict) -> bool:
    age = time.time() - entry.get("fetched_at", 0)
    ttl = _POSITIVE_TTL if entry.get("items") else _NEGATIVE_TTL
    return age < ttl


def get_top_tracks(session, artist_id, limit: int = _DEFAULT_LIMIT) -> list:
    """Return the artist's top tracks as Track objects, cached to disk.

    On a fresh cache hit the tracks are rebuilt from stored JSON with no network
    request. On a miss the raw items are fetched, persisted, and parsed. Returns
    an empty list on any failure.
    """
    key = str(artist_id)
    entry = _load_cache().get(key)
    if entry is not None and _is_fresh(entry):
        items = entry.get("items", [])
    else:
        items = _fetch_items(session, artist_id, limit)
        with _cache_lock:
            _load_cache()[key] = {"items": items, "fetched_at": int(time.time())}
            _save_cache()

    tracks = []
    for obj in items:
        try:
            tracks.append(session.parse_track(obj))
        except Exception:
            logger.debug("artist_tracks: failed to parse a cached track", exc_info=True)
    return tracks


def _fetch_items(session, artist_id, limit: int) -> list:
    try:
        response = session.request.request(
            "GET",
            f"artists/{artist_id}/toptracks",
            {"limit": limit, "offset": 0},
        )
        items = response.json().get("items", [])
        logger.debug("artist_tracks: fetched %d top tracks for artist %s", len(items), artist_id)
        return items
    except Exception:
        logger.warning(
            "artist_tracks: failed to fetch top tracks for artist %s",
            artist_id,
            exc_info=True,
        )
        return []
