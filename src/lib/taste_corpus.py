# taste_corpus.py
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import logging
import os
import random
import threading
import time

from gi.repository import GLib
from tidalapi.media import Track
from tidalapi.mix import MixType, MixV2

from . import utils

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1
_TTL_SECONDS = 24 * 3600
_WEIGHT_CAP_MULTIPLIER = 3
_MAX_ARTISTS = 300
_MAX_TRACKS = 600

_MIX_TYPE_WEIGHT = {
    MixType.history_monthly: 3,
    MixType.history_yearly: 2,
    MixType.history_alltime: 2,
    MixType.daily: 2,
}
_MIX_TYPE_CAP = {
    MixType.history_monthly: 6,
    MixType.history_yearly: 3,
    MixType.history_alltime: 1,
    MixType.daily: None,  # unlimited — all daily mixes kept
}
_TRACKLIST_WEIGHT = 1
_PLAYLIST_USER_WEIGHT = 2
_PLAYLIST_FAV_WEIGHT = 1

_build_lock = threading.Lock()


def _cache_path() -> str:
    return os.path.join(utils.CACHE_DIR, "taste_corpus.json")


def _read_cache_if_fresh() -> dict | None:
    try:
        with open(_cache_path()) as f:
            data = json.load(f)
        age = time.time() - data.get("generated_at", 0)
        user_id = getattr(getattr(utils.session, "user", None), "id", None)
        if age < _TTL_SECONDS and (user_id is None or data.get("user_id") == user_id):
            return data
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return None


def _accumulate(artists: dict, tracks: dict, item: Track, weight: int) -> None:
    if not hasattr(item, "id") or not hasattr(item, "name"):
        return
    artist = getattr(item, "artist", None)
    a_valid = artist and hasattr(artist, "id") and hasattr(artist, "name")

    if a_valid:
        a_id = artist.id
        cap = weight * _WEIGHT_CAP_MULTIPLIER
        if a_id in artists:
            artists[a_id]["weight"] = min(artists[a_id]["weight"] + weight, cap)
        else:
            artists[a_id] = {"id": a_id, "name": artist.name, "weight": weight}

    t_id = item.id
    cap = weight * _WEIGHT_CAP_MULTIPLIER
    if t_id in tracks:
        tracks[t_id]["weight"] = min(tracks[t_id]["weight"] + weight, cap)
    else:
        album = getattr(item, "album", None)
        year = None
        if album:
            try:
                year = album.year
            except Exception:
                pass
        bpm_raw = getattr(item, "bpm", 0)
        tracks[t_id] = {
            "id": t_id,
            "name": item.name,
            "artist_id": artist.id if a_valid else None,
            "artist_name": artist.name if a_valid else None,
            "album": album.name if album and hasattr(album, "name") else None,
            "year": year,
            "popularity": getattr(item, "popularity", None),
            "duration": getattr(item, "duration", None),
            "bpm": bpm_raw if bpm_raw else None,
            "explicit": getattr(item, "explicit", False),
            "weight": weight,
        }


def _build(session, cancel_event=None) -> dict:
    artists: dict = {}
    tracks: dict = {}

    mix_counts: dict = {}
    try:
        for item in session.mixes():
            if cancel_event and cancel_event.is_set():
                break
            if isinstance(item, MixV2):
                mix_type = item.mix_type
                if mix_type is None or mix_type not in _MIX_TYPE_WEIGHT:
                    continue
                weight = _MIX_TYPE_WEIGHT[mix_type]
                cap = _MIX_TYPE_CAP[mix_type]
                count = mix_counts.get(mix_type, 0)
                if cap is not None and count >= cap:
                    continue
                mix_counts[mix_type] = count + 1
                try:
                    for track_item in session.mix(item.id).items():
                        if isinstance(track_item, Track):
                            _accumulate(artists, tracks, track_item, weight)
                except Exception:
                    logger.exception(
                        "taste_corpus: failed to fetch items for mix id=%s", item.id
                    )
            elif isinstance(item, Track):
                _accumulate(artists, tracks, item, _TRACKLIST_WEIGHT)
    except Exception:
        logger.exception("taste_corpus: failed to iterate session.mixes()")

    for playlist in list(utils.user_playlists)[:30]:
        if cancel_event and cancel_event.is_set():
            break
        try:
            for track_item in playlist.tracks(limit=30):
                if isinstance(track_item, Track):
                    _accumulate(artists, tracks, track_item, _PLAYLIST_USER_WEIGHT)
        except Exception:
            logger.exception(
                "taste_corpus: failed to load user playlist: %s",
                getattr(playlist, "name", "?"),
            )

    for playlist in list(utils.favourite_playlists)[:30]:
        if cancel_event and cancel_event.is_set():
            break
        try:
            for track_item in playlist.tracks(limit=30):
                if isinstance(track_item, Track):
                    _accumulate(artists, tracks, track_item, _PLAYLIST_FAV_WEIGHT)
        except Exception:
            logger.exception(
                "taste_corpus: failed to load favourite playlist: %s",
                getattr(playlist, "name", "?"),
            )

    sorted_artists = sorted(
        artists.values(), key=lambda x: x["weight"], reverse=True
    )[:_MAX_ARTISTS]
    sorted_tracks = sorted(
        tracks.values(), key=lambda x: x["weight"], reverse=True
    )[:_MAX_TRACKS]

    if not sorted_artists and not sorted_tracks:
        logger.warning(
            "taste_corpus: corpus is empty — no history mixes or playlist tracks found"
        )

    user_id = getattr(getattr(session, "user", None), "id", None)
    corpus = {
        "version": _CACHE_VERSION,
        "generated_at": int(time.time()),
        "user_id": user_id,
        "artists": sorted_artists,
        "tracks": sorted_tracks,
    }

    path = _cache_path()
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(corpus, f)
        os.replace(tmp_path, path)
        logger.info(
            "taste_corpus: wrote %d artists, %d tracks",
            len(sorted_artists),
            len(sorted_tracks),
        )
    except Exception:
        logger.exception("taste_corpus: failed to write corpus cache")

    return corpus


def ensure_corpus(session, cancel_event=None) -> dict:
    """Return corpus dict, building synchronously if stale/missing.

    Must be called from a worker thread — blocks on the build lock if a
    concurrent refresh is in flight.
    """
    cached = _read_cache_if_fresh()
    if cached:
        return cached
    with _build_lock:
        cached = _read_cache_if_fresh()
        if cached:
            return cached
        return _build(session, cancel_event=cancel_event)


def refresh_in_background(session, on_done=None) -> None:
    """Fire-and-forget corpus rebuild. Safe to call from any thread."""
    def _worker():
        if _build_lock.locked():
            return
        with _build_lock:
            if _read_cache_if_fresh():
                return
            _build(session)
        if on_done:
            GLib.idle_add(on_done)

    threading.Thread(target=_worker, daemon=True).start()


def sample_for_radio(
    corpus: dict,
    target_artists: int = 80,
    target_tracks: int = 120,
) -> dict:
    """Return a sampled subset: 70% top-weight anchor + 30% random tail."""

    def _sample(items: list, target: int) -> list:
        if len(items) <= target:
            return list(items)
        anchor_n = round(target * 0.7)
        tail_n = target - anchor_n
        anchor = items[:anchor_n]
        rest = items[anchor_n:]
        tail = random.sample(rest, min(tail_n, len(rest)))
        return anchor + tail

    return {
        "artists": _sample(corpus.get("artists", []), target_artists),
        "tracks": _sample(corpus.get("tracks", []), target_tracks),
    }
