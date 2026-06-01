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

_CACHE_VERSION = 2
_TTL_SECONDS = 24 * 3600
_WEIGHT_CAP_MULTIPLIER = 3
# Half-life (in days) for time-based decay of carried-over weights. Decay is
# applied globally on each rebuild based on elapsed wall-clock time, so it is
# independent of how often _build() actually runs.
_HALF_LIFE_DAYS = 45.0

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
        if data.get("version") != _CACHE_VERSION:
            return None
        age = time.time() - data.get("generated_at", 0)
        user_id = getattr(getattr(utils.session, "user", None), "id", None)
        if age < _TTL_SECONDS and (user_id is None or data.get("user_id") == user_id):
            return data
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return None


def _read_existing() -> dict | None:
    """Read the persisted corpus regardless of TTL, for incremental merge.

    Returns None if missing, unreadable, owned by a different user, or written
    by an older schema version (older corpora are discarded, not migrated).
    """
    try:
        with open(_cache_path()) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("version") != _CACHE_VERSION:
        return None
    user_id = getattr(getattr(utils.session, "user", None), "id", None)
    if user_id is not None and data.get("user_id") != user_id:
        return None
    return data


def _decay_factor(generated_at: float, now: float) -> float:
    """Exponential decay factor for the time elapsed since the last build."""
    elapsed_days = max(0.0, (now - generated_at) / 86400.0)
    return 0.5 ** (elapsed_days / _HALF_LIFE_DAYS)


def _merge_entries(
    existing: list,
    fresh: dict,
    now: int,
    factor: float,
    preserve_keys: tuple,
) -> dict:
    """Merge freshly-accumulated entries onto time-decayed existing ones.

    Existing weights are scaled by `factor`; this run's contributions are added
    on top (an exponential moving average). Nothing is pruned or capped — old
    entries simply decay toward (but never reach) zero and sink in the ranking.
    `preserve_keys` are carried over from the existing entry when it is re-seen
    this run (e.g. MusicBrainz genres/tags that the fresh fetch doesn't carry).
    """
    merged: dict = {}
    for entry in existing:
        eid = entry.get("id")
        if eid is None:
            continue
        decayed = round(entry.get("weight", 0) * factor, 4)
        merged[eid] = {**entry, "weight": decayed}
        merged[eid].setdefault("first_seen", now)

    for fid, fresh_entry in fresh.items():
        old = merged.get(fid)
        if old is not None:
            carried = {k: old[k] for k in preserve_keys if k in old}
            merged[fid] = {
                **fresh_entry,
                **carried,
                "weight": round(old["weight"] + fresh_entry["weight"], 4),
                "first_seen": old.get("first_seen", now),
                "last_seen": now,
            }
        else:
            merged[fid] = {**fresh_entry, "first_seen": now, "last_seen": now}
    return merged


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

    now = int(time.time())
    existing = _read_existing()
    if existing:
        factor = _decay_factor(existing.get("generated_at", now), now)
        logger.debug(
            "taste_corpus: merging onto existing corpus (decay factor %.4f)", factor
        )
        merged_artists = _merge_entries(
            existing.get("artists", []), artists, now, factor,
            preserve_keys=("genres", "tags", "mbid"),
        )
        merged_tracks = _merge_entries(
            existing.get("tracks", []), tracks, now, factor, preserve_keys=(),
        )
    else:
        merged_artists = {
            aid: {**a, "first_seen": now, "last_seen": now}
            for aid, a in artists.items()
        }
        merged_tracks = {
            tid: {**t, "first_seen": now, "last_seen": now}
            for tid, t in tracks.items()
        }

    # No pruning or size cap — weights decay over time but entries persist.
    sorted_artists = sorted(
        merged_artists.values(), key=lambda x: x["weight"], reverse=True
    )
    sorted_tracks = sorted(
        merged_tracks.values(), key=lambda x: x["weight"], reverse=True
    )

    if not sorted_artists and not sorted_tracks:
        logger.warning(
            "taste_corpus: corpus is empty — no history mixes or playlist tracks found"
        )

    user_id = getattr(getattr(session, "user", None), "id", None)
    corpus = {
        "version": _CACHE_VERSION,
        "generated_at": now,
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
