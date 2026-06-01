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

from . import musicbrainz, utils

logger = logging.getLogger(__name__)

_CACHE_VERSION = 2
_TTL_SECONDS = 24 * 3600
_WEIGHT_CAP_MULTIPLIER = 3
# Half-life (in days) for time-based decay of carried-over weights. Decay is
# applied globally on each rebuild based on elapsed wall-clock time, so it is
# independent of how often _build() actually runs.
_HALF_LIFE_DAYS = 45.0
# Max artists to enrich via MusicBrainz per rebuild. Bounded so a rebuild adds
# at most ~_MB_ENRICH_PER_RUN * 2 throttled requests; the corpus gets fully
# tagged over several rebuilds, and tags then persist across merges.
_MB_ENRICH_PER_RUN = 30

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


def _enrich_artists(artists: dict, artist_isrc: dict, cancel_event=None) -> None:
    """Attach MusicBrainz genres/tags to artist entries in place.

    Cached genres (from previous runs) are applied to every artist for free;
    the network budget (_MB_ENRICH_PER_RUN) is spent only on the highest-weight
    not-yet-tagged artists, so the most relevant ones are tagged first.
    """
    budget = _MB_ENRICH_PER_RUN
    fetched = 0
    tagged = 0
    ordered = sorted(artists.values(), key=lambda x: x["weight"], reverse=True)
    logger.info(
        "taste_corpus: MusicBrainz enrichment starting (%d artists, budget=%d)",
        len(ordered), budget,
    )
    for entry in ordered:
        if cancel_event and cancel_event.is_set():
            break
        if entry.get("genres") or entry.get("tags"):
            continue  # already tagged in a prior run (carried through merge)
        info = musicbrainz.get_cached(entry["id"])
        if info is None:
            if budget <= 0:
                continue
            info = musicbrainz.enrich_artist(
                entry["id"],
                entry.get("name", ""),
                artist_isrc.get(entry["id"]),
                cancel_event,
            )
            budget -= 1
            fetched += 1
        if info.get("genres"):
            entry["genres"] = info["genres"]
        if info.get("tags"):
            entry["tags"] = info["tags"]
        if info.get("genres") or info.get("tags"):
            tagged += 1
    logger.info(
        "taste_corpus: MusicBrainz enrichment done (%d newly fetched, %d artists tagged)",
        fetched, tagged,
    )


def _accumulate(
    artists: dict,
    tracks: dict,
    item: Track,
    weight: int,
    artist_isrc: dict | None = None,
) -> None:
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
        # Remember one ISRC per artist — the reliable join key for MusicBrainz.
        if artist_isrc is not None and a_id not in artist_isrc:
            isrc = getattr(item, "isrc", None)
            if isrc:
                artist_isrc[a_id] = isrc

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


def _build(session, cancel_event=None, use_musicbrainz=True) -> dict:
    logger.info("taste_corpus: build starting (musicbrainz=%s)", use_musicbrainz)
    artists: dict = {}
    tracks: dict = {}
    artist_isrc: dict = {}

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
                            _accumulate(
                                artists, tracks, track_item, weight, artist_isrc
                            )
                except Exception:
                    logger.exception(
                        "taste_corpus: failed to fetch items for mix id=%s", item.id
                    )
            elif isinstance(item, Track):
                _accumulate(artists, tracks, item, _TRACKLIST_WEIGHT, artist_isrc)
    except Exception:
        logger.exception("taste_corpus: failed to iterate session.mixes()")

    for playlist in list(utils.user_playlists)[:30]:
        if cancel_event and cancel_event.is_set():
            break
        try:
            for track_item in playlist.tracks(limit=30):
                if isinstance(track_item, Track):
                    _accumulate(
                        artists, tracks, track_item,
                        _PLAYLIST_USER_WEIGHT, artist_isrc,
                    )
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
                    _accumulate(
                        artists, tracks, track_item,
                        _PLAYLIST_FAV_WEIGHT, artist_isrc,
                    )
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

    # Background MusicBrainz enrichment: tag the highest-weight artists that
    # aren't tagged yet, applying any previously-cached tags for free.
    if use_musicbrainz and not (cancel_event and cancel_event.is_set()):
        try:
            _enrich_artists(merged_artists, artist_isrc, cancel_event)
        except Exception:
            logger.exception("taste_corpus: MusicBrainz enrichment pass failed")

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


def ensure_corpus(session, cancel_event=None, use_musicbrainz=True) -> dict:
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
        return _build(
            session, cancel_event=cancel_event, use_musicbrainz=use_musicbrainz
        )


def refresh_in_background(
    session, on_done=None, use_musicbrainz=True, force=False
) -> None:
    """Fire-and-forget corpus rebuild. Safe to call from any thread.

    A manual refresh (force=True) rebuilds even when the cache is still fresh;
    background refreshes (force=False) skip the rebuild if a fresh cache exists.
    """
    def _worker():
        if _build_lock.locked():
            logger.info("taste_corpus: refresh skipped — a build is already running")
            return
        with _build_lock:
            if not force and _read_cache_if_fresh():
                logger.info("taste_corpus: refresh skipped — cache still fresh")
                if on_done:
                    GLib.idle_add(on_done)
                return
            logger.info(
                "taste_corpus: rebuilding corpus (force=%s, musicbrainz=%s)",
                force, use_musicbrainz,
            )
            _build(session, use_musicbrainz=use_musicbrainz)
        if on_done:
            GLib.idle_add(on_done)

    threading.Thread(target=_worker, daemon=True).start()


def _cap_per_artist(tracks: list, max_per_artist: int) -> list:
    """Keep at most `max_per_artist` (weight-sorted) tracks per artist."""
    if not max_per_artist:
        return list(tracks)
    capped: list = []
    counts: dict = {}
    for t in tracks:
        aid = t.get("artist_id")
        if aid is not None:
            if counts.get(aid, 0) >= max_per_artist:
                continue
            counts[aid] = counts.get(aid, 0) + 1
        capped.append(t)
    return capped


def _genre_prerank(items: list, prompt: str, genres_of) -> list:
    """Float entries whose genres/tags appear in the prompt to the front.

    A boost, not a filter: when the prompt names no known genre, order is
    unchanged. Matched entries land in the sample's top-weight anchor.
    """
    if not prompt:
        return items
    p = prompt.lower()
    matched, rest = [], []
    for it in items:
        gens = genres_of(it) or []
        if any(g and g.lower() in p for g in gens):
            matched.append(it)
        else:
            rest.append(it)
    if not matched:
        return items
    logger.debug("taste_corpus: genre pre-rank floated %d entries", len(matched))
    return matched + rest


def sample_for_radio(
    corpus: dict,
    prompt: str = "",
    target_artists: int = 80,
    target_tracks: int = 120,
    max_tracks_per_artist: int = 2,
) -> dict:
    """Return a sampled subset: 70% top-weight anchor + 30% random tail.

    Tracks are first capped per-artist (so a heavily-played artist can't flood
    the list) and both lists are genre-pre-ranked against the prompt.
    """

    def _sample(items: list, target: int) -> list:
        if len(items) <= target:
            return list(items)
        anchor_n = round(target * 0.7)
        tail_n = target - anchor_n
        anchor = items[:anchor_n]
        rest = items[anchor_n:]
        tail = random.sample(rest, min(tail_n, len(rest)))
        return anchor + tail

    artists = corpus.get("artists", [])
    tracks = _cap_per_artist(corpus.get("tracks", []), max_tracks_per_artist)

    artist_genres = {
        a["id"]: (a.get("genres") or []) + (a.get("tags") or [])
        for a in artists if "id" in a
    }
    artists = _genre_prerank(
        artists, prompt, lambda a: (a.get("genres") or []) + (a.get("tags") or [])
    )
    tracks = _genre_prerank(
        tracks, prompt, lambda t: artist_genres.get(t.get("artist_id"), [])
    )

    return {
        "artists": _sample(artists, target_artists),
        "tracks": _sample(tracks, target_tracks),
    }
