"""Shared shaping of API payloads into compact records and two output formats."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional

_ISO_DURATION = re.compile(
    r"^P(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$"
)


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


def iso_duration_to_seconds(value: Optional[str]) -> Optional[int]:
    """'PT1H2M3S' -> 3723. Returns None for live streams or unparseable input."""
    if not value:
        return None
    match = _ISO_DURATION.match(value)
    if not match:
        return None
    parts = {k: int(v) for k, v in match.groupdict(default="0").items()}
    return parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]


def humanize_seconds(seconds: Optional[int]) -> str:
    if seconds is None:
        return "live/unknown"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def compact_int(value: Optional[Any]) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def video_record(item: Dict[str, Any], *, position: Optional[int] = None) -> Dict[str, Any]:
    """Normalize a videos.list / search.list / playlistItems.list item.

    Returns a flat dict: video_id, title, channel, channel_id, published_at,
    duration_seconds, view_count, like_count, url, description (truncated),
    and playlist_item_id / position when the source was a playlist.
    """
    snippet = item.get("snippet", {}) or {}
    content = item.get("contentDetails", {}) or {}
    stats = item.get("statistics", {}) or {}
    kind_id = item.get("id")

    if isinstance(kind_id, dict):  # search.list
        video_id = kind_id.get("videoId") or kind_id.get("playlistId") or kind_id.get("channelId")
        playlist_item_id = None
    elif "resourceId" in snippet:  # playlistItems.list
        video_id = snippet["resourceId"].get("videoId")
        playlist_item_id = kind_id
    else:  # videos.list
        video_id = kind_id
        playlist_item_id = None

    duration = iso_duration_to_seconds(content.get("duration"))
    description = (snippet.get("description") or "").strip()

    record: Dict[str, Any] = {
        "video_id": video_id,
        "title": snippet.get("title"),
        "channel": snippet.get("videoOwnerChannelTitle") or snippet.get("channelTitle"),
        "channel_id": snippet.get("videoOwnerChannelId") or snippet.get("channelId"),
        "published_at": content.get("videoPublishedAt") or snippet.get("publishedAt"),
        "duration_seconds": duration,
        "view_count": compact_int(stats.get("viewCount")),
        "like_count": compact_int(stats.get("likeCount")),
        "url": f"https://www.youtube.com/watch?v={video_id}" if video_id else None,
        "description": description[:500] + ("…" if len(description) > 500 else ""),
    }
    if playlist_item_id:
        record["playlist_item_id"] = playlist_item_id
        record["position"] = snippet.get("position", position)
    return {k: v for k, v in record.items() if v is not None}


def render_videos(records: List[Dict[str, Any]], title: str) -> str:
    """Compact markdown table: one row per video, IDs kept so tools can chain."""
    if not records:
        return f"### {title}\n\n_No results._"
    lines = [
        f"### {title} ({len(records)})",
        "",
        "| # | Video | Channel | Length | Views | ID |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for idx, rec in enumerate(records, start=1):
        views = rec.get("view_count")
        lines.append(
            "| {i} | [{title}]({url}) | {ch} | {dur} | {views} | `{vid}` |".format(
                i=idx,
                title=(rec.get("title") or "untitled").replace("|", "\\|")[:80],
                url=rec.get("url", ""),
                ch=(rec.get("channel") or "-").replace("|", "\\|")[:30],
                dur=humanize_seconds(rec.get("duration_seconds")),
                views=f"{views:,}" if views is not None else "-",
                vid=rec.get("video_id", "-"),
            )
        )
    return "\n".join(lines)


def envelope(
    records: List[Dict[str, Any]],
    *,
    fmt: ResponseFormat,
    title: str,
    next_page_token: Optional[str] = None,
    total: Optional[int] = None,
    quota_spent: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """One consistent envelope for every list-returning tool."""
    if fmt is ResponseFormat.JSON:
        payload: Dict[str, Any] = {
            "title": title,
            "count": len(records),
            "total_available": total,
            "items": records,
            "has_more": bool(next_page_token),
            "next_page_token": next_page_token,
        }
        if quota_spent is not None:
            payload["quota_units_spent"] = quota_spent
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    out = [render_videos(records, title)]
    footer: List[str] = []
    if total is not None:
        footer.append(f"total available: {total}")
    if next_page_token:
        footer.append(f'more pages: pass `page_token="{next_page_token}"`')
    if quota_spent is not None:
        footer.append(f"quota spent this call: {quota_spent} units")
    if extra:
        footer.extend(f"{k}: {v}" for k, v in extra.items())
    if footer:
        out.extend(["", "_" + " · ".join(footer) + "_"])
    return "\n".join(out)
