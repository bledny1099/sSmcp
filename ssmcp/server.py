"""sSmcp (socialSimple MCP) — unified remote & local MCP server for YouTube, TikTok, Instagram, and Threads."""

from __future__ import annotations

import functools
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__
from .client import QUOTA_COSTS, QuotaLedger, YouTubeClient, YouTubeError
from .formatting import ResponseFormat, envelope, video_record
from .instagram import (
    InstagramError,
    get_instagram_media,
    get_instagram_transcript,
    get_instagram_user_posts,
)
from .threads import (
    ThreadsError,
    get_threads_post,
    get_threads_user_posts,
    search_threads_posts,
)
from .tiktok import (
    TikTokError,
    batch_tiktok_videos,
    get_tiktok_transcript,
    get_tiktok_user_videos,
    get_tiktok_video,
    publish_tiktok_video,
)
from .transcripts import (
    DEFAULT_CHUNK_CHARS,
    TranscriptError,
    default_languages,
    extract_video_id,
    fetch_transcript,
    list_languages,
    stitch,
)
from .whisper_transcriber import TranscriptionError

SERVER_NAME = "ssmcp"
MAX_IDS_PER_CALL = 50

mcp = MCPServer(
    SERVER_NAME,
    title="sSmcp — socialSimple MCP (YouTube + TikTok + Instagram + Threads)",
    version=__version__,
    instructions=(
        "socialSimple MCP (sSmcp): unified social media MCP server for YouTube, TikTok, Instagram, and Threads. "
        "YouTube: search, video metadata, transcripts (captions + Whisper + keyframe OCR), and playlist management. "
        "TikTok: single and batch video/photo-carousel metadata, metrics, transcripts (captions + Whisper + "
        "screenshot/slide OCR), creator video feeds, and custom-domain media publishing. "
        "Instagram: Reels, video & photo posts, multi-slide carousels, metrics, direct CDN streams, "
        "and Whisper + visual OCR transcripts with anonymous embed fallback. "
        "Threads: full post text, author info, metrics, images, videos, replies, search, and media OCR without login."
    ),
)

# Resilient caller: allow clients to pass arguments flat or wrapped inside {"params": {...}}
_orig_tool_manager_call = mcp._tool_manager.call_tool


async def _resilient_call_tool(name: str, arguments: Dict[str, Any], context=None, **kwargs):
    tool = mcp._tool_manager.get_tool(name)
    if tool and tool.parameters and "params" in tool.parameters.get("properties", {}):
        if "params" not in arguments:
            arguments = {"params": arguments}
    return await _orig_tool_manager_call(name, arguments, context, **kwargs)


mcp._tool_manager.call_tool = _resilient_call_tool

_ledger = QuotaLedger()
_client: Optional[YouTubeClient] = None


def client() -> YouTubeClient:
    global _client
    if _client is None:
        _client = YouTubeClient.from_env(ledger=_ledger)
    return _client


def watch_later_id(explicit: Optional[str]) -> str:
    """Resolve the playlist to act on, rejecting the unusable native WL/HL ids."""
    playlist_id = explicit or os.getenv("YT_WATCH_LATER_PLAYLIST_ID", "")
    if not playlist_id:
        raise YouTubeError(
            "No playlist_id given and YT_WATCH_LATER_PLAYLIST_ID is unset. Call "
            "youtube_list_playlists to find the ID of your migrated Watch Later playlist."
        )
    if playlist_id.upper() in {"WL", "HL"}:
        raise YouTubeError(
            "Playlist IDs 'WL' (Watch Later) and 'HL' (history) cannot be read or written "
            "through the YouTube Data API because Google disabled that in 2016. Use the ID of the "
            "regular playlist you migrated Watch Later into."
        )
    return playlist_id


def guard(func):
    """Return tool errors as text instead of raising protocol-level failures."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (
            YouTubeError,
            TranscriptError,
            TikTokError,
            InstagramError,
            ThreadsError,
            TranscriptionError,
        ) as exc:
            return f"Error: {exc}"
        except Exception as exc:  # pragma: no cover
            return f"Error: unexpected {type(exc).__name__} in {func.__name__}: {exc}"

    return wrapper


async def hydrate(video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch statistics + duration for up to 50 IDs in one 1-unit call."""
    ids = [vid for vid in video_ids if vid][:MAX_IDS_PER_CALL]
    if not ids:
        return {}
    data = await client().call(
        "videos.list",
        params={"part": "snippet,contentDetails,statistics", "id": ",".join(ids), "maxResults": 50},
    )
    return {item["id"]: item for item in data.get("items", [])}


# --------------------------------------------------------------------------- #
# YouTube: Search, Videos, Playlists, Transcripts, Quota
# --------------------------------------------------------------------------- #


class SearchInput(BaseModel):
    """Input for youtube_search."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(..., description="Free-text search query, e.g. 'rust async runtime deep dive'", min_length=1, max_length=200)
    limit: int = Field(default=10, description="Number of results to return (1-50)", ge=1, le=50)
    order: str = Field(default="relevance", description="Sort order: relevance, date, viewCount, rating, title")
    channel_id: Optional[str] = Field(default=None, description="Restrict to one channel, e.g. 'UCXuqSBlHAE6Xw-yeJA0Tunw'")
    published_after: Optional[str] = Field(default=None, description="RFC3339 lower bound, e.g. '2026-01-01T00:00:00Z'")
    published_before: Optional[str] = Field(default=None, description="RFC3339 upper bound, e.g. '2026-06-01T00:00:00Z'")
    video_duration: Optional[str] = Field(default=None, description="Filter by length: 'short' (<4m), 'medium' (4-20m), 'long' (>20m)")
    language: Optional[str] = Field(default=None, description="ISO-639-1 relevance hint, e.g. 'ru' or 'en'")
    with_stats: bool = Field(default=True, description="Add view/like counts and exact duration via one extra 1-unit call")
    page_token: Optional[str] = Field(default=None, description="next_page_token from a previous call")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")

    @field_validator("order")
    @classmethod
    def _valid_order(cls, value: str) -> str:
        allowed = {"relevance", "date", "viewCount", "rating", "title"}
        if value not in allowed:
            raise ValueError(f"order must be one of {sorted(allowed)}")
        return value

    @field_validator("video_duration")
    @classmethod
    def _valid_duration(cls, value: Optional[str]) -> Optional[str]:
        if value and value not in {"short", "medium", "long", "any"}:
            raise ValueError("video_duration must be 'short', 'medium', 'long' or 'any'")
        return value


@mcp.tool(
    name="youtube_search",
    annotations=ToolAnnotations(title="Search YouTube videos", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),
)
@guard
async def youtube_search(params: SearchInput) -> str:
    """Search YouTube for videos by keyword, with optional channel, date and length filters.

    Each call costs 100 of the 10,000 daily quota units. To read a known playlist or known
    video IDs, use youtube_list_playlist_items or youtube_get_videos instead (1 unit each).
    """
    query: Dict[str, Any] = {
        "part": "snippet",
        "q": params.query,
        "type": "video",
        "maxResults": params.limit,
        "order": params.order,
    }
    for key, value in (
        ("channelId", params.channel_id),
        ("publishedAfter", params.published_after),
        ("publishedBefore", params.published_before),
        ("videoDuration", params.video_duration),
        ("relevanceLanguage", params.language),
        ("pageToken", params.page_token),
    ):
        if value:
            query[key] = value

    data = await client().call("search.list", params=query)
    records = [video_record(item) for item in data.get("items", [])]
    spent = QUOTA_COSTS["search.list"]

    if params.with_stats and records:
        details = await hydrate([r["video_id"] for r in records])
        spent += QUOTA_COSTS["videos.list"]
        for rec in records:
            item = details.get(rec["video_id"])
            if item:
                rec.update({k: v for k, v in video_record(item).items() if k not in {"description"}})

    return envelope(
        records,
        fmt=params.response_format,
        title=f"Search: {params.query}",
        next_page_token=data.get("nextPageToken"),
        total=data.get("pageInfo", {}).get("totalResults"),
        quota_spent=spent,
    )


class GetVideosInput(BaseModel):
    """Input for youtube_get_videos."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video_ids: List[str] = Field(..., description="Up to 50 video IDs or URLs, e.g. ['dQw4w9WgXcQ']", min_length=1, max_length=50)
    full_description: bool = Field(default=False, description="Return the untruncated description instead of the first 500 chars")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="youtube_get_videos",
    annotations=ToolAnnotations(title="Get YouTube video details", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),
)
@guard
async def youtube_get_videos(params: GetVideosInput) -> str:
    """Fetch title, channel, duration, view/like counts and description for known video IDs.

    Costs 1 quota unit for the whole batch (up to 50 IDs or URLs).
    """
    ids = [extract_video_id(raw) for raw in params.video_ids]
    details = await hydrate(ids)
    records = []
    for vid in ids:
        item = details.get(vid)
        if not item:
            records.append({"video_id": vid, "title": "(unavailable: private, deleted or wrong ID)"})
            continue
        rec = video_record(item)
        if params.full_description:
            rec["description"] = (item.get("snippet", {}) or {}).get("description", "")
        records.append(rec)

    return envelope(
        records,
        fmt=params.response_format,
        title="Video details",
        quota_spent=QUOTA_COSTS["videos.list"],
    )


class ListPlaylistsInput(BaseModel):
    """Input for youtube_list_playlists."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    limit: int = Field(default=50, description="Number of playlists to return (1-50)", ge=1, le=50)
    page_token: Optional[str] = Field(default=None, description="next_page_token from a previous call")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="youtube_list_playlists",
    annotations=ToolAnnotations(title="List my YouTube playlists", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),
)
@guard
async def youtube_list_playlists(params: ListPlaylistsInput) -> str:
    """List the authenticated user's own playlists with their IDs and item counts."""
    data = await client().call(
        "playlists.list",
        params={
            "part": "snippet,contentDetails,status",
            "mine": "true",
            "maxResults": params.limit,
            **({"pageToken": params.page_token} if params.page_token else {}),
        },
    )
    items = [
        {
            "playlist_id": p["id"],
            "title": (p.get("snippet") or {}).get("title"),
            "item_count": (p.get("contentDetails") or {}).get("itemCount"),
            "privacy": (p.get("status") or {}).get("privacyStatus"),
            "description": ((p.get("snippet") or {}).get("description") or "")[:200],
            "url": f"https://www.youtube.com/playlist?list={p['id']}",
        }
        for p in data.get("items", [])
    ]

    if params.response_format is ResponseFormat.JSON:
        return json.dumps(
            {
                "count": len(items),
                "items": items,
                "has_more": bool(data.get("nextPageToken")),
                "next_page_token": data.get("nextPageToken"),
                "configured_watch_later_id": os.getenv("YT_WATCH_LATER_PLAYLIST_ID") or None,
            },
            ensure_ascii=False,
            indent=2,
        )

    lines = ["### My playlists", "", "| Title | Items | Privacy | ID |", "| --- | --- | --- | --- |"]
    for p in items:
        lines.append(
            f"| {(p['title'] or '-')[:60]} | {p['item_count']} | {p['privacy']} | `{p['playlist_id']}` |"
        )
    configured = os.getenv("YT_WATCH_LATER_PLAYLIST_ID")
    if configured:
        lines += ["", f"_Default playlist for the Watch Later tools: `{configured}`_"]
    return "\n".join(lines)


class ListPlaylistItemsInput(BaseModel):
    """Input for youtube_list_playlist_items."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    playlist_id: Optional[str] = Field(default=None, description="Playlist ID; omit to use the configured Watch Later playlist")
    limit: int = Field(default=25, description="Number of items to return (1-50)", ge=1, le=50)
    page_token: Optional[str] = Field(default=None, description="next_page_token from a previous call")
    with_stats: bool = Field(default=True, description="Add view/like counts and duration via one extra 1-unit call")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="youtube_list_playlist_items",
    annotations=ToolAnnotations(title="Read a YouTube playlist", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),
)
@guard
async def youtube_list_playlist_items(params: ListPlaylistItemsInput) -> str:
    """Read the videos in a playlist (by default the user's Watch Later replacement)."""
    playlist_id = watch_later_id(params.playlist_id)
    data = await client().call(
        "playlistItems.list",
        params={
            "part": "snippet,contentDetails",
            "playlistId": playlist_id,
            "maxResults": params.limit,
            **({"pageToken": params.page_token} if params.page_token else {}),
        },
    )
    records = [video_record(item, position=i) for i, item in enumerate(data.get("items", []))]
    spent = QUOTA_COSTS["playlistItems.list"]

    if params.with_stats and records:
        details = await hydrate([r["video_id"] for r in records if r.get("video_id")])
        spent += QUOTA_COSTS["videos.list"]
        for rec in records:
            item = details.get(rec.get("video_id"))
            if item:
                enriched = video_record(item)
                rec["duration_seconds"] = enriched.get("duration_seconds")
                rec["view_count"] = enriched.get("view_count")
                rec["like_count"] = enriched.get("like_count")

    return envelope(
        records,
        fmt=params.response_format,
        title=f"Playlist {playlist_id}",
        next_page_token=data.get("nextPageToken"),
        total=data.get("pageInfo", {}).get("totalResults"),
        quota_spent=spent,
    )


class AddToPlaylistInput(BaseModel):
    """Input for youtube_add_to_playlist."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video_ids: List[str] = Field(..., description="Video IDs or URLs to append, e.g. ['dQw4w9WgXcQ']", min_length=1, max_length=20)
    playlist_id: Optional[str] = Field(default=None, description="Target playlist ID; omit to use the configured Watch Later playlist")


@mcp.tool(
    name="youtube_add_to_playlist",
    annotations=ToolAnnotations(title="Add videos to a playlist", read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True),
)
@guard
async def youtube_add_to_playlist(params: AddToPlaylistInput) -> str:
    """Append videos to a playlist (costs 50 quota units per video)."""
    playlist_id = watch_later_id(params.playlist_id)
    added: List[Dict[str, str]] = []
    failed: List[Dict[str, str]] = []

    for raw in params.video_ids:
        try:
            video_id = extract_video_id(raw)
            result = await client().call(
                "playlistItems.insert",
                params={"part": "snippet"},
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {"kind": "youtube#video", "videoId": video_id},
                    }
                },
            )
            added.append({"video_id": video_id, "playlist_item_id": result.get("id", "")})
        except (YouTubeError, TranscriptError) as exc:
            failed.append({"video_id": raw, "error": str(exc)})

    return json.dumps(
        {
            "playlist_id": playlist_id,
            "added": added,
            "failed": failed,
            "quota_units_spent": len(added) * QUOTA_COSTS["playlistItems.insert"],
        },
        ensure_ascii=False,
        indent=2,
    )


class RemoveFromPlaylistInput(BaseModel):
    """Input for youtube_remove_from_playlist."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    playlist_item_ids: List[str] = Field(
        ...,
        description="playlist_item_id values from youtube_list_playlist_items (NOT video IDs)",
        min_length=1,
        max_length=20,
    )


@mcp.tool(
    name="youtube_remove_from_playlist",
    annotations=ToolAnnotations(title="Remove items from a playlist", read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=True),
)
@guard
async def youtube_remove_from_playlist(params: RemoveFromPlaylistInput) -> str:
    """Remove items from a playlist by playlist_item_id (costs 50 quota units per item)."""
    removed: List[str] = []
    failed: List[Dict[str, str]] = []

    for item_id in params.playlist_item_ids:
        try:
            await client().call("playlistItems.delete", params={"id": item_id})
            removed.append(item_id)
        except YouTubeError as exc:
            failed.append({"playlist_item_id": item_id, "error": str(exc)})

    return json.dumps(
        {
            "removed": removed,
            "failed": failed,
            "quota_units_spent": len(removed) * QUOTA_COSTS["playlistItems.delete"],
        },
        ensure_ascii=False,
        indent=2,
    )


class TranscriptInput(BaseModel):
    """Input for youtube_get_transcript."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video: str = Field(..., description="Video ID or URL, e.g. 'dQw4w9WgXcQ' or a full watch link")
    languages: Optional[List[str]] = Field(default=None, description="Preferred caption languages in order, e.g. ['ru','en']", max_length=5)
    with_timestamps: bool = Field(default=False, description="Prefix each line with [m:ss]")
    include_ocr: bool = Field(default=False, description="Also extract on-screen text from video keyframes via OCR (automatically used when captions are missing or music-only)")
    offset: int = Field(default=0, description="Character offset to resume from; use next_offset from the previous call", ge=0)
    chunk_chars: int = Field(default=DEFAULT_CHUNK_CHARS, description="Characters per chunk (1000-40000)", ge=1000, le=40000)


@mcp.tool(
    name="youtube_get_transcript",
    annotations=ToolAnnotations(title="Get a YouTube transcript", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),
)
@guard
async def youtube_get_transcript(params: TranscriptInput) -> str:
    """Fetch a YouTube video's captions and/or on-screen OCR text, paged by character offset.

    Costs no API quota: reads caption tracks directly and falls back to Whisper
    audio transcription + visual keyframe OCR for uncaptioned or screenshot/slide videos.
    """
    video_id = extract_video_id(params.video)
    segments, language = await fetch_transcript(
        video_id,
        default_languages(params.languages),
        include_ocr=params.include_ocr,
    )
    chunk = stitch(
        segments,
        with_timestamps=params.with_timestamps,
        offset=params.offset,
        chunk_chars=params.chunk_chars,
    )
    return json.dumps({"video_id": video_id, "language": language, **chunk}, ensure_ascii=False, indent=2)


class TranscriptLanguagesInput(BaseModel):
    """Input for youtube_list_transcript_languages."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video: str = Field(..., description="Video ID or URL")


@mcp.tool(
    name="youtube_list_transcript_languages",
    annotations=ToolAnnotations(title="List caption tracks", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True),
)
@guard
async def youtube_list_transcript_languages(params: TranscriptLanguagesInput) -> str:
    """List the caption tracks available for a YouTube video, manual and auto-generated."""
    video_id = extract_video_id(params.video)
    tracks = await list_languages(video_id)
    return json.dumps({"video_id": video_id, "tracks": tracks}, ensure_ascii=False, indent=2)


@mcp.tool(
    name="youtube_quota_status",
    annotations=ToolAnnotations(title="Check API quota budget", read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False),
)
async def youtube_quota_status() -> str:
    """Report this server's estimated YouTube Data API quota spend for today."""
    return json.dumps(_ledger.snapshot(), ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# TikTok Tools (Video Info, Batch Info, Transcript + OCR, User Feed, Publish)
# --------------------------------------------------------------------------- #


class TikTokVideoInput(BaseModel):
    """Input for tiktok_get_video."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video: str = Field(
        ...,
        description="TikTok video or photo URL (e.g. https://www.tiktok.com/@user/video/12345 or vt.tiktok.com/...) or ID",
        min_length=1,
    )
    fast: bool = Field(
        default=False,
        description="When True, uses lightweight public oEmbed for an instant response (omits extended statistics and stream URLs)",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="tiktok_get_video",
    annotations=ToolAnnotations(
        title="Get TikTok video or photo-carousel info",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
)
@guard
async def tiktok_get_video(params: TikTokVideoInput) -> str:
    """Fetch complete metadata for a TikTok video or Photo Mode slideshow.

    Returns creator name, handle, description, views, likes, comments, shares,
    saves, sound/music info, direct stream URLs, and carousel slide image URLs.
    """
    data = await get_tiktok_video(params.video, fast=params.fast)
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(data, ensure_ascii=False, indent=2)

    views = f"{data.get('view_count', 0):,}" if data.get("view_count") is not None else "N/A"
    likes = f"{data.get('like_count', 0):,}" if data.get("like_count") is not None else "N/A"
    comments = f"{data.get('comment_count', 0):,}" if data.get("comment_count") is not None else "N/A"
    shares = f"{data.get('repost_count', 0):,}" if data.get("repost_count") is not None else "N/A"
    saves = f"{data.get('save_count', 0):,}" if data.get("save_count") is not None else "N/A"

    uploader_md = (
        f"[@{data.get('uploader_id') or data.get('uploader')}]({data.get('uploader_url')})"
        if data.get("uploader_url")
        else data.get("uploader")
    )
    sound_info = (
        f"{data.get('track')} - {data.get('artist')}"
        if data.get("track") and data.get("artist")
        else (data.get("track") or "None")
    )
    subs_list = ", ".join(data.get("subtitles_available", [])) or "None"

    duration_val = data.get("duration")
    duration_str = data.get("duration_string") or (f"{duration_val}s" if duration_val else "N/A")
    media_type = data.get("media_type", "video").upper()

    lines = [
        f"### TikTok [{media_type}]: [{data.get('title') or 'Post'}]({data.get('url')})",
        "",
        f"- **Creator:** {uploader_md} ({data.get('uploader')})",
        f"- **Stats:** {views} views · {likes} likes · {comments} comments · {shares} shares · {saves} saves",
        f"- **Duration:** {duration_str}",
        f"- **Music:** {sound_info}",
        f"- **Subtitles:** {subs_list}",
    ]
    if data.get("direct_video_url"):
        lines.append(f"- **Direct Stream:** [Open Video Stream]({data.get('direct_video_url')})")
    if data.get("direct_audio_url"):
        lines.append(f"- **Direct Audio:** [Open Audio Stream]({data.get('direct_audio_url')})")
    if data.get("thumbnail"):
        lines.append(f"- **Cover Image:** [View Image]({data.get('thumbnail')})")

    if data.get("is_carousel") and data.get("carousel_items"):
        items = data["carousel_items"]
        lines.extend([
            "",
            f"#### Carousel Slides ({len(items)} items)",
        ])
        for item in items:
            item_type = item.get("type", "item").capitalize()
            dur_info = f" ({item.get('duration')}s)" if item.get("duration") else ""
            link_str = f"[Open {item_type}]({item.get('url')})" if item.get("url") else "No direct link"
            lines.append(f"- Slide {item.get('index')}: **{item_type}**{dur_info}: {link_str}")

    return "\n".join(lines)


class TikTokBatchInput(BaseModel):
    """Input for tiktok_batch_videos."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    urls: List[str] = Field(
        ...,
        description="Up to 20 TikTok video or photo URLs to fetch concurrently",
        min_length=1,
        max_length=20,
    )
    fast: bool = Field(
        default=True,
        description="Use lightweight oEmbed for instant concurrent metadata (set False for full stats and stream URLs)",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON, description="'json' or 'markdown'")


@mcp.tool(
    name="tiktok_batch_videos",
    annotations=ToolAnnotations(
        title="Batch fetch TikTok video metadata",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
)
@guard
async def tiktok_batch_videos(params: TikTokBatchInput) -> str:
    """Fetch metadata for up to 20 TikTok video URLs concurrently."""
    items = await batch_tiktok_videos(params.urls, fast=params.fast)
    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"count": len(items), "items": items}, ensure_ascii=False, indent=2)

    lines = [
        f"### TikTok Batch Metadata ({len(items)} items)",
        "",
        "| # | Title | Creator | Views | Likes | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for idx, item in enumerate(items, start=1):
        if "error" in item:
            lines.append(f"| {idx} | Error: {item['error'][:50]} | - | - | - | error |")
            continue
        title = (item.get("title") or "Untitled").replace("|", " ")[:50]
        url = item.get("url") or ""
        creator = item.get("uploader") or item.get("uploader_id") or "-"
        views = f"{item['view_count']:,}" if item.get("view_count") is not None else "-"
        likes = f"{item['like_count']:,}" if item.get("like_count") is not None else "-"
        src = item.get("source") or "-"
        lines.append(f"| {idx} | [{title}]({url}) | {creator} | {views} | {likes} | `{src}` |")
    return "\n".join(lines)


class TikTokTranscriptInput(BaseModel):
    """Input for tiktok_get_transcript."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video: str = Field(..., description="TikTok video/photo URL or ID", min_length=1)
    language: Optional[str] = Field(
        default=None,
        description="Preferred language code (e.g. 'ru', 'en', 'eng-US', 'ru-RU'). If omitted, auto-detects.",
    )
    languages: Optional[List[str]] = Field(
        default=None,
        description="Optional list of 1-3 language codes (e.g. ['ru', 'en']) for multilingual Whisper + OCR.",
        max_length=3,
    )
    model_size: Optional[str] = Field(
        default=None,
        description="Optional Whisper model override (tiny, base, small, medium, large-v3, large-v3-turbo)",
    )
    include_ocr: bool = Field(
        default=True,
        description="Extract on-screen text from video keyframes and photo carousel slides via OCR (enabled by default)",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="tiktok_get_transcript",
    annotations=ToolAnnotations(
        title="Get TikTok video/slideshow transcript (speech + on-screen OCR)",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
)
@guard
async def tiktok_get_transcript(params: TikTokTranscriptInput) -> str:
    """Fetch closed captions, spoken audio transcript (Whisper), and on-screen text (OCR) from a TikTok video or photo slideshow.

    Returns timestamped speech segments, extracted screenshot/slide text, and continuous full text.
    """
    lang_arg = params.languages or params.language
    data = await get_tiktok_transcript(
        params.video,
        language=lang_arg,
        include_ocr=params.include_ocr,
        model_size=params.model_size,
    )
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(data, ensure_ascii=False, indent=2)

    if not data.get("transcript_available"):
        avail = ", ".join(data.get("available_languages", [])) or "none"
        return f"No transcript or on-screen text available for this TikTok post. Available caption tracks: {avail}."

    lines = [
        f"### Transcript: [{data.get('title')}]",
        f"- **Source:** `{data.get('source', 'unknown')}`",
        f"- **Language:** `{data.get('language')}`",
        "",
        "#### Timestamped Segments",
    ]
    for seg in data.get("segments", [])[:100]:
        lines.append(f"> `[{seg.get('start', '')} -> {seg.get('end', '')}]` {seg.get('text', '')}")

    ocr_segs = data.get("ocr_segments") or []
    if ocr_segs and data.get("audio_text"):
        lines.extend([
            "",
            "#### On-Screen Text (OCR / Screenshots)",
        ])
        for oseg in ocr_segs[:50]:
            lines.append(f"> `[{oseg.get('start', '')} -> {oseg.get('end', '')}]` {oseg.get('text', '')}")

    lines.extend([
        "",
        "#### Full Text",
        data.get("full_text", ""),
    ])
    return "\n".join(lines)


class TikTokUserInput(BaseModel):
    """Input for tiktok_get_user_videos."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    username: str = Field(
        ...,
        description="TikTok username (e.g. '@tiktok' or 'tiktok') or profile URL",
        min_length=1,
    )
    limit: int = Field(default=10, description="Number of recent videos to return (1-50)", ge=1, le=50)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="tiktok_get_user_videos",
    annotations=ToolAnnotations(
        title="Get TikTok creator's recent videos",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
)
@guard
async def tiktok_get_user_videos(params: TikTokUserInput) -> str:
    """Fetch recent videos published by a TikTok creator."""
    videos = await get_tiktok_user_videos(params.username, limit=params.limit)
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(videos, ensure_ascii=False, indent=2)

    clean_user = params.username.strip().lstrip("@")
    lines = [
        f"### Recent videos by @{clean_user} ({len(videos)})",
        "",
        "| # | Video | Views | Likes | Duration | ID |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for idx, v in enumerate(videos, start=1):
        views = f"{v.get('view_count', 0):,}" if v.get("view_count") is not None else "-"
        likes = f"{v.get('like_count', 0):,}" if v.get("like_count") is not None else "-"
        dur = f"{v.get('duration')}s" if v.get("duration") is not None else "-"
        title = (v.get("title") or "Untitled").replace("|", " ")[:50]
        url = v.get("url") or f"https://www.tiktok.com/@{clean_user}/video/{v.get('id')}"
        lines.append(f"| {idx} | [{title}]({url}) | {views} | {likes} | {dur} | `{v.get('id')}` |")

    return "\n".join(lines)


class TikTokPublishInput(BaseModel):
    """Input for tiktok_publish_video."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    video: str = Field(..., description="TikTok video URL", min_length=1)
    slug: Optional[str] = Field(default=None, description="Output filename without extension (defaults to sanitized title)")
    include_transcript: bool = Field(default=True, description="Transcribe speech and on-screen OCR text and attach to metadata")
    languages: Optional[List[str]] = Field(default=None, description="1-3 language codes for transcription, e.g. ['ru', 'en']", max_length=3)
    max_height: int = Field(default=1080, description="Maximum video resolution height (e.g. 720, 1080)", ge=240, le=2160)


@mcp.tool(
    name="tiktok_publish_video",
    annotations=ToolAnnotations(
        title="Publish TikTok video and cover to custom storage",
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=True,
    ),
)
@guard
async def tiktok_publish_video(params: TikTokPublishInput) -> str:
    """Download TikTok video and cover thumbnail, optionally transcribe speech + OCR, and upload to custom domain storage."""
    res = await publish_tiktok_video(
        params.video,
        slug=params.slug,
        include_transcript=params.include_transcript,
        languages=params.languages,
        max_height=params.max_height,
    )
    return json.dumps(res, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Instagram Tools
# --------------------------------------------------------------------------- #


class InstagramMediaInput(BaseModel):
    """Input for instagram_get_media."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    media: str = Field(
        ...,
        description="Instagram Reel URL, Post URL, or shortcode (e.g. https://www.instagram.com/reel/ABC123xyz/ or 'ABC123xyz')",
        min_length=1,
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="instagram_get_media",
    annotations=ToolAnnotations(
        title="Get Instagram Reel or post info",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
)
@guard
async def instagram_get_media(params: InstagramMediaInput) -> str:
    """Fetch complete metadata and media stream links for an Instagram Reel or post."""
    data = await get_instagram_media(params.media)
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(data, ensure_ascii=False, indent=2)

    likes = f"{data.get('like_count', 0):,}" if data.get("like_count") is not None else "0"
    comments = f"{data.get('comment_count', 0):,}" if data.get("comment_count") is not None else "0"
    views = f"{data.get('view_count', 0):,}" if data.get("view_count") is not None else "0"

    creator_link = (
        f"[@{data.get('channel')}]({data.get('channel_url')})"
        if data.get("channel_url")
        else (data.get("channel") or data.get("uploader") or "Unknown")
    )
    uploader_name = data.get("uploader")
    creator_display = (
        f"{creator_link} ({uploader_name})"
        if uploader_name and uploader_name != data.get("channel")
        else creator_link
    )

    duration_val = data.get("duration")
    duration_str = f"{duration_val}s" if duration_val else "N/A"

    title_clean = data.get("title") or "Instagram Post"
    media_type = data.get("media_type", "media").upper()

    lines = [
        f"### Instagram [{media_type}]: [{title_clean}]({data.get('url')})",
        "",
        f"- **Creator:** {creator_display}",
        f"- **Stats:** {likes} likes · {comments} comments · {views} views",
        f"- **Duration:** {duration_str}",
        f"- **Upload Date:** {data.get('upload_date') or 'N/A'}",
    ]

    if data.get("description"):
        caption_preview = data["description"].strip()
        lines.extend([
            "",
            "#### Caption",
            f"> {caption_preview}",
        ])

    if data.get("direct_video_url"):
        lines.append(f"\n- **Direct Video Stream:** [Open Stream]({data.get('direct_video_url')})")
    if data.get("direct_audio_url"):
        lines.append(f"- **Direct Audio Stream:** [Open Audio]({data.get('direct_audio_url')})")
    if data.get("thumbnail"):
        lines.append(f"- **Cover Image:** [View Image]({data.get('thumbnail')})")

    if data.get("is_carousel") and data.get("carousel_items"):
        items = data["carousel_items"]
        lines.extend([
            "",
            f"#### Carousel Slides ({len(items)} items)",
        ])
        for item in items:
            item_type = item.get("type", "item").capitalize()
            dur_info = f" ({item.get('duration')}s)" if item.get("duration") else ""
            link_str = f"[Open {item_type}]({item.get('url')})" if item.get("url") else "No direct link"
            lines.append(f"- Slide {item.get('index')}: **{item_type}**{dur_info}: {link_str}")

    return "\n".join(lines)


class InstagramUserInput(BaseModel):
    """Input for instagram_get_user_posts."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    username: str = Field(
        ...,
        description="Instagram username (e.g. '@instagram' or 'instagram') or profile URL",
        min_length=1,
    )
    limit: int = Field(default=10, description="Number of recent posts to return (1-50)", ge=1, le=50)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="instagram_get_user_posts",
    annotations=ToolAnnotations(
        title="Get Instagram creator's recent posts",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
)
@guard
async def instagram_get_user_posts(params: InstagramUserInput) -> str:
    """Fetch recent posts published by an Instagram creator."""
    posts = await get_instagram_user_posts(params.username, limit=params.limit)
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(posts, ensure_ascii=False, indent=2)

    clean_user = params.username.strip().lstrip("@")
    lines = [
        f"### Recent posts by @{clean_user} ({len(posts)})",
        "",
        "| # | Post | Likes | Comments | Duration | Shortcode |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for idx, p in enumerate(posts, start=1):
        likes = f"{p.get('like_count', 0):,}" if p.get("like_count") is not None else "-"
        comments = f"{p.get('comment_count', 0):,}" if p.get("comment_count") is not None else "-"
        dur = f"{p.get('duration')}s" if p.get("duration") is not None else "-"
        title = (p.get("title") or "Post").replace("|", " ")[:50]
        url = p.get("url") or f"https://www.instagram.com/p/{p.get('id')}/"
        lines.append(f"| {idx} | [{title}]({url}) | {likes} | {comments} | {dur} | `{p.get('id')}` |")

    return "\n".join(lines)


class InstagramTranscriptInput(BaseModel):
    """Input for instagram_get_transcript."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    media: str = Field(
        ...,
        description="Instagram Reel URL, Post URL, or shortcode (e.g. https://www.instagram.com/reel/ABC123xyz/ or 'ABC123xyz')",
        min_length=1,
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language code (e.g. 'ru', 'en') to force Whisper/OCR language. If omitted, auto-detects.",
    )
    include_ocr: bool = Field(
        default=True,
        description="Extract on-screen text from video keyframes and carousel/post images via OCR (enabled by default)",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="instagram_get_transcript",
    annotations=ToolAnnotations(
        title="Get Instagram Reel / post transcript (Whisper + OCR)",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
)
@guard
async def instagram_get_transcript(params: InstagramTranscriptInput) -> str:
    """Transcribe spoken audio (Whisper AI) and on-screen screenshot/slide text (OCR) from an Instagram Reel, video, or carousel post."""
    data = await get_instagram_transcript(
        params.media,
        language=params.language,
        include_ocr=params.include_ocr,
    )
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(data, ensure_ascii=False, indent=2)

    segments = data.get("segments") or []
    ocr_segments = data.get("ocr_segments") or []
    lines = [
        f"### Instagram Transcript: [{data.get('title') or 'Reel'}]({data.get('url')})",
        f"- **Creator:** {data.get('uploader') or data.get('channel') or '-'}",
        f"- **Source:** `{data.get('source', 'whisper_ai')}`",
        f"- **Language:** `{data.get('language')}` (probability: {data.get('language_probability')})",
        f"- **Audio Duration:** {data.get('duration')}s",
        "",
        "#### Timestamped Segments",
    ]

    if not segments and not data.get("full_text"):
        lines.append("> *No spoken words or on-screen text detected.*")
    else:
        for seg in segments[:100]:
            lines.append(f"> `[{seg.get('start', '')} -> {seg.get('end', '')}]` {seg.get('text', '')}")

    if ocr_segments and data.get("audio_text"):
        lines.extend([
            "",
            "#### On-Screen Text (OCR / Screenshots)",
        ])
        for oseg in ocr_segments[:50]:
            lines.append(f"> `[{oseg.get('start', '')} -> {oseg.get('end', '')}]` {oseg.get('text', '')}")

    if data.get("full_text"):
        lines.extend([
            "",
            "#### Full Text",
            data["full_text"],
        ])

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Threads Tools
# --------------------------------------------------------------------------- #


class ThreadsPostInput(BaseModel):
    """Input for threads_get_post."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    post: str = Field(
        ...,
        description="Threads post URL (e.g. https://www.threads.net/@zuck/post/DdZ7sQvkTFn or https://www.threads.net/t/DdZ7sQvkTFn) or post code",
        min_length=1,
    )
    include_comments: bool = Field(default=False, description="Whether to extract top comments and replies to the post")
    comments_limit: int = Field(default=10, description="Max number of comments to return (1-50)", ge=1, le=50)
    include_ocr: bool = Field(default=False, description="Transcribe attached videos and extract text from attached screenshot images via Whisper + OCR")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="threads_get_post",
    annotations=ToolAnnotations(
        title="Get Threads post info",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
)
@guard
async def threads_get_post(params: ThreadsPostInput) -> str:
    """Fetch complete content, media, and comments for a Threads post without login."""
    data = await get_threads_post(
        params.post,
        include_comments=params.include_comments,
        comments_limit=params.comments_limit,
        include_ocr=params.include_ocr,
    )
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(data, ensure_ascii=False, indent=2)

    likes = f"{data.get('like_count', 0):,}" if data.get("like_count") is not None else "0"
    replies = f"{data.get('reply_count', 0):,}" if data.get("reply_count") is not None else "0"
    author = data.get("author") or "user"
    author_name = data.get("author_name") or author

    lines = [
        f"### Threads: [@{author}]({data.get('url')}) ({author_name})",
        "",
        f"- **Stats:** {likes} likes · {replies} replies",
        f"- **Post Code:** `{data.get('code')}`",
    ]

    if data.get("text"):
        lines.extend([
            "",
            "#### Text Content",
            f"> {data['text']}",
        ])

    if data.get("media_transcript"):
        lines.extend([
            "",
            "#### Media Transcript / Screenshot OCR",
            data["media_transcript"],
        ])

    videos = data.get("direct_video_urls") or []
    images = data.get("direct_image_urls") or []

    if videos:
        lines.append("")
        for idx, v in enumerate(videos, start=1):
            lines.append(f"- **Video Stream {idx}:** [Open Video]({v})")

    if images:
        lines.append("")
        for idx, img in enumerate(images, start=1):
            lines.append(f"- **Image {idx}:** [View Image]({img})")

    comments = data.get("comments") or []
    if comments:
        lines.extend([
            "",
            f"#### Comments and Replies ({len(comments)} items)",
        ])
        for idx, c in enumerate(comments, start=1):
            c_author = f"[@{c.get('author')}]({c.get('url')})" if c.get("author") else "Anonymous"
            c_likes = f"{c.get('like_count', 0)} likes" if c.get("like_count") is not None else ""
            stats_str = f" ({c_likes})" if c_likes else ""
            c_text = c.get("text", "").replace("\n", " ").strip()
            preview = f'"{c_text[:120]}..."' if len(c_text) > 120 else f'"{c_text}"'
            lines.append(f"- {idx}. {c_author}{stats_str}: {preview}")

    return "\n".join(lines)


class ThreadsSearchInput(BaseModel):
    """Input for threads_search_posts."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description="Search keyword, topic, or hashtag (e.g. 'ai', 'python', '#opensource')",
        min_length=1,
    )
    limit: int = Field(default=10, description="Number of posts to return (1-50)", ge=1, le=50)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="threads_search_posts",
    annotations=ToolAnnotations(
        title="Search Threads posts by topic or tag",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
)
@guard
async def threads_search_posts(params: ThreadsSearchInput) -> str:
    """Search public Threads posts by topic, query, or hashtag without login."""
    posts = await search_threads_posts(params.query, limit=params.limit)
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(posts, ensure_ascii=False, indent=2)

    if not posts:
        return f"No Threads posts found matching query '{params.query}'."

    lines = [
        f"### Threads Search: '{params.query}' ({len(posts)} posts found)",
        "",
    ]

    for idx, p in enumerate(posts, start=1):
        author = p.get("author") or "user"
        author_name = p.get("author_name") or author
        likes = f"{p.get('like_count', 0):,}" if p.get("like_count") is not None else "0"
        replies = f"{p.get('reply_count', 0):,}" if p.get("reply_count") is not None else "0"
        post_link = f"[@{author}]({p.get('url')})"
        clean_text = p.get("text", "").replace("\n", " ").strip()
        text_preview = f"{clean_text[:140]}..." if len(clean_text) > 140 else (clean_text or "No text")

        lines.extend([
            f"#### {idx}. {post_link} ({author_name}) : `{p.get('code')}`",
            f"- **Stats:** {likes} likes · {replies} replies",
            f"> {text_preview}",
        ])

        v_count = len(p.get("direct_video_urls", []))
        img_count = len(p.get("direct_image_urls", []))
        media_badges = []
        if v_count:
            media_badges.append(f"{v_count} video(s)")
        if img_count:
            media_badges.append(f"{img_count} image(s)")
        if media_badges:
            lines.append(f"- **Media:** {', '.join(media_badges)}")
        lines.append("")

    return "\n".join(lines).strip()


class ThreadsUserInput(BaseModel):
    """Input for threads_get_user_posts."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    username: str = Field(
        ...,
        description="Threads username (e.g. '@zuck' or 'zuck') or profile URL",
        min_length=1,
    )
    limit: int = Field(default=10, description="Number of recent posts to return (1-50)", ge=1, le=50)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="threads_get_user_posts",
    annotations=ToolAnnotations(
        title="Get Threads creator's recent posts",
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
)
@guard
async def threads_get_user_posts(params: ThreadsUserInput) -> str:
    """Fetch recent posts and profile info for a Threads creator without login."""
    feed = await get_threads_user_posts(params.username, limit=params.limit)
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(feed, ensure_ascii=False, indent=2)

    user = feed.get("user") or {}
    posts = feed.get("posts") or []
    handle = user.get("username") or params.username.strip().lstrip("@")
    full_name = user.get("full_name") or handle
    verified = " (Verified)" if user.get("is_verified") else ""

    lines = [
        f"### Threads Feed: [{full_name}](https://www.threads.net/@{handle}){verified}",
        f"**Recent Posts ({len(posts)}):**",
        "",
        "| # | Post Code | Likes | Replies | Content Preview | Link |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for idx, p in enumerate(posts, start=1):
        likes = f"{p.get('like_count', 0):,}" if p.get("like_count") is not None else "-"
        replies = f"{p.get('reply_count', 0):,}" if p.get("reply_count") is not None else "-"
        txt = (p.get("text") or "No text").replace("|", " ").replace("\n", " ")[:60]
        code = p.get("code") or ""
        url = p.get("url") or f"https://www.threads.net/@{handle}/post/{code}"
        lines.append(f"| {idx} | `{code}` | {likes} | {replies} | {txt} | [Open]({url}) |")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #


def build_app():
    """Starlette app for streamable HTTP, optionally behind a static bearer token or OAuth 2.1."""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import JSONResponse

    mcp_path = os.getenv("SSMCP_PATH") or os.getenv("MCP_PATH") or "/mcp"
    mcp_host = os.getenv("SSMCP_HOST") or os.getenv("MCP_HOST") or "127.0.0.1"

    app = mcp.streamable_http_app(
        streamable_http_path=mcp_path,
        stateless_http=True,
        json_response=True,
        host=mcp_host,
    )
    from .oauth import (
        authorization_server_metadata,
        authorize,
        get_admin_passwords,
        get_issuer,
        google_callback,
        google_login,
        protected_resource_metadata,
        register_client,
        token_endpoint,
    )

    token = os.getenv("SSMCP_BEARER_TOKEN") or os.getenv("MCP_BEARER_TOKEN")

    PUBLIC_PREFIXES = (
        "/.well-known/",
        "/oauth/",
        "/healthz",
    )

    if token:

        class BearerAuth(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.method == "OPTIONS":
                    return await call_next(request)

                path = request.url.path
                if any(path.startswith(prefix) for prefix in PUBLIC_PREFIXES) or path in (
                    "/.well-known/oauth-protected-resource",
                    "/.well-known/oauth-authorization-server",
                    "/.well-known/openid-configuration",
                    "/oauth/register",
                    "/oauth/authorize",
                    "/oauth/google/login",
                    "/oauth/google/callback",
                    "/oauth/token",
                    "/healthz",
                ):
                    return await call_next(request)

                auth_header = request.headers.get("authorization", "")
                api_token = request.headers.get("api-token", "")
                x_api_key = request.headers.get("x-api-key", "")
                query_token = request.query_params.get("token") or request.query_params.get("key")

                req_token = (
                    auth_header.removeprefix("Bearer ").strip()
                    if auth_header.startswith("Bearer ")
                    else auth_header
                )
                valid_tokens = set(get_admin_passwords())

                is_valid = (
                    req_token in valid_tokens
                    or api_token in valid_tokens
                    or x_api_key in valid_tokens
                    or query_token in valid_tokens
                )
                if not is_valid:
                    issuer = get_issuer(request)
                    headers = {
                        "WWW-Authenticate": f'Bearer resource_metadata="{issuer}/.well-known/oauth-protected-resource"'
                    }
                    return JSONResponse({"error": "unauthorized"}, status_code=401, headers=headers)
                return await call_next(request)

        app.add_middleware(BearerAuth)

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r".*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["WWW-Authenticate", "Content-Type", "mcp-session-id"],
        max_age=86400,
    )

    async def healthz(request):
        return JSONResponse({"status": "ok", "service": "sSmcp", "version": __version__})

    app.add_route("/healthz", healthz, methods=["GET"])
    app.add_route("/.well-known/oauth-protected-resource", protected_resource_metadata, methods=["GET"])
    app.add_route("/.well-known/oauth-protected-resource/{path:path}", protected_resource_metadata, methods=["GET"])
    app.add_route("/mcp/.well-known/oauth-protected-resource", protected_resource_metadata, methods=["GET"])
    app.add_route("/.well-known/oauth-authorization-server", authorization_server_metadata, methods=["GET"])
    app.add_route("/.well-known/oauth-authorization-server/{path:path}", authorization_server_metadata, methods=["GET"])
    app.add_route("/mcp/.well-known/oauth-authorization-server", authorization_server_metadata, methods=["GET"])
    app.add_route("/.well-known/openid-configuration", authorization_server_metadata, methods=["GET"])
    app.add_route("/oauth/register", register_client, methods=["POST"])
    app.add_route("/oauth/authorize", authorize, methods=["GET", "POST"])
    app.add_route("/oauth/google/login", google_login, methods=["GET"])
    app.add_route("/oauth/google/callback", google_callback, methods=["GET"])
    app.add_route("/oauth/token", token_endpoint, methods=["POST"])

    return app


def main() -> None:
    import argparse

    host_env = os.getenv("SSMCP_HOST") or os.getenv("MCP_HOST")
    default_transport = (
        os.getenv("SSMCP_TRANSPORT")
        or os.getenv("MCP_TRANSPORT")
        or ("http" if host_env else "stdio")
    )

    parser = argparse.ArgumentParser(description="sSmcp — socialSimple MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "streamable-http", "sse"],
        default=default_transport,
        help="Transport protocol (stdio, http, streamable-http, sse; default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default=host_env or "127.0.0.1",
        help="Host for HTTP/SSE server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("SSMCP_PORT") or os.getenv("MCP_PORT") or "8080"),
        help="Port for HTTP/SSE server",
    )
    args, _ = parser.parse_known_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport in ("http", "streamable-http"):
        import uvicorn

        uvicorn.run(
            build_app(),
            host=args.host,
            port=args.port,
            log_level=os.getenv("LOG_LEVEL", "info"),
        )
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
