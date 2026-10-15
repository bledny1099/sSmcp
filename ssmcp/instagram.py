"""Instagram data extractor: metadata, streams, carousels, and Whisper + OCR transcripts."""

from __future__ import annotations

import asyncio
import copy
import html
import json
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional

from curl_cffi import requests
import yt_dlp

from .ocr import extract_images_ocr
from .whisper_transcriber import (
    download_and_transcribe_stream,
    merge_audio_and_ocr,
    transcribe_media_file,
)

logger = logging.getLogger("ssmcp.instagram")

_MEDIA_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL = 600.0
_MAX_CACHE_SIZE = 128


class InstagramError(RuntimeError):
    """Instagram extraction error phrased for the agent."""


def _get_proxy() -> Optional[str]:
    return (
        (os.getenv("SSMCP_PROXY") or os.getenv("SOCIAL_MCP_PROXY"))
        or os.getenv("INSTAGRAM_PROXY")
        or os.getenv("YT_TRANSCRIPT_PROXY")
        or os.getenv("HTTPS_PROXY")
        or None
    )


def normalize_instagram_url(url_or_id: str) -> str:
    """Normalize any Instagram URL, shortcode, or path into a canonical URL."""
    val = url_or_id.strip()
    if not val:
        raise InstagramError("Instagram URL or shortcode cannot be empty.")

    if val.startswith("http://") or val.startswith("https://"):
        cleaned = re.sub(r"\?[^#]*", "", val)
        return cleaned.rstrip("/") + "/"

    clean_path = val.strip("/")
    if "/" in clean_path:
        return f"https://www.instagram.com/{clean_path}/"

    return f"https://www.instagram.com/p/{clean_path}/"


def extract_shortcode(url_or_id: str) -> str:
    """Extract bare alphanumeric shortcode from URL or string."""
    m = re.search(r"/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)", url_or_id)
    if m:
        return m.group(1)
    val = url_or_id.strip().strip("/")
    if "/" in val:
        val = val.split("/")[-1]
    return re.sub(r"\?[^#]*", "", val)


def _clean_media_url(u: Optional[str]) -> Optional[str]:
    """Clean JSON-escaped CDN URLs into valid HTTP links."""
    if not u:
        return None
    cleaned = u.replace("\\/", "/")
    if "\\u" in cleaned:
        try:
            cleaned = cleaned.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass
    return cleaned


def _get_ydl_opts() -> Dict[str, Any]:
    """Build yt-dlp options, incorporating cookies and proxy if configured."""
    opts: Dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extract_flat": False,
        "writesubtitles": True,
        "allsubtitles": True,
    }

    proxy = _get_proxy()
    if proxy:
        opts["proxy"] = proxy

    cookies_path = os.getenv("INSTAGRAM_COOKIES_PATH")
    if cookies_path and os.path.exists(cookies_path):
        opts["cookiefile"] = cookies_path

    session_id = os.getenv("INSTAGRAM_SESSION_ID")
    if session_id:
        opts["http_headers"] = {
            "Cookie": f"sessionid={session_id}",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        }
    return opts


def _extract_embed_fallback(target_url: str) -> Dict[str, Any]:
    """Bypass login wall by extracting data from Instagram's public embed endpoint, including carousels."""
    shortcode = extract_shortcode(target_url)
    embed_url = f"https://www.instagram.com/p/{shortcode}/embed/captioned/"
    proxy = _get_proxy()

    r = None
    for p in ([None, proxy] if proxy else [None]):
        try:
            s = requests.Session(impersonate="chrome")
            if p:
                s.proxies = {"http": p, "https": p}
            resp = s.get(embed_url, timeout=12)
            if resp.status_code == 200 and "EmbedBrokenMedia" not in (resp.text or ""):
                r = resp
                break
        except Exception:
            continue

    if not r:
        raise InstagramError(f"Post '{shortcode}' not found or unavailable.")

    text = r.text
    if r.status_code != 200 or "EmbedBrokenMedia" in text:
        raise InstagramError(f"Post '{shortcode}' not found or unavailable.")

    caption_m = re.findall(r'<div class="Caption"[^>]*>(.*?)</div>', text, re.DOTALL)
    user_m = re.findall(r'class="UsernameText"[^>]*>(.*?)</span>', text, re.DOTALL) or re.findall(
        r'class="UsernameText"[^>]*>(.*?)</div>', text, re.DOTALL
    )

    username = html.unescape(re.sub(r"<[^>]+>", "", user_m[0])).strip() if user_m else ""
    caption = html.unescape(re.sub(r"<[^>]+>", "", caption_m[0])).strip() if caption_m else ""

    if caption.startswith(username):
        caption = caption[len(username) :].strip()

    img_m = re.findall(r'<img[^>]+class="EmbeddedMediaImage"[^>]+src="([^"]+)"', text)
    img_url = _clean_media_url(html.unescape(img_m[0])) if img_m else None

    direct_video_url = None
    direct_audio_url = None
    for sc in re.findall(r"<script[^>]*>(.*?)</script>", text, re.DOTALL):
        if "video_url" in sc:
            vm = re.search(r'"video_url":\s*"([^"]+)"', sc)
            if vm:
                direct_video_url = _clean_media_url(vm.group(1))
        if "audio_src" in sc:
            am = re.search(r'"audio_src":\s*"([^"]+)"', sc)
            if am:
                direct_audio_url = _clean_media_url(am.group(1))

    carousel_items = []
    idx_sidecar = text.find("edge_sidecar_to_children")
    if idx_sidecar != -1:
        sub = text[idx_sidecar:]
        unescaped = sub.replace('\\"', '"').replace("\\/", "/")
        m = re.search(r'edge_sidecar_to_children":\{"edges":(\[.*?\])\}', unescaped)
        if m:
            try:
                edges = json.loads(m.group(1))
                for idx, edge in enumerate(edges, start=1):
                    node = edge.get("node", {})
                    is_vid = bool(node.get("is_video"))
                    sc = node.get("shortcode") or f"{shortcode}_{idx}"
                    d_url = _clean_media_url(node.get("display_url"))
                    v_url = _clean_media_url(node.get("video_url")) if is_vid else None
                    carousel_items.append(
                        {
                            "index": idx,
                            "id": node.get("id"),
                            "shortcode": sc,
                            "type": "video" if is_vid else "image",
                            "url": v_url or d_url,
                            "thumbnail": d_url,
                            "direct_video_url": v_url,
                            "duration": node.get("video_duration"),
                            "view_count": node.get("video_view_count"),
                        }
                    )
            except Exception:
                pass

    if not carousel_items and len(img_m) > 1:
        for idx, src in enumerate(img_m, start=1):
            clean_src = _clean_media_url(html.unescape(src))
            carousel_items.append(
                {
                    "index": idx,
                    "id": f"{shortcode}_{idx}",
                    "type": "image",
                    "url": clean_src,
                    "thumbnail": clean_src,
                }
            )

    is_carousel = bool(carousel_items and len(carousel_items) > 1)
    if is_carousel:
        media_type = "carousel"
        if not img_url and carousel_items:
            img_url = carousel_items[0].get("thumbnail")
        if not direct_video_url and carousel_items:
            for item in carousel_items:
                if item.get("direct_video_url"):
                    direct_video_url = item.get("direct_video_url")
                    break
    elif direct_video_url or "GraphVideo" in text or "video" in text.lower():
        media_type = "video"
    else:
        media_type = "image"

    canonical = f"https://www.instagram.com/p/{shortcode}/"

    return {
        "id": shortcode,
        "url": canonical,
        "media_type": media_type,
        "title": (
            f"{'Carousel' if is_carousel else ('Video' if media_type == 'video' else 'Post')} by {username}"
            if username
            else f"Instagram {media_type.capitalize()}"
        ),
        "description": caption,
        "uploader": username,
        "channel": username,
        "channel_url": f"https://www.instagram.com/{username}/" if username else None,
        "uploader_id": username,
        "duration": None,
        "upload_date": None,
        "timestamp": None,
        "like_count": None,
        "comment_count": None,
        "view_count": None,
        "thumbnail": img_url,
        "direct_video_url": direct_video_url,
        "direct_audio_url": direct_audio_url,
        "subtitles_available": [],
        "is_carousel": is_carousel,
        "carousel_items": carousel_items,
        "source": "embed_fallback",
    }


def _store_cache(key: str, val: Dict[str, Any]) -> None:
    if len(_MEDIA_CACHE) >= _MAX_CACHE_SIZE:
        _MEDIA_CACHE.pop(next(iter(_MEDIA_CACHE)))
    _MEDIA_CACHE[key] = (time.time(), val)


def _extract_media_sync(target_url: str) -> Dict[str, Any]:
    now = time.time()
    cached = _MEDIA_CACHE.get(target_url)
    if cached and now - cached[0] < _CACHE_TTL:
        return copy.deepcopy(cached[1])

    opts = _get_ydl_opts()
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(target_url, download=False)
    except Exception:
        res = _extract_embed_fallback(target_url)
        _store_cache(target_url, res)
        return copy.deepcopy(res)

    if not info:
        res = _extract_embed_fallback(target_url)
        _store_cache(target_url, res)
        return copy.deepcopy(res)

    entries = info.get("entries")
    is_carousel = bool(entries and len(entries) > 0)

    carousel_items = []
    if is_carousel:
        for idx, entry in enumerate(entries or [], start=1):
            c_ext = entry.get("ext") or ("mp4" if entry.get("vcodec") != "none" else "jpg")
            c_type = "video" if c_ext == "mp4" or entry.get("vcodec") != "none" else "image"
            carousel_items.append(
                {
                    "index": idx,
                    "id": entry.get("id"),
                    "type": c_type,
                    "url": entry.get("url"),
                    "thumbnail": entry.get("thumbnail"),
                    "duration": entry.get("duration"),
                    "view_count": entry.get("view_count"),
                }
            )

    formats = info.get("formats") or []
    direct_video_url = info.get("url")
    direct_audio_url = None

    # Prefer combined progressive video+audio MP4 first
    for f in reversed(formats):
        if (
            not direct_video_url
            and f.get("vcodec") not in (None, "none")
            and f.get("acodec") not in (None, "none")
            and f.get("url")
        ):
            direct_video_url = f.get("url")
            break

    for f in reversed(formats):
        if not direct_video_url and f.get("vcodec") not in (None, "none") and f.get("url"):
            direct_video_url = f.get("url")
        if (
            not direct_audio_url
            and f.get("acodec") not in (None, "none")
            and f.get("vcodec") in (None, "none")
            and f.get("url")
        ):
            direct_audio_url = f.get("url")

    if is_carousel:
        media_type = "carousel"
    elif info.get("ext") == "mp4" or info.get("vcodec") != "none" or direct_video_url:
        media_type = "video"
    else:
        media_type = "image"

    channel = info.get("channel") or info.get("uploader_id") or ""
    channel_url = f"https://www.instagram.com/{channel}/" if channel else None

    subtitles_dict = info.get("subtitles") or info.get("automatic_captions") or {}
    available_subs = list(subtitles_dict.keys())

    result = {
        "id": info.get("id"),
        "url": info.get("webpage_url") or target_url,
        "media_type": media_type,
        "title": info.get("title") or "",
        "description": info.get("description") or "",
        "uploader": info.get("uploader") or channel or "",
        "channel": channel,
        "channel_url": channel_url,
        "uploader_id": info.get("uploader_id"),
        "duration": info.get("duration"),
        "upload_date": info.get("upload_date"),
        "timestamp": info.get("timestamp"),
        "like_count": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "view_count": info.get("view_count"),
        "thumbnail": info.get("thumbnail"),
        "direct_video_url": direct_video_url,
        "direct_audio_url": direct_audio_url,
        "subtitles_available": available_subs,
        "is_carousel": is_carousel,
        "carousel_items": carousel_items,
        "source": "native",
    }
    _store_cache(target_url, result)
    return copy.deepcopy(result)


async def get_instagram_media(url_or_id: str) -> Dict[str, Any]:
    """Fetch complete metadata and stream links for an Instagram Reel or post asynchronously."""
    target_url = normalize_instagram_url(url_or_id)
    return await asyncio.to_thread(_extract_media_sync, target_url)


def _extract_user_sync(username_or_url: str, limit: int = 10) -> List[Dict[str, Any]]:
    opts = _get_ydl_opts()
    opts["extract_flat"] = True
    opts["playlist_items"] = f"1:{max(1, min(limit, 50))}"

    val = username_or_url.strip()
    if val.startswith("http://") or val.startswith("https://"):
        target_url = val
    else:
        target_url = f"https://www.instagram.com/{val.lstrip('@')}/"

    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(target_url, download=False)
        except Exception as e:
            err_msg = str(e)
            if "login" in err_msg.lower() or "Unable to extract" in err_msg:
                raise InstagramError(
                    "Instagram profile feed is protected by login wall. "
                    "Use instagram_get_media with direct post/reel URLs (which bypass login), "
                    "or check the creator's Threads feed with threads_get_user_posts."
                ) from e
            raise InstagramError(f"Failed to fetch Instagram user posts: {err_msg}") from e

    entries = info.get("entries") or []
    results = []
    for entry in entries:
        results.append(
            {
                "id": entry.get("id"),
                "url": entry.get("url") or f"https://www.instagram.com/p/{entry.get('id')}/",
                "title": entry.get("title") or entry.get("description") or "",
                "uploader": entry.get("uploader") or info.get("uploader") or "",
                "like_count": entry.get("like_count"),
                "comment_count": entry.get("comment_count"),
                "duration": entry.get("duration"),
            }
        )
    return results


async def get_instagram_user_posts(username_or_url: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent posts published by an Instagram creator."""
    return await asyncio.to_thread(_extract_user_sync, username_or_url, limit)


async def get_instagram_transcript(
    url_or_id: str,
    language: Optional[str] = None,
    include_ocr: bool = True,
) -> Dict[str, Any]:
    """Fetch Reel/Post and transcribe both spoken audio (Whisper) and on-screen text/slides (OCR)."""
    data = await get_instagram_media(url_or_id)
    target_url = data.get("url") or normalize_instagram_url(url_or_id)
    proxy = _get_proxy()

    transcription: Optional[Dict[str, Any]] = None
    direct_vid = data.get("direct_video_url")
    direct_aud = data.get("direct_audio_url")

    # 1. Fast path: use direct CDN URLs already extracted by get_instagram_media
    if direct_vid or direct_aud:
        try:
            transcription = await download_and_transcribe_stream(
                stream_url=direct_aud or direct_vid,
                video_stream_url=direct_vid,
                language=language,
                include_ocr=include_ocr,
                proxy=proxy,
            )
        except Exception as e:
            logger.debug(f"Direct stream transcription failed for {target_url}: {e}")

    # 2. Fallback: download via yt-dlp if direct stream wasn't available or failed
    if not transcription and data.get("media_type") == "video":
        with tempfile.TemporaryDirectory() as tmpdir:
            outtmpl = os.path.join(tmpdir, "ig_media.%(ext)s")
            ydl_opts = _get_ydl_opts()
            ydl_opts.update(
                {
                    "format": "best[height<=720][ext=mp4]/best",
                    "outtmpl": outtmpl,
                }
            )
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = await asyncio.to_thread(ydl.extract_info, target_url, True)
                    downloaded_file = ydl.prepare_filename(info)
                    if os.path.exists(downloaded_file):
                        transcription = await transcribe_media_file(
                            downloaded_file,
                            language=language,
                            include_ocr=include_ocr,
                        )
            except Exception as e:
                logger.debug(f"yt-dlp fallback failed for {target_url}: {e}")

    # 3. Carousel / Image post OCR (or supplement carousel video with image slide OCR)
    slide_urls: List[str] = []
    if data.get("is_carousel") and data.get("carousel_items"):
        for item in data["carousel_items"]:
            if item.get("type") == "image":
                u = item.get("url") or item.get("thumbnail")
                if u:
                    slide_urls.append(u)
    elif data.get("media_type") == "image" and data.get("thumbnail"):
        slide_urls.append(data["thumbnail"])

    if slide_urls and include_ocr:
        slides_ocr = await extract_images_ocr(slide_urls, language=language, proxy=proxy)
        if slides_ocr.get("ocr_available"):
            if transcription:
                transcription = merge_audio_and_ocr(
                    transcription,
                    slides_ocr,
                    default_lang=language,
                )
            else:
                transcription = {
                    "language": language or "auto",
                    "language_probability": 1.0,
                    "duration": data.get("duration"),
                    "source": "ocr_slides",
                    "full_text": slides_ocr["full_text"],
                    "audio_text": "",
                    "ocr_text": slides_ocr["full_text"],
                    "segments": slides_ocr["segments"],
                    "ocr_segments": slides_ocr["segments"],
                }

    if not transcription:
        # Final fallback: try cover thumbnail OCR if present
        if include_ocr and data.get("thumbnail"):
            cover_ocr = await extract_images_ocr([data["thumbnail"]], language=language, proxy=proxy)
            if cover_ocr.get("ocr_available"):
                transcription = {
                    "language": language or "auto",
                    "language_probability": 1.0,
                    "duration": data.get("duration"),
                    "source": "ocr_cover",
                    "full_text": cover_ocr["full_text"],
                    "audio_text": "",
                    "ocr_text": cover_ocr["full_text"],
                    "segments": cover_ocr["segments"],
                    "ocr_segments": cover_ocr["segments"],
                }

    if not transcription:
        raise InstagramError(
            f"No spoken audio or readable on-screen text found in {target_url}."
        )

    return {
        "id": data.get("id"),
        "url": data.get("url"),
        "title": data.get("title"),
        "uploader": data.get("uploader"),
        "channel": data.get("channel"),
        "source": transcription.get("source", "whisper_ai"),
        "language": transcription.get("language"),
        "language_probability": transcription.get("language_probability"),
        "duration": transcription.get("duration"),
        "full_text": transcription.get("full_text"),
        "audio_text": transcription.get("audio_text", ""),
        "ocr_text": transcription.get("ocr_text", ""),
        "segments": transcription.get("segments", []),
        "ocr_segments": transcription.get("ocr_segments", []),
    }
