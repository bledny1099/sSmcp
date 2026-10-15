"""TikTok data extractor: metadata, batch info, oEmbed, transcripts (captions + Whisper + visual OCR), user videos, and custom domain publishing."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from curl_cffi import requests
import httpx
import yt_dlp

from .ocr import extract_images_ocr, extract_video_ocr
from .whisper_transcriber import (
    _download_stream_to_file,
    download_and_transcribe_stream,
    merge_audio_and_ocr,
    transcribe_media_file,
)

logger = logging.getLogger("ssmcp.tiktok")

_VIDEO_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL = 600.0
_MAX_CACHE_SIZE = 128
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class TikTokError(RuntimeError):
    """TikTok extraction error phrased for the agent."""


def _get_proxy() -> Optional[str]:
    return (
        os.getenv("SSMCP_PROXY")
        or (os.getenv("SSMCP_PROXY") or os.getenv("SOCIAL_MCP_PROXY"))
        or os.getenv("TIKTOK_MCP_PROXY")
        or os.getenv("TIKTOK_PROXY")
        or os.getenv("YT_TRANSCRIPT_PROXY")
        or os.getenv("HTTPS_PROXY")
        or None
    )


def _get_cookies_file() -> Optional[str]:
    for var in ("SSMCP_COOKIES_FILE", "TIKTOK_MCP_COOKIES_FILE", "TIKTOK_COOKIES_PATH", "YT_COOKIES_PATH"):
        val = os.getenv(var)
        if val and os.path.exists(val):
            return val
    return None


def _slugify(text: str, fallback: str) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")[:60]
    return slug or fallback


def extract_tiktok_id(url_or_id: str) -> str:
    """Extract numeric TikTok video or photo ID from URL or string."""
    m = re.search(r"/(?:video|photo)/(\d+)", url_or_id)
    if m:
        return m.group(1)
    digits = re.search(r"(\d{15,25})", url_or_id)
    if digits:
        return digits.group(1)
    val = url_or_id.strip().strip("/")
    if "/" in val:
        val = val.split("/")[-1]
    return re.sub(r"\?[^#]*", "", val)


def normalize_tiktok_url(url_or_id: str) -> str:
    """Normalize any TikTok URL or ID into a canonical URL."""
    val = url_or_id.strip()
    if val.startswith("http://") or val.startswith("https://"):
        return val
    if val.isdigit():
        return f"https://www.tiktok.com/@tiktok/video/{val}"
    if "/video/" in val or "/photo/" in val:
        return f"https://www.tiktok.com/{val.lstrip('/')}"
    return val


def _resolve_short_url(url: str, proxy: Optional[str] = None) -> str:
    """Resolve shortened vm.tiktok.com / vt.tiktok.com / /t/ redirect links to full URLs."""
    if re.search(r"\d{15,25}", url):
        return url
    for p in ([proxy, None] if proxy else [None]):
        try:
            s = requests.Session(impersonate="chrome")
            if p:
                s.proxies = {"http": p, "https": p}
            resp = s.get(url, timeout=10, allow_redirects=True)
            if resp.url and resp.url != url:
                return str(resp.url)
        except Exception:
            continue
    return url


def parse_vtt_subtitles(vtt_text: str) -> Dict[str, Any]:
    """Parse WebVTT subtitle content into clean text and timestamped segments."""
    segments = []
    blocks = re.split(r"\n\s*\n", vtt_text.strip())
    for b in blocks:
        lines = [line.strip() for line in b.splitlines() if line.strip()]
        if not lines or lines[0].startswith("WEBVTT") or lines[0].startswith("NOTE"):
            continue
        time_idx = -1
        for idx, line in enumerate(lines):
            if "-->" in line:
                time_idx = idx
                break
        if time_idx != -1:
            time_line = lines[time_idx]
            match = re.match(r"([\d:.]+)\s*-->\s*([\d:.]+)", time_line)
            start = match.group(1) if match else ""
            end = match.group(2) if match else ""
            text_lines = lines[time_idx + 1 :]
            cleaned = re.sub(r"<[^>]+>", "", " ".join(text_lines)).strip()
            if cleaned:
                segments.append({"start": start, "end": end, "text": cleaned})

    full_text = " ".join(s["text"] for s in segments)
    return {"full_text": full_text, "segments": segments}


async def _fetch_oembed(url: str) -> Dict[str, Any]:
    """Lightweight public oEmbed endpoint: title, author, and thumbnail without API keys."""
    proxy = _get_proxy()
    client_kwargs: Dict[str, Any] = {"timeout": 10.0, "follow_redirects": True}
    if proxy and (proxy.startswith("http") or proxy.startswith("socks")):
        client_kwargs["proxy"] = proxy
    async with httpx.AsyncClient(**client_kwargs) as client:
        r = await client.get("https://www.tiktok.com/oembed", params={"url": url})
        r.raise_for_status()
        data = r.json()
    author_url = data.get("author_url") or ""
    author_id = author_url.rstrip("/").split("@")[-1] if "@" in author_url else data.get("author_unique_id", "")
    return {
        "id": str(data.get("embed_product_id") or extract_tiktok_id(url)),
        "url": url,
        "media_type": "video",
        "title": data.get("title") or "",
        "description": data.get("title") or "",
        "uploader": data.get("author_name") or author_id,
        "uploader_id": author_id,
        "uploader_url": author_url or (f"https://www.tiktok.com/@{author_id}" if author_id else None),
        "duration": None,
        "duration_string": None,
        "upload_date": None,
        "timestamp": None,
        "view_count": None,
        "like_count": None,
        "comment_count": None,
        "repost_count": None,
        "save_count": None,
        "track": None,
        "artist": None,
        "thumbnail": data.get("thumbnail_url"),
        "direct_video_url": None,
        "direct_audio_url": None,
        "subtitles_available": [],
        "is_carousel": False,
        "carousel_items": [],
        "source": "oembed",
        "_raw_subtitles": {},
    }


def _extract_tikwm_fallback(target_url: str) -> Dict[str, Any]:
    """Extract metadata, direct video/audio streams, and Photo Mode slides via public tikwm API."""
    proxy = _get_proxy()
    client_kwargs: Dict[str, Any] = {"timeout": 20.0, "follow_redirects": True}
    if proxy and (proxy.startswith("http") or proxy.startswith("socks")):
        client_kwargs["proxy"] = proxy

    with httpx.Client(**client_kwargs) as client:
        resp = client.post(
            "https://www.tikwm.com/api/",
            data={"url": target_url, "count": 12, "cursor": 0, "web": 1, "hd": 1},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                )
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") != 0 or not payload.get("data"):
            raise TikTokError(f"tikwm fallback error: {payload.get('msg', 'unknown error')}")

        def _tikwm_url(u: Optional[str]) -> Optional[str]:
            if not u:
                return None
            if u.startswith("http://") or u.startswith("https://"):
                return u
            return "https://www.tikwm.com" + u

        d = payload["data"]
        raw_images = d.get("images") or []
        images = [u for u in (_tikwm_url(img) for img in raw_images if img) if u]
        is_carousel = bool(images)
        carousel_items = [
            {"index": idx, "type": "image", "url": img_url, "thumbnail": img_url}
            for idx, img_url in enumerate(images, start=1)
        ]

        video_url = _tikwm_url(d.get("play") or d.get("wmplay") or d.get("hdplay"))
        audio_url = _tikwm_url(d.get("music")) or video_url
        uploader_id = d.get("author", {}).get("unique_id") or ""
        uploader_name = d.get("author", {}).get("nickname") or uploader_id
        vid_id = str(d.get("id") or extract_tiktok_id(target_url))

        return {
            "id": vid_id,
            "url": f"https://www.tiktok.com/@{uploader_id or 'user'}/{'photo' if is_carousel else 'video'}/{vid_id}",
            "media_type": "carousel" if is_carousel else "video",
            "title": d.get("title") or "",
            "description": d.get("title") or "",
            "uploader": uploader_name,
            "uploader_id": uploader_id,
            "uploader_url": f"https://www.tiktok.com/@{uploader_id}" if uploader_id else None,
            "duration": d.get("duration"),
            "duration_string": f"{d.get('duration')}s" if d.get("duration") else None,
            "upload_date": None,
            "timestamp": d.get("create_time"),
            "view_count": d.get("play_count"),
            "like_count": d.get("digg_count"),
            "comment_count": d.get("comment_count"),
            "repost_count": d.get("share_count"),
            "save_count": d.get("collect_count"),
            "track": d.get("music_info", {}).get("title"),
            "artist": d.get("music_info", {}).get("author"),
            "thumbnail": _tikwm_url(d.get("cover")) or (images[0] if images else None),
            "direct_video_url": None if is_carousel else video_url,
            "direct_audio_url": audio_url,
            "subtitles_available": [],
            "is_carousel": is_carousel,
            "carousel_items": carousel_items,
            "source": "tikwm_fallback",
            "_raw_subtitles": {},
        }


def _extract_embed_fallback(target_url: str) -> Dict[str, Any]:
    """Extract metadata, media streams, and carousel slides from TikTok embed v2 endpoint."""
    proxy = _get_proxy()
    resolved_url = _resolve_short_url(target_url, proxy=proxy)
    video_id = extract_tiktok_id(resolved_url)
    embed_url = f"https://www.tiktok.com/embed/v2/{video_id}"

    r = None
    for p in ([None, proxy] if proxy else [None]):
        try:
            s = requests.Session(impersonate="chrome")
            if p:
                s.proxies = {"http": p, "https": p}
            resp = s.get(embed_url, timeout=12)
            if resp.status_code == 200 and "__FRONTITY_CONNECT_STATE__" in (resp.text or ""):
                r = resp
                break
        except Exception:
            continue

    if not r:
        raise TikTokError(f"Failed to fetch TikTok data for '{video_id}'.")

    p = r.text.find("__FRONTITY_CONNECT_STATE__")
    if p == -1:
        raise TikTokError(f"Could not locate media data in TikTok response for '{video_id}'.")

    tag_end = r.text.find(">", p)
    script_end = r.text.find("</script>", tag_end)
    try:
        data = json.loads(r.text[tag_end + 1 : script_end])
    except Exception as e:
        raise TikTokError(f"Failed to parse TikTok response: {e}") from e

    source = data.get("source", {})
    video_data = None
    for v in source.get("data", {}).values():
        if isinstance(v, dict) and "videoData" in v:
            video_data = v["videoData"]
            break

    if not video_data:
        raise TikTokError(f"No videoData found in TikTok response for '{video_id}'.")

    item = video_data.get("itemInfos", {})
    author = video_data.get("authorInfos", {})
    music = video_data.get("musicInfos", {})
    image_post_info = video_data.get("imagePostInfo") or item.get("imagePostInfo")

    is_carousel = bool(image_post_info and image_post_info.get("images"))
    carousel_items = []
    if is_carousel:
        for idx, img in enumerate(image_post_info.get("images", []), start=1):
            img_url = None
            if isinstance(img, dict):
                for subk in ("display_image", "imageURL", "image", "thumbnail", "imageThumbnail"):
                    sub = img.get(subk)
                    if isinstance(sub, dict):
                        urls = sub.get("urlList") or sub.get("url_list") or sub.get("urls")
                        if urls and isinstance(urls, list) and urls[0]:
                            img_url = urls[0]
                            break
                        if isinstance(sub.get("url"), str):
                            img_url = sub["url"]
                            break
                    elif isinstance(sub, list) and sub:
                        img_url = sub[0]
                        break
                    elif isinstance(sub, str):
                        img_url = sub
                        break
                if not img_url:
                    urls = img.get("urlList") or img.get("url_list") or img.get("urls")
                    if urls and isinstance(urls, list) and urls[0]:
                        img_url = urls[0]
            elif isinstance(img, str):
                img_url = img

            carousel_items.append(
                {
                    "index": idx,
                    "type": "image",
                    "url": img_url,
                    "thumbnail": img_url,
                }
            )

    v_urls = item.get("video", {}).get("urls", [])
    m_urls = music.get("playUrl")
    if isinstance(m_urls, dict):
        m_urls = m_urls.get("urls", [])
    elif isinstance(m_urls, str):
        m_urls = [m_urls]
    elif not isinstance(m_urls, list):
        m_urls = []

    covers = item.get("covers", [])
    direct_video_url = v_urls[0] if v_urls else None
    direct_audio_url = m_urls[0] if m_urls else None
    thumbnail_url = (
        covers[0]
        if covers
        else (carousel_items[0].get("thumbnail") if carousel_items else None)
    )

    uploader_id = author.get("uniqueId") or ""
    uploader_name = author.get("nickName") or uploader_id

    return {
        "id": video_id,
        "url": f"https://www.tiktok.com/@{uploader_id or 'user'}/video/{video_id}",
        "media_type": "carousel" if is_carousel else ("video" if direct_video_url else "media"),
        "title": item.get("text", ""),
        "description": item.get("text", ""),
        "uploader": uploader_name,
        "uploader_id": uploader_id,
        "uploader_url": f"https://www.tiktok.com/@{uploader_id}" if uploader_id else None,
        "duration": item.get("video", {}).get("videoMeta", {}).get("duration"),
        "duration_string": None,
        "upload_date": None,
        "timestamp": item.get("createTime"),
        "view_count": item.get("playCount"),
        "like_count": item.get("diggCount"),
        "comment_count": item.get("commentCount"),
        "repost_count": item.get("shareCount"),
        "save_count": None,
        "track": music.get("musicName"),
        "artist": music.get("authorName"),
        "thumbnail": thumbnail_url,
        "direct_video_url": direct_video_url,
        "direct_audio_url": direct_audio_url,
        "subtitles_available": [],
        "is_carousel": is_carousel,
        "carousel_items": carousel_items,
        "source": "embed_fallback",
        "_raw_subtitles": {},
    }


def _extract_fallback_cascade(canonical_url: str) -> Dict[str, Any]:
    """Run tikwm fallback first, then embed/v2 fallback."""
    last_err: Optional[Exception] = None
    for fn in (_extract_tikwm_fallback, _extract_embed_fallback):
        try:
            return fn(canonical_url)
        except Exception as e:
            last_err = e
            continue
    raise TikTokError(f"Failed to extract TikTok post '{canonical_url}': {last_err}")


def _extract_video_sync(target_url: str) -> Dict[str, Any]:
    canonical_url = normalize_tiktok_url(target_url)
    now = time.time()
    cached = _VIDEO_CACHE.get(canonical_url)
    if cached and now - cached[0] < _CACHE_TTL:
        return copy.deepcopy(cached[1])

    # Photo Mode URLs (/photo/) are handled directly via tikwm / embed fallback
    if "/photo/" in canonical_url:
        res = _extract_fallback_cascade(canonical_url)
        _store_cache(canonical_url, res)
        return copy.deepcopy(res)

    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extract_flat": False,
        "writesubtitles": True,
        "allsubtitles": True,
        "socket_timeout": 10,
        "retries": 1,
        "extractor_retries": 1,
    }
    proxy = _get_proxy()
    if proxy:
        ydl_opts["proxy"] = proxy

    cookies_path = _get_cookies_file()
    if cookies_path:
        ydl_opts["cookiefile"] = cookies_path

    info = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(canonical_url, download=False)
    except Exception:
        res = _extract_fallback_cascade(canonical_url)
        _store_cache(canonical_url, res)
        return copy.deepcopy(res)

    if not info:
        res = _extract_fallback_cascade(canonical_url)
        _store_cache(canonical_url, res)
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

    if (not direct_video_url or not carousel_items) and info.get("duration") == 0:
        try:
            fb_data = _extract_fallback_cascade(canonical_url)
            if fb_data.get("is_carousel") or fb_data.get("direct_video_url"):
                _store_cache(canonical_url, fb_data)
                return copy.deepcopy(fb_data)
        except Exception:
            pass

    media_type = "carousel" if is_carousel else ("video" if direct_video_url else "media")
    subtitles_dict = info.get("subtitles") or info.get("automatic_captions") or {}
    available_subs = list(subtitles_dict.keys())

    result = {
        "id": info.get("id"),
        "url": info.get("webpage_url") or canonical_url,
        "media_type": media_type,
        "title": info.get("title") or info.get("description") or "",
        "description": info.get("description") or "",
        "uploader": info.get("uploader") or info.get("channel") or "",
        "uploader_id": info.get("uploader_id") or "",
        "uploader_url": info.get("uploader_url") or "",
        "duration": info.get("duration"),
        "duration_string": info.get("duration_string"),
        "upload_date": info.get("upload_date"),
        "timestamp": info.get("timestamp"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "repost_count": info.get("repost_count"),
        "save_count": info.get("save_count"),
        "track": info.get("track"),
        "artist": info.get("artist") or (info.get("artists")[0] if info.get("artists") else None),
        "thumbnail": info.get("thumbnail"),
        "direct_video_url": direct_video_url,
        "direct_audio_url": direct_audio_url,
        "subtitles_available": available_subs,
        "is_carousel": is_carousel,
        "carousel_items": carousel_items,
        "source": "native",
        "_raw_subtitles": subtitles_dict,
    }
    _store_cache(canonical_url, result)
    return copy.deepcopy(result)


def _store_cache(key: str, val: Dict[str, Any]) -> None:
    if len(_VIDEO_CACHE) >= _MAX_CACHE_SIZE:
        _VIDEO_CACHE.pop(next(iter(_VIDEO_CACHE)))
    _VIDEO_CACHE[key] = (time.time(), val)


async def get_tiktok_video(url_or_id: str, fast: bool = False) -> Dict[str, Any]:
    """Fetch complete TikTok video metadata asynchronously, with optional fast oEmbed mode."""
    target_url = normalize_tiktok_url(url_or_id)
    if fast:
        try:
            return await _fetch_oembed(target_url)
        except Exception:
            pass
    try:
        data = await asyncio.to_thread(_extract_video_sync, target_url)
        data.pop("_raw_subtitles", None)
        return data
    except Exception as exc:
        try:
            fallback = await _fetch_oembed(target_url)
            fallback["warning"] = f"Full extraction failed ({exc}); returning oEmbed fallback"
            return fallback
        except Exception:
            raise exc


async def batch_tiktok_videos(urls: List[str], fast: bool = True) -> List[Dict[str, Any]]:
    """Fetch metadata for up to 20 TikTok video URLs concurrently."""
    if len(urls) > 20:
        raise TikTokError("Maximum 20 URLs allowed per batch request.")
    results = await asyncio.gather(
        *(get_tiktok_video(u, fast=fast) for u in urls),
        return_exceptions=True,
    )
    return [
        {"url": u, "error": str(r)} if isinstance(r, Exception) else r
        for u, r in zip(urls, results)
    ]


async def get_tiktok_transcript(
    url_or_id: str,
    language: Optional[str | List[str]] = None,
    include_ocr: bool = True,
    model_size: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch transcript for a TikTok video or photo slideshow using captions, Whisper, and visual OCR."""
    target_url = normalize_tiktok_url(url_or_id)
    data = await asyncio.to_thread(_extract_video_sync, target_url)
    subtitles_dict = data.get("_raw_subtitles") or {}
    proxy = _get_proxy()

    lang_str = language[0] if isinstance(language, list) and language else (language if isinstance(language, str) else None)

    # 1. Carousel / Photo Mode post: extract text from all slide images via OCR
    if data.get("is_carousel") and data.get("carousel_items"):
        slide_urls = [
            item.get("url") or item.get("thumbnail")
            for item in data["carousel_items"]
            if item.get("url") or item.get("thumbnail")
        ]
        if slide_urls:
            ocr_slides = await extract_images_ocr(slide_urls, language=language, proxy=proxy)
            if ocr_slides.get("ocr_available"):
                return {
                    "video_id": data.get("id"),
                    "title": data.get("title"),
                    "uploader": data.get("uploader"),
                    "transcript_available": True,
                    "source": "ocr_slides",
                    "language": lang_str or "auto",
                    "available_languages": list(subtitles_dict.keys()),
                    "full_text": ocr_slides["full_text"],
                    "audio_text": "",
                    "ocr_text": ocr_slides["full_text"],
                    "segments": ocr_slides["segments"],
                    "ocr_segments": ocr_slides["segments"],
                }

    # 2. Check if VTT subtitles exist
    vtt_parsed: Optional[Dict[str, Any]] = None
    vtt_url = None
    chosen_lang = lang_str
    if subtitles_dict:
        if not chosen_lang or chosen_lang not in subtitles_dict:
            for pref in ("ru-RU", "ru", "eng-US", "en"):
                if pref in subtitles_dict:
                    chosen_lang = pref
                    break
            if not chosen_lang:
                chosen_lang = next(iter(subtitles_dict))

        sub_formats = subtitles_dict.get(chosen_lang, [])
        for sf in sub_formats:
            if sf.get("url"):
                vtt_url = sf["url"]
                break

    if vtt_url:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as http_client:
                resp = await http_client.get(vtt_url)
                if resp.status_code == 200:
                    parsed = parse_vtt_subtitles(resp.text)
                    if parsed.get("full_text"):
                        vtt_parsed = parsed
        except Exception as e:
            logger.debug(f"VTT subtitle download failed: {e}")

    if vtt_parsed:
        ocr_res: Dict[str, Any] = {"ocr_available": False, "full_text": "", "segments": []}
        direct_vid = data.get("direct_video_url")
        if include_ocr and direct_vid:
            try:
                tmp_vid = await _download_stream_to_file(direct_vid, suffix=".mp4", proxy=proxy)
                try:
                    ocr_res = await extract_video_ocr(tmp_vid, language=language)
                finally:
                    if os.path.exists(tmp_vid):
                        os.remove(tmp_vid)
            except Exception as e:
                logger.debug(f"Optional OCR pass alongside VTT failed: {e}")

        merged = merge_audio_and_ocr(
            {
                "language": chosen_lang,
                "full_text": vtt_parsed["full_text"],
                "segments": vtt_parsed["segments"],
            },
            ocr_res,
            default_lang=chosen_lang,
        )
        source_label = "subtitles+ocr" if ocr_res.get("ocr_available") else "subtitles"
        return {
            "video_id": data.get("id"),
            "title": data.get("title"),
            "uploader": data.get("uploader"),
            "transcript_available": True,
            "source": source_label,
            "language": chosen_lang,
            "available_languages": list(subtitles_dict.keys()),
            "full_text": merged["full_text"],
            "audio_text": merged["audio_text"],
            "ocr_text": merged["ocr_text"],
            "segments": merged["segments"],
            "ocr_segments": merged["ocr_segments"],
        }

    # 3. Direct video/audio stream -> Whisper + Visual Keyframe OCR in parallel
    direct_vid = data.get("direct_video_url")
    direct_aud = data.get("direct_audio_url") if not direct_vid else direct_vid
    if direct_vid or direct_aud:
        try:
            result = await download_and_transcribe_stream(
                stream_url=direct_aud or direct_vid,
                video_stream_url=direct_vid,
                language=language,
                include_ocr=include_ocr,
                proxy=proxy,
                model_size=model_size,
            )
            if result.get("full_text"):
                return {
                    "video_id": data.get("id"),
                    "title": data.get("title"),
                    "uploader": data.get("uploader"),
                    "transcript_available": True,
                    "source": result.get("source", "whisper_ai"),
                    "language": result.get("language", lang_str or "auto"),
                    "available_languages": list(subtitles_dict.keys()),
                    "full_text": result.get("full_text", ""),
                    "audio_text": result.get("audio_text", ""),
                    "ocr_text": result.get("ocr_text", ""),
                    "segments": result.get("segments", []),
                    "ocr_segments": result.get("ocr_segments", []),
                }
        except Exception as e:
            logger.debug(f"Direct stream transcription failed: {e}")

    # 4. Fallback: download via yt-dlp (video+audio MP4 so both Whisper and OCR can run)
    with tempfile.TemporaryDirectory() as tmpdir:
        outtmpl = os.path.join(tmpdir, "tiktok_media.%(ext)s")
        ydl_opts: Dict[str, Any] = {
            "quiet": True,
            "format": "best[height<=720][ext=mp4]/best",
            "outtmpl": outtmpl,
            "no_warnings": True,
        }
        if proxy:
            ydl_opts["proxy"] = proxy
        cookies_path = _get_cookies_file()
        if cookies_path:
            ydl_opts["cookiefile"] = cookies_path
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, target_url, True)
                dl_file = ydl.prepare_filename(info)
                if os.path.exists(dl_file):
                    result = await transcribe_media_file(
                        dl_file,
                        language=language,
                        include_ocr=include_ocr,
                        model_size=model_size,
                    )
                    if result.get("full_text"):
                        return {
                            "video_id": data.get("id"),
                            "title": data.get("title"),
                            "uploader": data.get("uploader"),
                            "transcript_available": True,
                            "source": result.get("source", "whisper_ai"),
                            "language": result.get("language", lang_str or "auto"),
                            "available_languages": list(subtitles_dict.keys()),
                            "full_text": result.get("full_text", ""),
                            "audio_text": result.get("audio_text", ""),
                            "ocr_text": result.get("ocr_text", ""),
                            "segments": result.get("segments", []),
                            "ocr_segments": result.get("ocr_segments", []),
                        }
        except Exception as e:
            logger.debug(f"TikTok yt-dlp fallback failed: {e}")

    # 5. Final fallback: cover image OCR if available
    if include_ocr and data.get("thumbnail"):
        ocr_cover = await extract_images_ocr([data["thumbnail"]], language=language, proxy=proxy)
        if ocr_cover.get("ocr_available"):
            return {
                "video_id": data.get("id"),
                "title": data.get("title"),
                "uploader": data.get("uploader"),
                "transcript_available": True,
                "source": "ocr_cover",
                "language": lang_str or "auto",
                "available_languages": list(subtitles_dict.keys()),
                "full_text": ocr_cover["full_text"],
                "audio_text": "",
                "ocr_text": ocr_cover["full_text"],
                "segments": ocr_cover["segments"],
                "ocr_segments": ocr_cover["segments"],
            }

    return {
        "video_id": data.get("id"),
        "title": data.get("title"),
        "uploader": data.get("uploader"),
        "transcript_available": False,
        "message": "No spoken speech or on-screen text found in this TikTok post.",
        "available_languages": list(subtitles_dict.keys()),
        "full_text": "",
        "segments": [],
    }


async def _upload_media_asset(path: Path, content_type: str, meta: Dict[str, Any]) -> str:
    """Upload a downloaded video or cover file to local storage or a remote HTTP endpoint."""
    upload_mode = (
        os.getenv("SSMCP_UPLOAD_MODE")
        or os.getenv("TIKTOK_MCP_UPLOAD_MODE")
        or "off"
    ).lower()
    local_dir = os.getenv("SSMCP_LOCAL_DIR") or os.getenv("TIKTOK_MCP_LOCAL_DIR")
    upload_url = os.getenv("SSMCP_UPLOAD_URL") or os.getenv("TIKTOK_MCP_UPLOAD_URL")
    upload_token = os.getenv("SSMCP_UPLOAD_TOKEN") or os.getenv("TIKTOK_MCP_UPLOAD_TOKEN")
    public_base_url = (
        os.getenv("SSMCP_PUBLIC_BASE_URL") or os.getenv("TIKTOK_MCP_PUBLIC_BASE_URL") or ""
    )

    if upload_mode == "off":
        raise TikTokError(
            "Publishing is disabled. Set SSMCP_UPLOAD_MODE=local|http and related environment variables."
        )
    if upload_mode == "local":
        if not local_dir:
            raise TikTokError("SSMCP_LOCAL_DIR is not configured.")
        dest_dir = Path(local_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest_dir / path.name)
        if not public_base_url:
            return str(dest_dir / path.name)
        return f"{public_base_url.rstrip('/')}/{path.name}"

    if not upload_url:
        raise TikTokError("SSMCP_UPLOAD_URL is not configured.")
    headers = {"Authorization": f"Bearer {upload_token}"} if upload_token else {}
    async with httpx.AsyncClient(timeout=600.0) as client:
        with path.open("rb") as fh:
            resp = await client.post(
                upload_url,
                headers=headers,
                files={"file": (path.name, fh, content_type)},
                data={"meta": json.dumps(meta, ensure_ascii=False)},
            )
    resp.raise_for_status()
    try:
        return resp.json().get("url") or f"{public_base_url.rstrip('/')}/{path.name}"
    except ValueError:
        return f"{public_base_url.rstrip('/')}/{path.name}"


async def publish_tiktok_video(
    url_or_id: str,
    slug: Optional[str] = None,
    include_transcript: bool = True,
    languages: Optional[List[str]] = None,
    max_height: int = 1080,
) -> Dict[str, Any]:
    """Download TikTok video + cover thumbnail, attach transcript, and upload to custom domain storage."""
    target_url = normalize_tiktok_url(url_or_id)
    meta = await get_tiktok_video(target_url, fast=False)
    proxy = _get_proxy()

    with tempfile.TemporaryDirectory(prefix="ssmcp_publish_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        name = _slugify(slug or meta.get("title") or "", fallback=str(meta.get("id") or "tiktok"))

        video_file: Optional[Path] = None
        if meta.get("direct_video_url"):
            dl_path = await _download_stream_to_file(
                meta["direct_video_url"], suffix=".mp4", proxy=proxy
            )
            target = tmp_dir / f"{name}.mp4"
            shutil.move(dl_path, target)
            video_file = target
        else:
            ydl_opts: Dict[str, Any] = {
                "quiet": True,
                "no_warnings": True,
                "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
                "merge_output_format": "mp4",
                "outtmpl": str(tmp_dir / "video.%(ext)s"),
            }
            if proxy:
                ydl_opts["proxy"] = proxy
            cookies_path = _get_cookies_file()
            if cookies_path:
                ydl_opts["cookiefile"] = cookies_path
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.to_thread(ydl.extract_info, target_url, True)
            raw_vid = next(tmp_dir.glob("video.*"), None)
            if not raw_vid:
                raise TikTokError("Failed to download TikTok video file for publishing.")
            target = tmp_dir / f"{name}{raw_vid.suffix}"
            raw_vid.rename(target)
            video_file = target

        if include_transcript:
            tr = await get_tiktok_transcript(target_url, language=languages, include_ocr=True)
            meta["transcript"] = tr.get("full_text", "")
            meta["audio_transcript"] = tr.get("audio_text", "")
            meta["ocr_transcript"] = tr.get("ocr_text", "")
            meta["transcript_source"] = tr.get("source", "")
            meta["transcript_language"] = tr.get("language", "")

        meta["video_url"] = await _upload_media_asset(video_file, "video/mp4", meta)

        if meta.get("thumbnail"):
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                r = await client.get(meta["thumbnail"])
            if r.status_code == 200:
                ext = ".webp" if "webp" in r.headers.get("content-type", "") else ".jpg"
                cover = tmp_dir / f"{name}-cover{ext}"
                cover.write_bytes(r.content)
                meta["cover_url"] = await _upload_media_asset(
                    cover, r.headers.get("content-type", "image/jpeg"), meta
                )

        meta["published"] = True
        return meta


def _extract_user_sync(username_or_url: str, limit: int = 10) -> List[Dict[str, Any]]:
    val = username_or_url.strip()
    if val.startswith("http://") or val.startswith("https://"):
        target_url = val
    else:
        target_url = f"https://www.tiktok.com/@{val.lstrip('@')}"

    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlist_items": f"1:{max(1, min(limit, 50))}",
    }
    proxy = _get_proxy()
    if proxy:
        ydl_opts["proxy"] = proxy
    cookies_path = _get_cookies_file()
    if cookies_path:
        ydl_opts["cookiefile"] = cookies_path

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(target_url, download=False)
        except Exception as e:
            raise TikTokError(f"Failed to fetch TikTok user videos: {e}") from e

    results = []
    for entry in info.get("entries") or []:
        results.append(
            {
                "id": entry.get("id"),
                "url": entry.get("url")
                or f"https://www.tiktok.com/@{entry.get('uploader')}/video/{entry.get('id')}",
                "title": entry.get("title") or entry.get("description") or "",
                "uploader": entry.get("uploader") or info.get("uploader") or "",
                "duration": entry.get("duration"),
                "view_count": entry.get("view_count"),
                "like_count": entry.get("like_count"),
                "comment_count": entry.get("comment_count"),
                "repost_count": entry.get("repost_count"),
                "save_count": entry.get("save_count"),
                "track": entry.get("track"),
                "timestamp": entry.get("timestamp"),
            }
        )
    return results


async def get_tiktok_user_videos(username_or_url: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch recent videos published by a TikTok user."""
    return await asyncio.to_thread(_extract_user_sync, username_or_url, limit)
