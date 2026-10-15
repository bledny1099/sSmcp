"""Threads data extractor: metadata, posts, media streams, creator feeds, search, replies, and media OCR/transcription."""

from __future__ import annotations

import asyncio
import copy
import html
import json
import logging
import os
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional

from curl_cffi import requests

from .ocr import extract_images_ocr
from .whisper_transcriber import download_and_transcribe_stream, merge_audio_and_ocr

logger = logging.getLogger("ssmcp.threads")

_POST_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL = 300.0
_MAX_CACHE_SIZE = 128


class ThreadsError(RuntimeError):
    """Threads extraction error phrased for the agent."""


def _get_proxy() -> Optional[str]:
    return (
        (os.getenv("SSMCP_PROXY") or os.getenv("SOCIAL_MCP_PROXY"))
        or os.getenv("THREADS_PROXY")
        or os.getenv("YT_TRANSCRIPT_PROXY")
        or os.getenv("HTTPS_PROXY")
        or None
    )


def normalize_threads_url(url_or_code: str) -> tuple[str, Optional[str]]:
    """Normalize any Threads URL, handle, or post code into canonical URL and code."""
    val = url_or_code.strip()
    if not val:
        raise ThreadsError("Threads URL or identifier cannot be empty.")

    if val.startswith("http://") or val.startswith("https://"):
        clean = re.sub(r"\?[^#]*", "", val).rstrip("/")
        post_match = re.search(r"/(?:post|t)/([A-Za-z0-9_-]+)", clean)
        code = post_match.group(1) if post_match else None
        return clean, code

    if not val.startswith("@") and "/" not in val and len(val) >= 9:
        return f"https://www.threads.net/t/{val}", val

    clean_handle = val.lstrip("@").strip("/")
    return f"https://www.threads.net/@{clean_handle}", None


def _extract_media_urls(post_dict: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """Extract all video stream URLs and image URLs from a Threads post object."""
    videos: List[str] = []
    images: List[str] = []

    for v in post_dict.get("video_versions") or []:
        if v.get("url") and v["url"] not in videos:
            videos.append(v["url"])

    candidates = (post_dict.get("image_versions2") or {}).get("candidates") or []
    if candidates and candidates[0].get("url") and candidates[0]["url"] not in images:
        images.append(candidates[0]["url"])

    for item in post_dict.get("carousel_media") or []:
        for v in item.get("video_versions") or []:
            if v.get("url") and v["url"] not in videos:
                videos.append(v["url"])
        item_cands = (item.get("image_versions2") or {}).get("candidates") or []
        if item_cands and item_cands[0].get("url") and item_cands[0]["url"] not in images:
            images.append(item_cands[0]["url"])

    return videos, images


def _find_target_post(obj: Any, code_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Recursively search for a post object matching code_hint or main post structure."""
    if isinstance(obj, dict):
        if code_hint and obj.get("code") == code_hint:
            return obj
        if not code_hint and "code" in obj and "caption" in obj and "user" in obj:
            return obj
        for v in obj.values():
            res = _find_target_post(v, code_hint)
            if res:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = _find_target_post(item, code_hint)
            if res:
                return res
    return None


def _find_all_thread_items(obj: Any) -> List[Dict[str, Any]]:
    """Recursively collect all thread_items containers from Relay preloaded tree."""
    results: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if "thread_items" in obj and isinstance(obj["thread_items"], list):
            results.append(obj)
        for v in obj.values():
            results.extend(_find_all_thread_items(v))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_find_all_thread_items(item))
    return results


def _extract_comments_from_scripts(
    scripts: List[str], target_code: str, limit: int = 15
) -> List[Dict[str, Any]]:
    """Recursively find reply posts and comments in Threads embedded JSON scripts."""
    comments: List[Dict[str, Any]] = []
    seen_codes = {target_code}

    def find_posts_recursive(obj: Any, acc: List[Dict[str, Any]]):
        if isinstance(obj, dict):
            if (
                "caption" in obj
                and isinstance(obj["caption"], dict)
                and "user" in obj
                and isinstance(obj["user"], dict)
            ):
                code = obj.get("code")
                if code and code not in seen_codes:
                    seen_codes.add(code)
                    user = obj["user"]
                    caption_text = obj["caption"].get("text") or ""
                    vids, imgs = _extract_media_urls(obj)
                    author = user.get("username") or ""
                    acc.append(
                        {
                            "id": str(obj.get("pk") or code),
                            "code": code,
                            "url": (
                                f"https://www.threads.net/@{author}/post/{code}"
                                if author
                                else f"https://www.threads.net/t/{code}"
                            ),
                            "author": author,
                            "author_name": user.get("full_name") or author,
                            "author_avatar": user.get("profile_pic_url"),
                            "is_verified": bool(user.get("is_verified")),
                            "text": caption_text,
                            "like_count": obj.get("like_count", 0),
                            "reply_count": obj.get("reply_count", 0),
                            "taken_at": obj.get("taken_at"),
                            "direct_video_urls": vids,
                            "direct_image_urls": imgs,
                        }
                    )
            for v in obj.values():
                find_posts_recursive(v, acc)
        elif isinstance(obj, list):
            for item in obj:
                find_posts_recursive(item, acc)

    for sc in scripts:
        if '"caption":' in sc and '"user":' in sc:
            try:
                data = json.loads(sc)
                find_posts_recursive(data, comments)
                if len(comments) >= limit:
                    break
            except Exception:
                pass

    return comments[:limit]


def _fetch_threads_page(url: str) -> str:
    """Fetch raw HTML of a Threads page with Chrome TLS fingerprint, proxy and optional cookies."""
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Dest": "document",
    }

    session_id = os.getenv("THREADS_SESSION_ID") or os.getenv("INSTAGRAM_SESSION_ID")
    if session_id:
        headers["Cookie"] = f"sessionid={session_id}"

    cookies_path = os.getenv("THREADS_COOKIES_PATH") or ".session/session.json"
    if os.path.exists(cookies_path):
        try:
            with open(cookies_path, "r", encoding="utf-8") as f:
                cookie_data = json.load(f)
                cookies_list = (
                    cookie_data.get("cookies", [])
                    if isinstance(cookie_data, dict)
                    else cookie_data
                )
                cookie_parts = []
                for c in cookies_list:
                    if isinstance(c, dict) and "name" in c and "value" in c:
                        cookie_parts.append(f"{c['name']}={c['value']}")
                if cookie_parts:
                    headers["Cookie"] = "; ".join(cookie_parts)
        except Exception:
            pass

    proxy = _get_proxy()
    resp = None
    last_exc: Optional[Exception] = None

    for p in ([proxy, None] if proxy else [None]):
        try:
            s = requests.Session(impersonate="chrome")
            if p:
                s.proxies = {"http": p, "https": p}
            r = s.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                resp = r
                break
            resp = r
        except Exception as e:
            last_exc = e

    if not resp:
        raise ThreadsError(f"Failed to connect to Threads: {last_exc}")

    if resp.status_code == 404:
        raise ThreadsError(f"Threads page not found (404): {url}")
    if resp.status_code != 200:
        raise ThreadsError(f"Threads returned HTTP status {resp.status_code} for {url}")

    return resp.text


def _parse_post_sync(
    url: str,
    code_hint: Optional[str],
    include_comments: bool = False,
    comments_limit: int = 10,
) -> Dict[str, Any]:
    cache_key = f"{url}:{include_comments}:{comments_limit}"
    now = time.time()
    cached = _POST_CACHE.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL:
        return copy.deepcopy(cached[1])

    html_text = _fetch_threads_page(url)
    scripts = re.findall(r'<script type="application/json"[^>]*>(.+?)</script>', html_text)

    for sc in scripts:
        if code_hint and code_hint not in sc:
            continue
        if '"post":' in sc or '"caption":' in sc:
            try:
                data = json.loads(sc)
                target = _find_target_post(data, code_hint)
                if target:
                    user = target.get("user") or {}
                    caption_obj = target.get("caption") or {}
                    text_content = caption_obj.get("text") or ""
                    videos, images = _extract_media_urls(target)
                    code = target.get("code") or code_hint or ""
                    author = user.get("username") or ""

                    comments: List[Dict[str, Any]] = []
                    if include_comments:
                        comments = _extract_comments_from_scripts(
                            scripts, code, limit=comments_limit
                        )

                    canonical_url = (
                        f"https://www.threads.net/@{author}/post/{code}"
                        if author
                        else url
                    )
                    res = {
                        "id": str(target.get("pk") or code),
                        "code": code,
                        "url": canonical_url,
                        "author": author,
                        "author_name": user.get("full_name") or author,
                        "author_avatar": user.get("profile_pic_url"),
                        "is_verified": bool(user.get("is_verified")),
                        "text": text_content,
                        "like_count": target.get("like_count"),
                        "reply_count": target.get("reply_count")
                        or target.get("direct_reply_count"),
                        "taken_at": target.get("taken_at"),
                        "direct_video_urls": videos,
                        "direct_image_urls": images,
                        "comments": comments,
                        "comments_count": len(comments),
                    }
                    if len(_POST_CACHE) >= _MAX_CACHE_SIZE:
                        _POST_CACHE.pop(next(iter(_POST_CACHE)))
                    _POST_CACHE[cache_key] = (time.time(), res)
                    return copy.deepcopy(res)
            except Exception:
                pass

    raise ThreadsError(
        f"Could not extract post content from {url}. Post may be private or deleted."
    )


def _parse_user_sync(url: str, limit: int = 10) -> Dict[str, Any]:
    html_text = _fetch_threads_page(url)
    scripts = re.findall(r'<script type="application/json"[^>]*>(.+?)</script>', html_text)

    user_info: Dict[str, Any] = {}
    posts: List[Dict[str, Any]] = []
    seen_codes = set()

    for sc in scripts:
        if '"thread_items":' in sc:
            try:
                data = json.loads(sc)
                containers = _find_all_thread_items(data)
                for container in containers:
                    for ti in container.get("thread_items", []):
                        post = ti.get("post") or {}
                        code = post.get("code")
                        if not code or code in seen_codes:
                            continue
                        user = post.get("user") or {}
                        if not user_info and user.get("username"):
                            user_info = {
                                "username": user.get("username"),
                                "full_name": user.get("full_name") or user.get("username"),
                                "profile_pic_url": user.get("profile_pic_url"),
                                "is_verified": bool(user.get("is_verified")),
                                "pk": str(user.get("pk") or ""),
                            }

                        caption_obj = post.get("caption") or {}
                        text_val = caption_obj.get("text") or ""
                        videos, images = _extract_media_urls(post)

                        author = user.get("username") or user_info.get("username") or ""
                        seen_codes.add(code)
                        posts.append(
                            {
                                "id": str(post.get("pk") or code),
                                "code": code,
                                "url": f"https://www.threads.net/@{author}/post/{code}",
                                "author": author,
                                "author_name": user.get("full_name") or author,
                                "text": text_val,
                                "like_count": post.get("like_count"),
                                "reply_count": post.get("reply_count")
                                or post.get("direct_reply_count"),
                                "taken_at": post.get("taken_at"),
                                "direct_video_urls": videos,
                                "direct_image_urls": images,
                            }
                        )
                        if len(posts) >= limit:
                            break
                    if len(posts) >= limit:
                        break
            except Exception:
                pass

    if not user_info and not posts:
        title_m = re.search(r"<title>(.*?)</title>", html_text)
        page_title = html.unescape(title_m.group(1)).strip() if title_m else ""
        if "• Threads" in page_title:
            user_info = {"title": page_title}

    return {
        "user": user_info,
        "posts": posts[:limit],
    }


def _parse_search_sync(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search Threads for public posts by topic, query, or hashtag."""
    clean_q = query.strip()
    encoded = urllib.parse.quote(clean_q)
    url = f"https://www.threads.net/search?q={encoded}&serp_type=default"

    html_text = _fetch_threads_page(url)
    scripts = re.findall(r'<script type="application/json"[^>]*>(.+?)</script>', html_text)

    posts: List[Dict[str, Any]] = []
    seen_codes = set()

    for sc in scripts:
        if '"thread_items":' in sc:
            try:
                data = json.loads(sc)
                containers = _find_all_thread_items(data)
                for container in containers:
                    for ti in container.get("thread_items", []):
                        post = ti.get("post") or {}
                        code = post.get("code")
                        if not code or code in seen_codes:
                            continue
                        seen_codes.add(code)
                        user = post.get("user") or {}
                        caption_obj = post.get("caption") or {}
                        text_val = caption_obj.get("text") or ""
                        videos, images = _extract_media_urls(post)
                        author = user.get("username") or ""

                        posts.append(
                            {
                                "id": str(post.get("pk") or code),
                                "code": code,
                                "url": f"https://www.threads.net/@{author}/post/{code}",
                                "author": author,
                                "author_name": user.get("full_name") or author,
                                "author_avatar": user.get("profile_pic_url"),
                                "is_verified": bool(user.get("is_verified")),
                                "text": text_val,
                                "like_count": post.get("like_count"),
                                "reply_count": post.get("reply_count")
                                or post.get("direct_reply_count"),
                                "taken_at": post.get("taken_at"),
                                "direct_video_urls": videos,
                                "direct_image_urls": images,
                            }
                        )
                        if len(posts) >= limit:
                            break
                    if len(posts) >= limit:
                        break
            except Exception:
                pass

    return posts[:limit]


async def get_threads_post(
    url_or_code: str,
    include_comments: bool = False,
    comments_limit: int = 10,
    include_ocr: bool = False,
) -> Dict[str, Any]:
    """Fetch single Threads post asynchronously without login, optionally with comments and media OCR."""
    target_url, code = normalize_threads_url(url_or_code)
    data = await asyncio.to_thread(
        _parse_post_sync,
        target_url,
        code,
        include_comments=include_comments,
        comments_limit=comments_limit,
    )
    if include_ocr:
        proxy = _get_proxy()
        videos = data.get("direct_video_urls") or []
        images = data.get("direct_image_urls") or []
        media_texts: List[str] = []

        if videos:
            try:
                vid_res = await download_and_transcribe_stream(
                    stream_url=videos[0],
                    video_stream_url=videos[0],
                    include_ocr=True,
                    proxy=proxy,
                )
                if vid_res.get("full_text"):
                    media_texts.append(vid_res["full_text"])
            except Exception as e:
                logger.debug(f"Threads video transcription failed: {e}")

        if images:
            try:
                img_ocr = await extract_images_ocr(images, proxy=proxy)
                if img_ocr.get("ocr_available"):
                    media_texts.append(img_ocr["full_text"])
            except Exception as e:
                logger.debug(f"Threads image OCR failed: {e}")

        if media_texts:
            data["media_transcript"] = "\n\n".join(media_texts)

    return data


async def get_threads_user_posts(username_or_url: str, limit: int = 10) -> Dict[str, Any]:
    """Fetch creator profile and recent Threads posts asynchronously without login."""
    val = username_or_url.strip()
    if val.startswith("http://") or val.startswith("https://"):
        target_url = val
    else:
        clean_handle = val.lstrip("@").strip("/")
        target_url = f"https://www.threads.net/@{clean_handle}"

    return await asyncio.to_thread(_parse_user_sync, target_url, limit)


async def search_threads_posts(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search public Threads posts by keyword, phrase, or hashtag asynchronously."""
    return await asyncio.to_thread(_parse_search_sync, query, limit)
