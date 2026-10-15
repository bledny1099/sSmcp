"""Caption and visual OCR transcript retrieval for YouTube.

Uses a multi-tier resilient architecture:
1. youtube-transcript.ai direct mirror (fast, global, handles all languages).
2. yt-dlp subtitle extraction (manual & auto-captions) with optional proxy/cookies.
3. LemnosLife YouTube API mirror fallback.
4. Faster-Whisper audio transcription + keyframe OCR fallback for uncaptioned or screenshot/slide videos.

All transcripts undergo word-level & phrase-level ASR deduplication to eliminate
rolling repetitive phrases from auto-translated/ASR caption streams, and are cached
in memory so multi-page offset reads are instantaneous.
"""

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
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yt_dlp

from .ocr import extract_video_ocr

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_URL_PATTERNS = [
    re.compile(r"(?:v=|/shorts/|/embed/|youtu\.be/|/live/)([A-Za-z0-9_-]{11})"),
]
_MUSIC_ONLY_RE = re.compile(
    r"^[\s\[\]()♪♫*(?:music|музыка|аплодисменты|applause|смех|laughter)]+$",
    re.IGNORECASE,
)

DEFAULT_CHUNK_CHARS = int(os.getenv("YT_TRANSCRIPT_CHUNK_CHARS", "12000"))
logger = logging.getLogger("ssmcp.transcripts")

_TRANSCRIPT_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]], str]] = {}
_CACHE_TTL = 900.0
_MAX_CACHE_SIZE = 128


class TranscriptError(RuntimeError):
    """Transcript failure with an agent-readable message."""


def extract_video_id(value: str) -> str:
    """Accept a bare 11-char ID or any common YouTube URL shape."""
    value = value.strip()
    if _VIDEO_ID.match(value):
        return value
    for pattern in _URL_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1)
    raise TranscriptError(
        f"Could not parse a video ID from {value!r}. Pass an 11-character ID "
        "(e.g. 'dQw4w9WgXcQ') or a full youtube.com/youtu.be URL."
    )


def _get_proxy() -> Optional[str]:
    """Resolve configured proxy from environment."""
    return (
        os.getenv("YT_TRANSCRIPT_PROXY")
        or (os.getenv("SSMCP_PROXY") or os.getenv("SOCIAL_MCP_PROXY"))
        or os.getenv("HTTPS_PROXY")
        or None
    )


def _get_proxy_candidates() -> List[Optional[str]]:
    """Return ordered list of proxy fallbacks."""
    primary = _get_proxy()
    candidates: List[Optional[str]] = []
    if primary:
        candidates.append(primary)
    candidates.append(None)
    seen = set()
    deduped = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _build_ytdlp_opts(proxy: Optional[str] = None) -> Dict[str, Any]:
    opts: Dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "no_warnings": True,
        "extract_flat": False,
    }
    if proxy:
        opts["proxy"] = proxy

    cookies_path = os.getenv("YT_COOKIES_PATH")
    if cookies_path and os.path.exists(cookies_path):
        opts["cookiefile"] = cookies_path
    elif os.path.exists("/data/cookies.txt"):
        opts["cookiefile"] = "/data/cookies.txt"

    return opts


def _norm_word(w: str) -> str:
    """Normalize word for boundary matching, stripping punctuation and lowercase."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", w).lower()


def _deduplicate_repeated_phrases(text: str) -> str:
    """Eliminate stuttering and duplicate phrases from auto-translated/ASR tracks."""
    words = text.split()
    if not words:
        return ""
    n = len(words)
    i = 0
    res = []
    while i < n:
        matched = False
        for k in range(min(12, (n - i) // 2), 0, -1):
            phrase1 = words[i : i + k]
            phrase2 = words[i + k : i + 2 * k]
            if phrase1 == phrase2:
                res.extend(phrase1)
                i += 2 * k
                while i + k <= n and words[i : i + k] == phrase1:
                    i += k
                matched = True
                break
        if not matched:
            res.append(words[i])
            i += 1
    return " ".join(res)


def _deduplicate_segment_boundaries(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove overlapping words at the boundary between consecutive subtitle cues."""
    for i in range(1, len(segments)):
        prev_words = segments[i - 1]["text"].split()
        curr_words = segments[i]["text"].split()
        if not prev_words or not curr_words:
            continue
        max_overlap = min(len(prev_words), len(curr_words), 8)
        best_k = 0
        for k in range(max_overlap, 0, -1):
            if [_norm_word(w) for w in prev_words[-k:]] == [_norm_word(w) for w in curr_words[:k]]:
                best_k = k
                break
        if best_k > 0:
            segments[i]["text"] = " ".join(curr_words[best_k:])
    return [s for s in segments if s.get("text", "").strip()]


def _is_sparse_or_music_only(segments: List[Dict[str, Any]]) -> bool:
    """Return True if caption segments contain only [Music] tags or fewer than 6 words total."""
    if not segments:
        return True
    full = " ".join(s.get("text", "") for s in segments).strip()
    if not full or _MUSIC_ONLY_RE.match(full):
        return True
    stripped = re.sub(r"\[[^\]]*\]|\([^)]*\)|♪|♫", "", full).strip()
    words = stripped.split()
    return len(words) < 6


def _parse_json3_subtitles(raw_json: str) -> List[Dict[str, Any]]:
    """Parse YouTube JSON3 subtitle format into segment dictionaries."""
    data = json.loads(raw_json)
    events = data.get("events", [])
    segments: List[Dict[str, Any]] = []

    for ev in events:
        segs = ev.get("segs", [])
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        text = html.unescape(re.sub(r"\s+", " ", text)).strip()
        text = _deduplicate_repeated_phrases(text)
        if not text or text == "\n":
            continue

        start_sec = round(ev.get("tStartMs", 0) / 1000.0, 2)
        dur_sec = round(ev.get("dDurationMs", 0) / 1000.0, 2)
        segments.append(
            {
                "start": start_sec,
                "duration": dur_sec,
                "text": text,
            }
        )
    return _deduplicate_segment_boundaries(segments)


def _parse_xml_subtitles(raw_xml: str) -> List[Dict[str, Any]]:
    """Parse standard YouTube XML timedtext into clean segments."""
    import xml.etree.ElementTree as ET

    segments: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(raw_xml)
        for el in root.iter("text"):
            txt = html.unescape(el.text or "").strip()
            txt = _deduplicate_repeated_phrases(re.sub(r"\s+", " ", txt))
            if not txt:
                continue
            start_sec = float(el.attrib.get("start", 0.0))
            dur_sec = float(el.attrib.get("dur", 2.0))
            segments.append(
                {
                    "start": round(start_sec, 2),
                    "duration": round(dur_sec, 2),
                    "text": txt,
                }
            )
    except Exception:
        pass
    return _deduplicate_segment_boundaries(segments)


def _parse_vtt_subtitles(vtt_text: str) -> List[Dict[str, Any]]:
    """Parse WebVTT subtitle format into segment dictionaries."""
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
            match = re.match(
                r"(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})",
                time_line,
            )
            if match:
                h1, m1, s1, ms1 = (
                    int(match.group(1) or 0),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                )
                h2, m2, s2, ms2 = (
                    int(match.group(5) or 0),
                    int(match.group(6)),
                    int(match.group(7)),
                    int(match.group(8)),
                )
                start_sec = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
                end_sec = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
                dur_sec = max(0.1, round(end_sec - start_sec, 2))
                text_lines = lines[time_idx + 1 :]
                seen_lines = set()
                clean_lines = []
                for tl in text_lines:
                    cleaned_tl = html.unescape(re.sub(r"<[^>]+>", "", tl)).strip()
                    cleaned_tl = re.sub(r"\s+", " ", cleaned_tl).strip()
                    if cleaned_tl and cleaned_tl not in seen_lines:
                        seen_lines.add(cleaned_tl)
                        clean_lines.append(cleaned_tl)
                cleaned = _deduplicate_repeated_phrases(" ".join(clean_lines))
                if cleaned:
                    segments.append(
                        {
                            "start": round(start_sec, 2),
                            "duration": dur_sec,
                            "text": cleaned,
                        }
                    )
    return _deduplicate_segment_boundaries(segments)


async def _fetch_youtube_transcript_ai(
    video_id: str, languages: List[str]
) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    """Retrieve clean transcripts via youtube-transcript.ai proxy mirror asynchronously."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
    }

    lang_targets: List[Optional[str]] = [l for l in languages] if languages else [None]
    if None not in lang_targets:
        lang_targets.append(None)

    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=12.0) as client:
        for target_lang in lang_targets:
            url = f"https://youtube-transcript.ai/transcript/{video_id}.txt"
            if target_lang:
                url += f"?lang={target_lang}"

            try:
                r = await client.get(url)
                if r.status_code != 200 or "## Transcript" not in r.text:
                    continue

                detected_lang = target_lang or "en"
                lang_match = re.search(r"Language:\s*([a-zA-Z0-9_-]+)", r.text)
                if lang_match:
                    detected_lang = lang_match.group(1).lower()

                pattern = re.compile(
                    r"^\[(?:(\d+):)?(\d{1,2}):(\d{2})\]\s*(?:-\s*)?(.*)$",
                    re.MULTILINE,
                )
                matches = pattern.findall(r.text)
                if not matches:
                    continue

                segments: List[Dict[str, Any]] = []
                for m in matches:
                    h = int(m[0]) if m[0] else 0
                    minutes = int(m[1])
                    sec = int(m[2])
                    start_sec = float(h * 3600 + minutes * 60 + sec)
                    raw_text = html.unescape(m[3]).strip()
                    clean_text = _deduplicate_repeated_phrases(raw_text)
                    if clean_text:
                        segments.append(
                            {
                                "start": start_sec,
                                "duration": 4.0,
                                "text": clean_text,
                            }
                        )

                for i in range(len(segments) - 1):
                    segments[i]["duration"] = max(
                        1.0, round(segments[i + 1]["start"] - segments[i]["start"], 2)
                    )

                segments = _deduplicate_segment_boundaries(segments)
                if segments:
                    return segments, detected_lang
            except Exception as e:
                logger.debug(f"youtube-transcript.ai failed for {video_id} ({target_lang}): {e}")
                continue

    return None


async def _fetch_lemnoslife(
    video_id: str, languages: List[str]
) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    """Fallback using LemnosLife YouTube operational API asynchronously."""
    url = f"https://yt.lemnoslife.com/videos?part=captions&id={video_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return None
            data = r.json()
            items = data.get("items", [])
            if not items:
                return None
            tracks = (
                items[0]
                .get("captions", {})
                .get("playerCaptionsTracklistRenderer", {})
                .get("captionTracks", [])
            )
            if not tracks:
                return None

            chosen_track = None
            for req in languages:
                for t in tracks:
                    if t.get("languageCode") == req:
                        chosen_track = t
                        break
                if chosen_track:
                    break
            if not chosen_track:
                chosen_track = tracks[0]

            sub_url = chosen_track.get("baseUrl")
            if not sub_url:
                return None

            sub_resp = await client.get(sub_url)
            if sub_resp.status_code == 200 and sub_resp.text.strip():
                raw_text = sub_resp.text.strip()
                if raw_text.startswith("{"):
                    segs = _parse_json3_subtitles(raw_text)
                elif "<transcript" in raw_text or "<text" in raw_text:
                    segs = _parse_xml_subtitles(raw_text)
                else:
                    segs = _parse_vtt_subtitles(raw_text)
                if segs:
                    return segs, chosen_track.get("languageCode", "en")
    except Exception as e:
        logger.debug(f"LemnosLife fetch failed for {video_id}: {e}")
    return None


def _fetch_ytdlp_sync(
    video_id: str, languages: List[str]
) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    """Extract and download captions directly with yt-dlp."""
    target_url = f"https://www.youtube.com/watch?v={video_id}"

    for proxy in _get_proxy_candidates():
        try:
            ydl_opts = _build_ytdlp_opts(proxy)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(target_url, download=False)
            if not info:
                continue

            subs = info.get("subtitles") or {}
            autos = info.get("automatic_captions") or {}
            if not subs and not autos:
                continue

            chosen_lang: Optional[str] = None
            chosen_track: Optional[List[Dict[str, Any]]] = None

            for req_lang in languages:
                if req_lang in subs:
                    chosen_lang = req_lang
                    chosen_track = subs[req_lang]
                    break
                if req_lang in autos:
                    chosen_lang = req_lang
                    chosen_track = autos[req_lang]
                    break

            if not chosen_track:
                for req_lang in languages:
                    prefix = req_lang.split("-")[0].lower()
                    for key in list(subs.keys()) + list(autos.keys()):
                        if key.split("-")[0].lower() == prefix:
                            chosen_lang = key
                            chosen_track = subs.get(key) or autos.get(key)
                            break
                    if chosen_track:
                        break

            if not chosen_track:
                if subs:
                    chosen_lang = next(iter(subs))
                    chosen_track = subs[chosen_lang]
                elif autos:
                    chosen_lang = next(iter(autos))
                    chosen_track = autos[chosen_lang]

            if not chosen_track or not chosen_lang:
                continue

            format_entry = next((f for f in chosen_track if f.get("ext") == "json3"), None)
            if not format_entry:
                format_entry = next(
                    (f for f in chosen_track if f.get("ext") in ("srv3", "vtt")),
                    chosen_track[0],
                )

            sub_url = format_entry.get("url")
            if not sub_url:
                continue

            client_kwargs: Dict[str, Any] = {"timeout": 15.0}
            if proxy and proxy.startswith("http"):
                client_kwargs["proxy"] = proxy

            with httpx.Client(**client_kwargs) as client:
                resp = client.get(sub_url)
                if resp.status_code != 200 or not resp.text.strip():
                    continue

                if format_entry.get("ext") == "json3" or resp.text.strip().startswith("{"):
                    segments = _parse_json3_subtitles(resp.text)
                else:
                    segments = _parse_vtt_subtitles(resp.text)

                if segments:
                    return segments, chosen_lang
        except Exception as e:
            logger.debug(f"yt-dlp fetch failed with proxy {proxy}: {e}")
            continue

    return None


async def _fetch_whisper_and_ocr_fallback(
    video_id: str,
    language: Optional[str] = None,
    include_ocr: bool = True,
) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    """Transcribe audio via faster-whisper and extract on-screen text via keyframe OCR."""
    from .whisper_transcriber import transcribe_media_file

    target_url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        outtmpl = os.path.join(tmpdir, "yt_media.%(ext)s")
        proxy = _get_proxy()
        # Download mp4 (<=720p) when OCR is enabled so keyframes can be extracted
        fmt = "best[height<=720][ext=mp4]/bestaudio/best" if include_ocr else "bestaudio/best"
        ydl_opts: Dict[str, Any] = {
            "quiet": True,
            "format": fmt,
            "outtmpl": outtmpl,
            "no_warnings": True,
        }
        if proxy:
            ydl_opts["proxy"] = proxy
        cookies_path = os.getenv("YT_COOKIES_PATH")
        if cookies_path and os.path.exists(cookies_path):
            ydl_opts["cookiefile"] = cookies_path
        elif os.path.exists("/data/cookies.txt"):
            ydl_opts["cookiefile"] = "/data/cookies.txt"

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, target_url, True)
                downloaded_file = ydl.prepare_filename(info)
                if os.path.exists(downloaded_file):
                    duration = info.get("duration") or 0
                    # Run keyframe OCR on videos up to 10 minutes
                    do_ocr = include_ocr and (duration <= 600)
                    result = await transcribe_media_file(
                        downloaded_file,
                        language=language,
                        include_ocr=do_ocr,
                    )
                    raw_segs = result.get("segments") or []
                    ocr_segs = result.get("ocr_segments") or []

                    segments: List[Dict[str, Any]] = [
                        {
                            "start": s.get("start_seconds", 0.0),
                            "duration": max(
                                1.0,
                                round(
                                    s.get("end_seconds", 0.0) - s.get("start_seconds", 0.0),
                                    2,
                                ),
                            ),
                            "text": s.get("text", ""),
                        }
                        for s in raw_segs
                        if s.get("text")
                    ]
                    segments = _deduplicate_segment_boundaries(segments)

                    # Append on-screen OCR segments if present and distinct from spoken text
                    if ocr_segs:
                        audio_norm = re.sub(
                            r"[^\wа-яё]+",
                            "",
                            " ".join(s["text"] for s in segments).lower(),
                        )
                        for oseg in ocr_segs:
                            otext = (oseg.get("text") or "").strip()
                            onorm = re.sub(r"[^\wа-яё]+", "", otext.lower())
                            if onorm and (not audio_norm or onorm not in audio_norm):
                                segments.append(
                                    {
                                        "start": oseg.get("start_seconds", 0.0),
                                        "duration": max(
                                            1.0,
                                            round(
                                                oseg.get("end_seconds", 2.0)
                                                - oseg.get("start_seconds", 0.0),
                                                2,
                                            ),
                                        ),
                                        "text": f"[Экран: {otext}]" if audio_norm else otext,
                                    }
                                )

                    if segments:
                        segments.sort(key=lambda x: x.get("start", 0.0))
                        return segments, result.get("language", language or "auto")
        except Exception as e:
            logger.debug(f"Whisper+OCR fallback failed for {video_id}: {e}")
    return None


async def _augment_with_ocr_only(
    video_id: str,
    existing_segments: List[Dict[str, Any]],
    language: Optional[str],
) -> List[Dict[str, Any]]:
    """Download video keyframes and append on-screen OCR segments to existing captions."""
    target_url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        outtmpl = os.path.join(tmpdir, "yt_video.%(ext)s")
        proxy = _get_proxy()
        ydl_opts: Dict[str, Any] = {
            "quiet": True,
            "format": "best[height<=720][ext=mp4]/best",
            "outtmpl": outtmpl,
            "no_warnings": True,
        }
        if proxy:
            ydl_opts["proxy"] = proxy
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = await asyncio.to_thread(ydl.extract_info, target_url, True)
                dl_file = ydl.prepare_filename(info)
                if os.path.exists(dl_file):
                    ocr_res = await extract_video_ocr(dl_file, language=language)
                    ocr_segs = ocr_res.get("segments") or []
                    if ocr_segs:
                        combined = list(existing_segments)
                        audio_norm = re.sub(
                            r"[^\wа-яё]+",
                            "",
                            " ".join(s["text"] for s in combined).lower(),
                        )
                        for oseg in ocr_segs:
                            otext = (oseg.get("text") or "").strip()
                            onorm = re.sub(r"[^\wа-яё]+", "", otext.lower())
                            if onorm and (not audio_norm or onorm not in audio_norm):
                                combined.append(
                                    {
                                        "start": oseg.get("start_seconds", 0.0),
                                        "duration": max(
                                            1.0,
                                            round(
                                                oseg.get("end_seconds", 2.0)
                                                - oseg.get("start_seconds", 0.0),
                                                2,
                                            ),
                                        ),
                                        "text": f"[Экран: {otext}]" if audio_norm else otext,
                                    }
                                )
                        combined.sort(key=lambda x: x.get("start", 0.0))
                        return combined
        except Exception as e:
            logger.debug(f"OCR augmentation failed for {video_id}: {e}")
    return existing_segments


async def fetch_transcript(
    video_id: str,
    languages: List[str],
    include_ocr: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """Fetch caption segments off the event loop, with caching and Whisper+OCR fallback."""
    cache_key = f"{video_id}:{','.join(languages)}:ocr={include_ocr}"
    now = time.time()
    cached = _TRANSCRIPT_CACHE.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL:
        return copy.deepcopy(cached[1]), cached[2]

    target_lang = languages[0] if languages else None
    res: Optional[Tuple[List[Dict[str, Any]], str]] = None

    # Tier 1: youtube-transcript.ai mirror
    res = await _fetch_youtube_transcript_ai(video_id, languages)

    # Tier 2: Direct yt-dlp subtitle extraction
    if not res or not res[0]:
        res = await asyncio.to_thread(_fetch_ytdlp_sync, video_id, languages)

    # Tier 3: LemnosLife caption mirror
    if not res or not res[0]:
        res = await _fetch_lemnoslife(video_id, languages)

    # If captions were found, check if they are just "[Music]" or if user requested OCR
    if res and res[0]:
        segs, lang = res
        if include_ocr or _is_sparse_or_music_only(segs):
            segs = await _augment_with_ocr_only(video_id, segs, target_lang)
        _store_cache(cache_key, segs, lang)
        return copy.deepcopy(segs), lang

    # Tier 4: Whisper AI audio transcription + Keyframe OCR fallback
    res_whisper = await _fetch_whisper_and_ocr_fallback(video_id, target_lang, include_ocr=True)
    if res_whisper and res_whisper[0]:
        _store_cache(cache_key, res_whisper[0], res_whisper[1])
        return copy.deepcopy(res_whisper[0]), res_whisper[1]

    raise TranscriptError(
        f"Could not retrieve transcripts for video {video_id}. "
        "No caption tracks, spoken speech, or on-screen text found."
    )


def _store_cache(key: str, segments: List[Dict[str, Any]], lang: str) -> None:
    if len(_TRANSCRIPT_CACHE) >= _MAX_CACHE_SIZE:
        _TRANSCRIPT_CACHE.pop(next(iter(_TRANSCRIPT_CACHE)))
    _TRANSCRIPT_CACHE[key] = (time.time(), segments, lang)


async def list_languages(video_id: str) -> List[Dict[str, Any]]:
    """List available caption tracks for a video."""
    languages: List[Dict[str, Any]] = []
    seen = set()

    # 1. Try youtube-transcript.ai header parse asynchronously
    try:
        url = f"https://youtube-transcript.ai/transcript/{video_id}.txt"
        async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}, timeout=8.0) as client:
            r = await client.get(url)
            if r.status_code == 200:
                m_main = re.search(r"Language:\s*([a-zA-Z0-9_-]+)(?:\s*\(([^)]+)\))?", r.text)
                if m_main:
                    code = m_main.group(1).lower()
                    desc = m_main.group(2) or ""
                    is_gen = "auto" in desc.lower()
                    languages.append(
                        {
                            "language": f"{code} ({desc})" if desc else code,
                            "language_code": code,
                            "generated": is_gen,
                            "translatable": True,
                        }
                    )
                    seen.add(code)

                m_other = re.search(r"Other available languages:\s*(.*)", r.text)
                if m_other:
                    for t in m_other.group(1).split(","):
                        t = t.strip()
                        m_tok = re.search(
                            r"(?:a-)?([a-zA-Z0-9_-]+)\s*(?:\(([^)]+)\))?\s*(\[auto\])?",
                            t,
                        )
                        if m_tok:
                            code = (
                                (m_tok.group(2) if m_tok.group(2) else m_tok.group(1))
                                .lower()
                                .strip()
                            )
                            is_gen = bool(m_tok.group(3)) or t.startswith("a-")
                            if code not in seen:
                                languages.append(
                                    {
                                        "language": code,
                                        "language_code": code,
                                        "generated": is_gen,
                                        "translatable": True,
                                    }
                                )
                                seen.add(code)
                if languages:
                    return languages
    except Exception:
        pass

    # 2. Try yt-dlp
    def _sync_ytdlp() -> List[Dict[str, Any]]:
        target_url = f"https://www.youtube.com/watch?v={video_id}"
        for proxy in _get_proxy_candidates():
            try:
                ydl_opts = _build_ytdlp_opts(proxy)
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(target_url, download=False)
                if not info:
                    continue

                subs = info.get("subtitles") or {}
                autos = info.get("automatic_captions") or {}

                for code, tracks in subs.items():
                    name = tracks[0].get("name") if tracks and tracks[0].get("name") else code
                    languages.append(
                        {
                            "language": name,
                            "language_code": code,
                            "generated": False,
                            "translatable": True,
                        }
                    )
                    seen.add(code)

                for code, tracks in autos.items():
                    if code not in seen:
                        name = tracks[0].get("name") if tracks and tracks[0].get("name") else code
                        languages.append(
                            {
                                "language": name,
                                "language_code": code,
                                "generated": True,
                                "translatable": True,
                            }
                        )
                        seen.add(code)

                if languages:
                    return languages
            except Exception:
                continue
        return languages

    tracks = await asyncio.to_thread(_sync_ytdlp)
    if tracks:
        return tracks

    # 3. Try LemnosLife asynchronously
    try:
        url = f"https://yt.lemnoslife.com/videos?part=captions&id={video_id}"
        async with httpx.AsyncClient(timeout=8.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            r = await client.get(url)
            if r.status_code == 200:
                data = r.json()
                items = data.get("items", [])
                if items:
                    ct_list = (
                        items[0]
                        .get("captions", {})
                        .get("playerCaptionsTracklistRenderer", {})
                        .get("captionTracks", [])
                    )
                    for t in ct_list:
                        c = t.get("languageCode")
                        if c and c not in seen:
                            name_val = t.get("name", {}).get("runs", [{}])[0].get("text") or c
                            is_gen = t.get("kind") == "asr"
                            languages.append(
                                {
                                    "language": name_val,
                                    "language_code": c,
                                    "generated": is_gen,
                                    "translatable": True,
                                }
                            )
                            seen.add(c)
                if languages:
                    return languages
    except Exception:
        pass

    return [
        {
            "language": "AI Whisper + OCR Auto-Transcription",
            "language_code": "auto",
            "generated": True,
            "translatable": False,
        }
    ]


def stitch(
    segments: List[Dict[str, Any]],
    *,
    with_timestamps: bool,
    offset: int,
    chunk_chars: int,
) -> Dict[str, Any]:
    """Join segments into a text chunk starting at a character offset."""
    if with_timestamps:
        pieces = [
            f"[{int(s['start']) // 60}:{int(s['start']) % 60:02d}] {s['text']}"
            for s in segments
        ]
        full = "\n\n".join(pieces)
    else:
        full = " ".join(s["text"].replace("\n", " ") for s in segments)

    total = len(full)
    offset = max(0, min(offset, total))
    end = min(offset + chunk_chars, total)

    if end < total:
        space = full.rfind(" ", offset, end)
        if space > offset:
            end = space

    return {
        "text": full[offset:end],
        "offset": offset,
        "next_offset": end if end < total else None,
        "total_chars": total,
        "segment_count": len(segments),
        "duration_seconds": int(segments[-1]["start"] + segments[-1]["duration"])
        if segments
        else 0,
    }


def default_languages(raw: Optional[List[str]]) -> List[str]:
    if raw:
        return raw
    env = os.getenv("YT_TRANSCRIPT_LANGS", "ru,en")
    return [code.strip() for code in env.split(",") if code.strip()]
