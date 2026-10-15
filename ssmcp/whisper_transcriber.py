"""Whisper audio transcription + visual keyframe OCR service for sSmcp (socialSimple MCP)."""

from __future__ import annotations

import asyncio
import difflib
import inspect
import logging
import os
import re
import tempfile
import threading
from typing import Any, Dict, List, Optional

import httpx

from .ocr import _normalize_for_comparison, extract_video_ocr

logger = logging.getLogger("ssmcp.whisper")

_model_cache: Dict[tuple[str, str, str], Any] = {}
_model_init_lock = threading.Lock()
_whisper_sem = asyncio.Semaphore(
    int(
        os.getenv("SSMCP_WHISPER_CONCURRENCY")
        or os.getenv("WHISPER_CONCURRENCY")
        or os.getenv("TIKTOK_MCP_MAX_CONCURRENT_TRANSCRIBE")
        or "2"
    )
)

# Known Whisper silence/music hallucinations in Russian and English
_HALLUCINATION_PATTERNS = re.compile(
    r"^(?:"
    r"субтитры\s+(?:сделал|подогнал|создавал|создал|подготовил|добавил).*|"
    r"редактор\s+субтитров.*|"
    r"корректор\s+.*|"
    r"перевод\s+и\s+озвучка.*|"
    r"продолжение\s+следует\.?|"
    r"подписывайтесь\s+на\s+канал\.?|"
    r"subtitles?\s+by\s+.*|"
    r"amara\.org.*|"
    r"thank\s+you\s+for\s+watching\.?|"
    r"thanks\s+for\s+watching\.?"
    r")$",
    re.IGNORECASE,
)


class TranscriptionError(RuntimeError):
    """Transcription error phrased for the agent."""


def _default_model_size() -> str:
    return (
        os.getenv("SSMCP_WHISPER_MODEL")
        or os.getenv("WHISPER_MODEL")
        or os.getenv("TIKTOK_MCP_WHISPER_MODEL")
        or "base"
    )


def _default_device() -> str:
    return (
        os.getenv("SSMCP_WHISPER_DEVICE")
        or os.getenv("WHISPER_DEVICE")
        or os.getenv("TIKTOK_MCP_WHISPER_DEVICE")
        or "cpu"
    )


def _default_compute_type() -> str:
    return (
        os.getenv("SSMCP_WHISPER_COMPUTE_TYPE")
        or os.getenv("WHISPER_COMPUTE_TYPE")
        or os.getenv("TIKTOK_MCP_WHISPER_COMPUTE")
        or "int8"
    )


def get_whisper_model(model_size: Optional[str] = None):
    """Thread-safe lazy initialization of faster-whisper model with caching by size."""
    size = model_size or _default_model_size()
    device = _default_device()
    compute_type = _default_compute_type()
    key = (size, device, compute_type)

    if key not in _model_cache:
        with _model_init_lock:
            if key not in _model_cache:
                from faster_whisper import WhisperModel

                cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "4"))
                download_root = os.getenv("SSMCP_MODELS_DIR") or None
                kwargs: Dict[str, Any] = {
                    "device": device,
                    "compute_type": compute_type,
                    "cpu_threads": cpu_threads,
                }
                if download_root:
                    kwargs["download_root"] = download_root
                _model_cache[key] = WhisperModel(size, **kwargs)
    return _model_cache[key]


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Discard parameters unsupported by the installed version of faster-whisper."""
    allowed = set(inspect.signature(fn).parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}


def _format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm or MM:SS.mmm string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    ms = int((seconds - int(seconds)) * 1000)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def _is_hallucination(text: str, no_speech_prob: float) -> bool:
    clean = text.strip()
    if not clean:
        return True
    if no_speech_prob > 0.80:
        return True
    if _HALLUCINATION_PATTERNS.match(clean):
        return True
    return False


def _normalize_lang_list(language: Optional[str | List[str]]) -> List[str]:
    if not language:
        return []
    if isinstance(language, list):
        return [l.split("-")[0].strip().lower() for l in language if l and l.strip()]
    return [l.split("-")[0].strip().lower() for l in re.split(r"[,+]", language) if l.strip()]


def _run_whisper_pass(
    model: Any,
    file_path: str,
    language: Optional[str | List[str]],
    vad_filter: bool,
) -> tuple[List[Dict[str, Any]], Any]:
    langs = _normalize_lang_list(language)
    kwargs: Dict[str, Any] = {
        "beam_size": 5,
        "vad_filter": vad_filter,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
    }
    if len(langs) == 1:
        kwargs["language"] = langs[0]
    elif len(langs) > 1:
        kwargs["multilingual"] = True
    if vad_filter:
        kwargs["vad_parameters"] = {"min_silence_duration_ms": 500}

    kwargs = _filter_kwargs(model.transcribe, kwargs)
    segments_gen, info = model.transcribe(file_path, **kwargs)
    segments: List[Dict[str, Any]] = []
    for seg in segments_gen:
        clean_text = seg.text.strip()
        no_speech = getattr(seg, "no_speech_prob", 0.0) or 0.0
        if _is_hallucination(clean_text, no_speech):
            continue
        segments.append(
            {
                "start": _format_timestamp(seg.start),
                "end": _format_timestamp(seg.end),
                "start_seconds": round(seg.start, 2),
                "end_seconds": round(seg.end, 2),
                "text": clean_text,
            }
        )
    return segments, info


def _transcribe_file_sync(
    file_path: str,
    language: Optional[str | List[str]] = None,
    vad_filter: bool = True,
    model_size: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe a local audio or video file with Whisper, with VAD fallback."""
    model = get_whisper_model(model_size)
    try:
        segments, info = _run_whisper_pass(model, file_path, language, vad_filter=vad_filter)
        if not segments and vad_filter and (getattr(info, "duration", 0) or 0) <= 120:
            segments, info = _run_whisper_pass(model, file_path, language, vad_filter=False)

        full_text = " ".join(s["text"] for s in segments)
        return {
            "language": getattr(info, "language", None),
            "language_probability": round(getattr(info, "language_probability", 0.0) or 0.0, 3),
            "duration": round(getattr(info, "duration", 0.0) or 0.0, 2),
            "model": model_size or _default_model_size(),
            "full_text": full_text,
            "segments": segments,
        }
    except Exception as e:
        raise TranscriptionError(f"Whisper transcription failed: {e}") from e


async def transcribe_media_file(
    file_path: str,
    language: Optional[str | List[str]] = None,
    vad_filter: bool = True,
    include_ocr: bool = True,
    model_size: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe a local media file with Whisper and (optionally) visual keyframe OCR in parallel."""
    async def _run_audio() -> Dict[str, Any]:
        async with _whisper_sem:
            return await asyncio.to_thread(
                _transcribe_file_sync, file_path, language, vad_filter, model_size
            )

    if not include_ocr:
        return await _run_audio()

    audio_task = asyncio.create_task(_run_audio())
    ocr_task = asyncio.create_task(extract_video_ocr(file_path, language=language))

    audio_res: Dict[str, Any] = {}
    ocr_res: Dict[str, Any] = {}

    try:
        audio_res = await audio_task
    except Exception as e:
        logger.debug(f"Audio transcription failed on {file_path}, relying on OCR if available: {e}")

    try:
        ocr_res = await ocr_task
    except Exception as e:
        logger.debug(f"Video OCR extraction failed on {file_path}: {e}")

    if not audio_res and not ocr_res.get("ocr_available"):
        raise TranscriptionError("Both audio transcription and visual frame OCR produced no results.")

    langs = _normalize_lang_list(language)
    return merge_audio_and_ocr(audio_res, ocr_res, default_lang=langs[0] if langs else None)


def merge_audio_and_ocr(
    audio_res: Dict[str, Any],
    ocr_res: Dict[str, Any],
    default_lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Combine spoken Whisper output and visual keyframe OCR into a unified transcript, avoiding subtitle duplication."""
    audio_text = (audio_res.get("full_text") or "").strip()
    ocr_text = (ocr_res.get("full_text") or "").strip()
    audio_segments = audio_res.get("segments") or []
    ocr_segments = ocr_res.get("segments") or []

    if audio_text and ocr_text:
        norm_a = _normalize_for_comparison(audio_text)
        norm_o = _normalize_for_comparison(ocr_text)
        if norm_o and (
            norm_o in norm_a
            or difflib.SequenceMatcher(None, norm_a, norm_o).ratio() >= 0.72
        ):
            combined_text = audio_text
            source = "whisper_ai"
        else:
            # Filter out individual OCR segments that merely repeat spoken lines
            unique_ocr_blocks: List[str] = []
            for seg in ocr_segments:
                seg_txt = (seg.get("text") or "").strip()
                seg_norm = _normalize_for_comparison(seg_txt)
                if not seg_norm or seg_norm in norm_a:
                    continue
                is_sub = any(
                    difflib.SequenceMatcher(
                        None, seg_norm, _normalize_for_comparison(a_seg.get("text", ""))
                    ).ratio()
                    >= 0.75
                    for a_seg in audio_segments
                )
                if not is_sub:
                    unique_ocr_blocks.append(seg_txt)

            if unique_ocr_blocks:
                filtered_ocr = "\n\n".join(unique_ocr_blocks)
                combined_text = f"{audio_text}\n\n[On-Screen Text (OCR)]\n{filtered_ocr}"
                source = "whisper_ai+ocr"
            else:
                combined_text = audio_text
                source = "whisper_ai"
    elif ocr_text:
        combined_text = ocr_text
        source = "ocr_frames"
    else:
        combined_text = audio_text
        source = "whisper_ai"

    return {
        "language": audio_res.get("language") or default_lang or "auto",
        "language_probability": audio_res.get("language_probability", 1.0 if ocr_text else 0.0),
        "duration": audio_res.get("duration"),
        "model": audio_res.get("model") or _default_model_size(),
        "source": source,
        "full_text": combined_text,
        "audio_text": audio_text,
        "ocr_text": ocr_text,
        "segments": audio_segments if audio_segments else ocr_segments,
        "ocr_segments": ocr_segments,
        "frames_analyzed": ocr_res.get("frames_analyzed", 0),
    }


async def _download_stream_to_file(
    url: str, suffix: str = ".mp4", proxy: Optional[str] = None
) -> str:
    """Stream-download a URL to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.tiktok.com/",
    }
    client_kwargs: Dict[str, Any] = {
        "timeout": 60.0,
        "follow_redirects": True,
        "headers": headers,
    }
    if proxy and (proxy.startswith("http") or proxy.startswith("socks")):
        client_kwargs["proxy"] = proxy

    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    raise TranscriptionError(
                        f"Failed to download media stream: HTTP {resp.status_code}"
                    )
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
        return tmp_path
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


async def download_and_transcribe_stream(
    stream_url: Optional[str],
    language: Optional[str | List[str]] = None,
    vad_filter: bool = True,
    include_ocr: bool = True,
    video_stream_url: Optional[str] = None,
    proxy: Optional[str] = None,
    model_size: Optional[str] = None,
) -> Dict[str, Any]:
    """Stream-download remote media and run Whisper audio transcription + keyframe OCR."""
    audio_url = stream_url or video_stream_url
    vid_url = video_stream_url or stream_url

    if not audio_url and not vid_url:
        raise TranscriptionError("No media stream URL provided for transcription.")

    if not include_ocr or not vid_url or audio_url == vid_url:
        target = audio_url or vid_url
        assert target is not None
        tmp_path = await _download_stream_to_file(target, suffix=".mp4", proxy=proxy)
        try:
            return await transcribe_media_file(
                tmp_path,
                language=language,
                vad_filter=vad_filter,
                include_ocr=include_ocr,
                model_size=model_size,
            )
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    try:
        dl_results = await asyncio.gather(
            _download_stream_to_file(audio_url, suffix=".m4a", proxy=proxy),
            _download_stream_to_file(vid_url, suffix=".mp4", proxy=proxy),
            return_exceptions=True,
        )
        if isinstance(dl_results[0], str):
            audio_path = dl_results[0]
        if isinstance(dl_results[1], str):
            video_path = dl_results[1]

        if not audio_path and not video_path:
            raise TranscriptionError("Failed to download both audio and video streams.")

        async def _run_audio() -> Dict[str, Any]:
            target_file = audio_path or video_path
            if not target_file:
                return {}
            async with _whisper_sem:
                return await asyncio.to_thread(
                    _transcribe_file_sync, target_file, language, vad_filter, model_size
                )

        async def _run_ocr() -> Dict[str, Any]:
            if not video_path:
                return {
                    "ocr_available": False,
                    "full_text": "",
                    "segments": [],
                    "frames_analyzed": 0,
                }
            return await extract_video_ocr(video_path, language=language)

        audio_res, ocr_res = await asyncio.gather(
            _run_audio(), _run_ocr(), return_exceptions=True
        )
        if isinstance(audio_res, Exception):
            logger.debug(f"Audio stream transcription failed: {audio_res}")
            audio_res = {}
        if isinstance(ocr_res, Exception):
            logger.debug(f"Video stream OCR failed: {ocr_res}")
            ocr_res = {
                "ocr_available": False,
                "full_text": "",
                "segments": [],
                "frames_analyzed": 0,
            }

        if not audio_res and not ocr_res.get("ocr_available"):
            raise TranscriptionError(
                "Both audio transcription and visual frame OCR produced no results."
            )

        langs = _normalize_lang_list(language)
        return merge_audio_and_ocr(
            audio_res, ocr_res, default_lang=langs[0] if langs else None
        )
    finally:
        for p in (audio_path, video_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
