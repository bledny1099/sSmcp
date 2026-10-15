"""Visual OCR extraction for videos with screenshots/slides and image carousels.

Extracts on-screen text from:
1. Video files or streams: scene-change + interval keyframe extraction via ffmpeg,
   center-crop perceptual deduplication (ignoring corner watermarks), dark-mode
   inversion + contrast preprocessing via Pillow, and Tesseract TSV OCR with
   confidence filtering and cross-frame fuzzy line deduplication.
2. Image carousels / photo posts: concurrent image download and per-slide OCR.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger("ssmcp.ocr")

_OCR_SEM = asyncio.Semaphore(
    int(os.getenv("SSMCP_OCR_CONCURRENCY") or os.getenv("OCR_CONCURRENCY") or "4")
)
_TESSERACT_BIN: Optional[str] = shutil.which("tesseract")
_FFMPEG_BIN: Optional[str] = shutil.which("ffmpeg")
_AVAILABLE_LANGS: Optional[set[str]] = None

_WATERMARK_RE = re.compile(
    r"^(?:tiktok|reels|instagram|youtube\s*shorts|capcut|inshot|vn)\b",
    re.IGNORECASE,
)


def is_ocr_available() -> bool:
    """Return True if tesseract binary is installed on the system."""
    global _TESSERACT_BIN
    if _TESSERACT_BIN is None:
        _TESSERACT_BIN = shutil.which("tesseract")
    return bool(_TESSERACT_BIN)


def _resolve_tesseract_langs(preferred: Optional[str | List[str]] = None) -> str:
    """Resolve Tesseract language string (default 'rus+eng') against installed packs."""
    global _AVAILABLE_LANGS, _TESSERACT_BIN
    if not _TESSERACT_BIN:
        _TESSERACT_BIN = shutil.which("tesseract")
    if not _TESSERACT_BIN:
        return "eng"

    if _AVAILABLE_LANGS is None:
        try:
            proc = subprocess.run(
                [_TESSERACT_BIN, "--list-langs"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = [
                ln.strip()
                for ln in (proc.stdout + "\n" + proc.stderr).splitlines()
                if ln.strip()
            ]
            _AVAILABLE_LANGS = {ln for ln in lines if not ln.lower().startswith("list of")}
        except Exception:
            _AVAILABLE_LANGS = {"eng", "rus"}

    if isinstance(preferred, list):
        target = "+".join(preferred)
    else:
        target = (
            preferred
            or os.getenv("SSMCP_OCR_LANGUAGES")
            or os.getenv("OCR_LANGUAGES")
            or os.getenv("TIKTOK_MCP_OCR_LANGUAGES")
            or "rus+eng"
        )

    lang_map = {
        "ru": "rus",
        "ru-ru": "rus",
        "en": "eng",
        "eng-us": "eng",
        "en-us": "eng",
        "uk": "ukr",
        "de": "deu",
        "fr": "fra",
        "es": "spa",
    }
    requested = []
    for part in re.split(r"[+,]", target):
        clean = part.strip().lower()
        if not clean or clean == "auto":
            continue
        mapped = lang_map.get(clean, clean)
        if mapped not in requested:
            requested.append(mapped)

    for fallback in ("rus", "eng"):
        if fallback not in requested:
            requested.append(fallback)

    valid = [lang for lang in requested if lang in _AVAILABLE_LANGS]
    if not valid:
        return "eng"
    return "+".join(valid)


def _format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS.mmm or HH:MM:SS.mmm."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    ms = int((seconds - int(seconds)) * 1000)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def _compute_center_dhash(img_path: str) -> Optional[Tuple[int, float]]:
    """Compute a 256-bit difference hash and mean brightness on the central 80% of an image."""
    try:
        from PIL import Image, ImageStat

        with Image.open(img_path) as im:
            gray = im.convert("L")
            w, h = gray.size
            if w > 40 and h > 40:
                crop_box = (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))
                center = gray.crop(crop_box)
            else:
                center = gray

            mean_lum = ImageStat.Stat(center).mean[0]
            resized = center.resize((17, 16), Image.Resampling.BILINEAR)
            pixels = list(resized.getdata())
            bits = 0
            for row in range(16):
                row_offset = row * 17
                for col in range(16):
                    bits = (bits << 1) | (
                        1 if pixels[row_offset + col] > pixels[row_offset + col + 1] else 0
                    )
            return bits, mean_lum
    except Exception:
        return None


def _hamming_distance(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def _prepare_ocr_variants(img_path: str, tmp_dir: str) -> List[str]:
    """Create grayscale (and inverted dark-mode if dark) variants for OCR without clipping text pixels."""
    try:
        from PIL import Image, ImageOps, ImageStat

        with Image.open(img_path) as im:
            gray = im.convert("L")
            w, h = gray.size
            if w < 1000:
                scale = min(2.0, 1200.0 / max(1, w))
                gray = gray.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            mean_lum = ImageStat.Stat(gray).mean[0]
            base_name = Path(img_path).stem

            # Full-range autocontrast with cutoff=0 so sparse text pixels (<1% area) are never clipped
            norm = ImageOps.autocontrast(gray, cutoff=0)
            norm_path = os.path.join(tmp_dir, f"{base_name}_norm.png")
            norm.save(norm_path)

            # Dark-mode or mixed-contrast screenshot: invert so white text becomes black on white
            if mean_lum < 140:
                inv = ImageOps.autocontrast(ImageOps.invert(gray), cutoff=0)
                inv_path = os.path.join(tmp_dir, f"{base_name}_inv.png")
                inv.save(inv_path)
                return [inv_path, norm_path] if mean_lum < 110 else [norm_path, inv_path]

            return [norm_path]
    except Exception:
        return [img_path]


def _is_valid_token(word: str, conf: float) -> bool:
    """Filter out OCR noise tokens while preserving real Russian/English words and numbers."""
    w = word.strip()
    if not w:
        return False
    if conf < 42:
        return False

    alnum = re.findall(r"[A-Za-zА-Яа-яЁё0-9]", w)
    if not alnum:
        return False

    if len(w) == 1:
        if w.lower() in {"в", "и", "с", "к", "о", "у", "я", "а", "i", "a"} or w.isdigit():
            return conf >= 55
        return conf >= 80

    if len(w) == 2 and conf < 55:
        return False

    if len(alnum) == 1 and len(w) >= 3:
        return False

    return True


def _run_tesseract_tsv_sync(img_path: str, lang_str: str) -> Tuple[str, float, int]:
    """Run Tesseract in TSV mode and return (cleaned_text, weighted_score, word_count)."""
    if not _TESSERACT_BIN:
        return "", 0.0, 0

    cmd = [
        _TESSERACT_BIN,
        img_path,
        "stdout",
        "-l",
        lang_str,
        "--oem",
        "1",
        "--psm",
        "6",
        "tsv",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if proc.returncode != 0 or not proc.stdout:
            return "", 0.0, 0
    except Exception:
        return "", 0.0, 0

    lines_map: Dict[Tuple[int, int, int], List[str]] = {}
    confs: List[float] = []

    for row in proc.stdout.splitlines()[1:]:
        parts = row.split("\t")
        if len(parts) < 12:
            continue
        try:
            level = int(parts[0])
            block_num = int(parts[2])
            par_num = int(parts[3])
            line_num = int(parts[4])
            conf = float(parts[10])
            text = parts[11].strip()
        except ValueError:
            continue

        if level != 5 or not text:
            continue
        if not _is_valid_token(text, conf):
            continue

        key = (block_num, par_num, line_num)
        lines_map.setdefault(key, []).append(text)
        confs.append(conf)

    if not confs:
        return "", 0.0, 0

    cleaned_lines: List[str] = []
    total_words = 0
    for words in lines_map.values():
        line_str = " ".join(words).strip()
        if not line_str:
            continue
        if _WATERMARK_RE.match(line_str) and len(words) <= 2:
            continue
        if len(words) == 1 and len(line_str) <= 2 and not line_str.isdigit():
            continue
        cleaned_lines.append(line_str)
        total_words += len(words)

    if not cleaned_lines:
        return "", 0.0, 0

    avg_conf = sum(confs) / len(confs)
    if avg_conf < 52 or (total_words < 2 and len(cleaned_lines[0]) < 5):
        return "", 0.0, 0

    final_text = "\n".join(cleaned_lines)
    score = avg_conf * (total_words**0.6)
    return final_text, score, total_words


def _ocr_single_image_sync(img_path: str, tmp_dir: str, lang_str: str) -> str:
    """Run OCR on an image, testing normal and dark-mode inverted variants when appropriate."""
    variants = _prepare_ocr_variants(img_path, tmp_dir)
    best_text = ""
    best_score = 0.0

    for var_path in variants:
        text, score, words = _run_tesseract_tsv_sync(var_path, lang_str)
        if score > best_score:
            best_score = score
            best_text = text
        if words >= 6 and score >= 180.0:
            break

    return best_text


def _normalize_for_comparison(text: str) -> str:
    """Normalize OCR text for cross-frame deduplication."""
    return re.sub(r"[^\wа-яё]+", " ", text.lower()).strip()


def _merge_or_deduplicate_frames(
    raw_frames: List[Tuple[float, str]]
) -> List[Dict[str, Any]]:
    """Deduplicate consecutive video frames that show the same or scrolling screenshots."""
    if not raw_frames:
        return []

    segments: List[Dict[str, Any]] = []
    seen_lines_norm: List[str] = []

    for idx, (ts, text) in enumerate(raw_frames):
        next_ts = raw_frames[idx + 1][0] if idx + 1 < len(raw_frames) else round(ts + 2.5, 2)
        norm_curr = _normalize_for_comparison(text)
        if not norm_curr:
            continue

        if segments:
            prev_seg = segments[-1]
            norm_prev = _normalize_for_comparison(prev_seg["text"])
            similarity = difflib.SequenceMatcher(None, norm_prev, norm_curr).ratio()

            if similarity >= 0.80 or norm_curr in norm_prev:
                prev_seg["end_seconds"] = round(next_ts, 2)
                prev_seg["end"] = _format_timestamp(next_ts)
                if len(norm_curr) > len(norm_prev) * 1.15 and norm_prev in norm_curr:
                    prev_seg["text"] = text
                continue

        new_lines: List[str] = []
        for line in text.splitlines():
            ln_norm = _normalize_for_comparison(line)
            if not ln_norm:
                continue
            is_dup = any(
                difflib.SequenceMatcher(None, ln_norm, prev_ln).ratio() >= 0.86
                for prev_ln in seen_lines_norm[-25:]
            )
            if not is_dup:
                new_lines.append(line)
                seen_lines_norm.append(ln_norm)

        if not new_lines:
            if segments:
                segments[-1]["end_seconds"] = round(next_ts, 2)
                segments[-1]["end"] = _format_timestamp(next_ts)
            continue

        clean_block = "\n".join(new_lines)
        segments.append(
            {
                "start": _format_timestamp(ts),
                "end": _format_timestamp(next_ts),
                "start_seconds": round(ts, 2),
                "end_seconds": round(next_ts, 2),
                "text": clean_block,
            }
        )

    return segments


def _extract_video_ocr_sync(
    video_path: str,
    language: Optional[str | List[str]] = None,
    max_frames: int = 18,
) -> Dict[str, Any]:
    """Extract scene/interval keyframes from a local video file and run OCR."""
    if not is_ocr_available() or not _FFMPEG_BIN or not os.path.exists(video_path):
        return {"ocr_available": False, "full_text": "", "segments": [], "frames_analyzed": 0}

    lang_str = _resolve_tesseract_langs(language)

    with tempfile.TemporaryDirectory(prefix="ssmcp_ocr_") as tmp_dir:
        out_pattern = os.path.join(tmp_dir, "frame_%03d.jpg")
        vf_expr = (
            "select='eq(n\\,0)+gt(scene\\,0.10)+isnan(prev_selected_t)+gte(t-prev_selected_t\\,2.0)',"
            "scale='min(1280\\,iw)':-2,"
            "showinfo"
        )
        cmd = [
            _FFMPEG_BIN,
            "-hide_banner",
            "-i",
            video_path,
            "-vf",
            vf_expr,
            "-vsync",
            "vfr",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "2",
            out_pattern,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception as e:
            logger.debug(f"ffmpeg frame extraction failed: {e}")
            return {"ocr_available": False, "full_text": "", "segments": [], "frames_analyzed": 0}

        timestamps = [float(m) for m in re.findall(r"pts_time:\s*([\d.]+)", proc.stderr or "")]
        frame_files = sorted(Path(tmp_dir).glob("frame_*.jpg"))
        if not frame_files:
            return {"ocr_available": False, "full_text": "", "segments": [], "frames_analyzed": 0}

        unique_frames: List[Tuple[float, str]] = []
        last_hash: Optional[int] = None

        for idx, fpath in enumerate(frame_files):
            ts = timestamps[idx] if idx < len(timestamps) else round(idx * 2.0, 2)
            h_res = _compute_center_dhash(str(fpath))
            if h_res is not None:
                curr_hash, _ = h_res
                if last_hash is not None and _hamming_distance(curr_hash, last_hash) <= 5:
                    continue
                last_hash = curr_hash
            unique_frames.append((ts, str(fpath)))

        ocr_raw: List[Tuple[float, str]] = []
        for ts, fpath in unique_frames:
            text = _ocr_single_image_sync(fpath, tmp_dir, lang_str)
            if text:
                ocr_raw.append((ts, text))

        segments = _merge_or_deduplicate_frames(ocr_raw)
        full_text = "\n\n".join(s["text"] for s in segments)
        return {
            "ocr_available": bool(segments),
            "full_text": full_text,
            "segments": segments,
            "frames_analyzed": len(unique_frames),
        }


async def extract_video_ocr(
    video_path: str,
    language: Optional[str | List[str]] = None,
    max_frames: int = 18,
) -> Dict[str, Any]:
    """Asynchronously extract on-screen text from a video file via keyframe OCR."""
    if not is_ocr_available():
        return {"ocr_available": False, "full_text": "", "segments": [], "frames_analyzed": 0}
    async with _OCR_SEM:
        return await asyncio.to_thread(_extract_video_ocr_sync, video_path, language, max_frames)


async def extract_images_ocr(
    image_urls: List[str],
    language: Optional[str | List[str]] = None,
    proxy: Optional[str] = None,
) -> Dict[str, Any]:
    """Download carousel/post images concurrently and extract text from each slide via OCR."""
    if not is_ocr_available() or not image_urls:
        return {"ocr_available": False, "full_text": "", "segments": [], "slides_analyzed": 0}

    lang_str = _resolve_tesseract_langs(language)
    urls = [u for u in image_urls if u][:25]

    client_kwargs: Dict[str, Any] = {
        "timeout": 20.0,
        "follow_redirects": True,
        "headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.tiktok.com/",
        },
    }
    if proxy and (proxy.startswith("http") or proxy.startswith("socks")):
        client_kwargs["proxy"] = proxy

    with tempfile.TemporaryDirectory(prefix="ssmcp_slides_") as tmp_dir:
        downloaded: List[Tuple[int, str]] = []

        async with httpx.AsyncClient(**client_kwargs) as client:
            async def _fetch_slide(idx: int, url: str) -> Optional[Tuple[int, str]]:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200 and resp.content:
                        dest = os.path.join(tmp_dir, f"slide_{idx:02d}.jpg")
                        with open(dest, "wb") as f:
                            f.write(resp.content)
                        return idx, dest
                except Exception as e:
                    logger.debug(f"Failed to download slide {idx} ({url}): {e}")
                return None

            results = await asyncio.gather(*(_fetch_slide(i, u) for i, u in enumerate(urls, start=1)))
            downloaded = [r for r in results if r is not None]

        if not downloaded:
            return {"ocr_available": False, "full_text": "", "segments": [], "slides_analyzed": 0}

        async with _OCR_SEM:
            def _run_slides() -> List[Dict[str, Any]]:
                segs: List[Dict[str, Any]] = []
                for idx, path in sorted(downloaded, key=lambda x: x[0]):
                    txt = _ocr_single_image_sync(path, tmp_dir, lang_str)
                    if txt:
                        segs.append(
                            {
                                "slide": idx,
                                "start": f"Slide {idx}",
                                "end": f"Slide {idx}",
                                "text": txt,
                            }
                        )
                return segs

            segments = await asyncio.to_thread(_run_slides)

        full_text = "\n\n".join(f"[Slide {s['slide']}]\n{s['text']}" for s in segments)
        return {
            "ocr_available": bool(segments),
            "full_text": full_text,
            "segments": segments,
            "slides_analyzed": len(downloaded),
        }
