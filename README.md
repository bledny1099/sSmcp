<p align="center">
  <img src="assets/logo.jpg" alt="sSmcp Logo" width="240" />
</p>

<h1 align="center">sSmcp (socialSimple MCP)</h1>

<p align="center">
  Unified Model Context Protocol server for YouTube, TikTok, Instagram, and Threads with Faster-Whisper audio transcription, Tesseract keyframe and slide OCR, and full media automation.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Protocol-MCP-8A2BE2?style=flat" alt="MCP" />
  <img src="https://img.shields.io/badge/Platforms-YouTube%20%7C%20TikTok%20%7C%20Instagram%20%7C%20Threads-FF0050?style=flat" alt="Platforms" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat" alt="License" />
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#available-tools">Tools</a> •
  <a href="#installation-and-setup">Setup</a> •
  <a href="#configuration-reference">Config</a> •
  <a href="#media-hosting-and-downloads">Media</a> •
  <a href="#security-and-authentication">Security</a> •
  <a href="#production-deployment">Deployment</a> •
  <a href="#client-configuration">Clients</a> •
  <a href="#troubleshooting">Troubleshooting</a>
</p>

---

## Overview

sSmcp connects your AI assistant (Claude Desktop, Cursor, Antigravity IDE, Windsurf, Claude Code, etc.) directly to **YouTube**, **TikTok**, **Instagram**, and **Threads** through a single MCP server. It pulls video metadata, author statistics, comments, spoken audio transcripts, and on-screen text from screenshots and photo carousels into your workspace.

TikTok, Instagram, Threads, and YouTube transcripts work immediately with **zero API keys required**.

### Key capabilities

- **4 platforms in 1 server (20 tools)**: Full coverage of YouTube, TikTok, Instagram, and Threads without running separate background services.
- **Multi-tier extraction with fallback**: Extracts video metadata, view counts, and direct streams via `yt-dlp`, with automatic fallback to platform-specific APIs (`tikwm` and `embed/v2` for TikTok, HTML parsers for Instagram and Threads, `youtube-transcript.ai` and `LemnosLife` for YouTube) when anti-bot challenges trigger.
- **Local speech-to-text**: Fast speech transcription powered by `faster-whisper` (`int8` on CPU by default). Runs locally on your machine or VPS without external cloud speech billing.
- **Visual OCR for screenshots and slides**: Automatically extracts keyframes from videos (`ffmpeg` scene-change detection and interval sampling), deduplicates identical frames with a 256-bit center-crop `dHash`, preprocesses dark-mode and light-mode text with `Pillow`, and runs `tesseract` (`rus+eng` by default) to read slides, chat logs, code snippets, and photo carousels.
- **Whisper hallucination filtering**: Strips repeating subtitle loops caused by background music on videos without human speech.
- **Outbound proxy and cookie support**: Native HTTP and SOCKS5 proxy routing (`SSMCP_PROXY`) and `cookies.txt` handling to bypass geo-restrictions and rate limits.
- **Custom domain publishing**: Download media and host watermark-free `.mp4` video and `.jpg` cover assets directly on your own server or CDN.
- **Flexible transports and authentication**: Supports local `stdio` mode for desktop clients, as well as `http` and `sse` modes with static Bearer tokens or built-in OAuth 2.1 (Google Sign-In and password login).

---

## Available Tools

The server registers 20 tools across all four supported platforms.

### TikTok Tools (No API Key Required)

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `tiktok_get_video` | `url` (string), `fast` (bool, default: False) | Returns post title, description, hashtags, links, author stats, view/like/share counts, cover image, and carousel items. `fast=True` uses the public oEmbed endpoint for sub-second responses without extended stats. |
| `tiktok_batch_videos` | `urls` (list[str]), `fast` (bool, default: True) | Fetches metadata for up to 20 TikTok links concurrently with a single call. |
| `tiktok_get_transcript` | `url` (string), `languages` (list[str], optional), `with_timestamps` (bool, default: False), `model_size` (string, optional), `include_ocr` (bool, default: True) | Returns native TikTok captions when available, or runs local Faster-Whisper and keyframe/slide OCR. Returns combined text, separate audio text, and on-screen OCR text. |
| `tiktok_get_user_videos` | `username` (string), `limit` (int, default: 20) | Lists recent videos from a public `@username` profile with stats and links. |
| `tiktok_publish_video` | `url` (string), `slug` (string, optional), `include_transcript` (bool, default: True), `languages` (list[str], optional), `max_height` (int, default: 1080) | Downloads video (`.mp4`) and cover (`.jpg`), transcribes speech, reads on-screen text, and publishes the media files under your own domain or CDN. |

### Instagram Tools (No API Key Required)

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `instagram_get_media` | `url` (string) | Fetches caption, author, like and comment counts, duration, and direct media URLs for Reels, posts, and multi-slide carousels. |
| `instagram_get_transcript` | `url` (string), `languages` (list[str], optional), `with_timestamps` (bool, default: False), `model_size` (string, optional), `include_ocr` (bool, default: True) | Transcribes spoken audio in Reels with Faster-Whisper and reads on-screen text from video keyframes or carousel slides with Tesseract OCR. |
| `instagram_get_user_posts` | `username` (string), `limit` (int, default: 12) | Fetches recent public posts and Reels from an Instagram `@username`. |

### Threads Tools (No API Key Required)

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `threads_get_post` | `url` (string), `include_replies` (bool, default: True), `include_transcript` (bool, default: False) | Fetches a Threads post, engagement metrics, top replies, and optionally transcribes and runs OCR on attached videos or images. |
| `threads_search_posts` | `query` (string), `limit` (int, default: 20) | Searches public Threads posts by keyword. |
| `threads_get_user_posts` | `username` (string), `limit` (int, default: 15) | Lists recent posts and replies from a public Threads `@username`. |

### YouTube Tools

| Tool | Parameters | Description | Needs API Key? |
| :--- | :--- | :--- | :---: |
| `youtube_get_transcript` | `url` (string), `languages` (list[str], optional), `with_timestamps` (bool, default: False), `model_size` (string, optional), `include_ocr` (bool, default: True) | 4-tier transcript extraction (`youtube-transcript.ai` -> `yt-dlp` subtitles -> `LemnosLife` -> local Whisper and keyframe OCR). | **No** |
| `youtube_list_transcript_languages` | `url` (string) | Lists available manual and auto-generated subtitle tracks for a video. | **No** |
| `youtube_quota_status` | *(none)* | Returns daily YouTube Data API v3 quota usage and remaining units. | **No** |
| `youtube_search` | `query` (string), `order` (string, default: relevance), `published_after` (string, optional), `max_results` (int, default: 10) | Searches YouTube for videos, channels, or playlists. | Yes (`YOUTUBE_API_KEY`) |
| `youtube_get_videos` | `video_ids` (list[str]) | Batch lookups for up to 50 video IDs with views, likes, comments, and duration. | Yes (`YOUTUBE_API_KEY`) |
| `youtube_list_playlists` | `channel_id` (string, optional), `mine` (bool, default: False), `max_results` (int, default: 25) | Lists playlists for a channel or the authenticated account. | Yes (API key or OAuth) |
| `youtube_list_playlist_items` | `playlist_id` (string), `max_results` (int, default: 50) | Lists videos inside a playlist with playlist item IDs for management. | Yes (API key or OAuth) |
| `youtube_add_to_playlist` | `playlist_id` (string), `video_id` (string), `position` (int, optional), `note` (string, optional) | Adds a video to a playlist. | Yes (OAuth) |
| `youtube_remove_from_playlist` | `playlist_item_id` (string) | Removes an entry from a playlist by item ID. | Yes (OAuth) |

### Language handling

The `languages` parameter controls Whisper speech recognition and Tesseract OCR:

| Value | Behavior |
| :--- | :--- |
| `["ru"]` | Locks Whisper to Russian and prioritizes Russian + English in Tesseract OCR. |
| `["ru", "en"]` | Runs Whisper in multilingual mode covering code-switched speech and bilingual `rus+eng` OCR. |
| `None` | Whisper auto-detects spoken language; OCR defaults to `rus+eng`. |

---

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- `ffmpeg` (required for audio extraction and keyframe sampling)
- `tesseract-ocr` with Russian and English language packs (required for visual OCR)

```bash
# macOS
brew install ffmpeg tesseract tesseract-lang

# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng
```

### 1. Clone repository

```bash
git clone https://github.com/bledny1099/sSmcp.git
cd sSmcp
```

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

### 3. Configure environment (optional)

```bash
cp .env.example .env
```

TikTok, Instagram, Threads, and YouTube transcripts work without any API keys. If you want remote access, YouTube search, or custom publishing, edit `.env`:

```bash
SSMCP_TRANSPORT=http
SSMCP_PORT=8890
SSMCP_BEARER_TOKEN=your_secure_random_token_here
SSMCP_WHISPER_MODEL=small
SSMCP_OCR_LANGUAGES=rus+eng
```

### 4. Pre-download the Whisper model (recommended)

Pre-downloading prevents the first transcription call from waiting on model weights:

```bash
.venv/bin/python scripts/download_model.py small
```

### 5. Run server

**Option A: Local stdio (for IDEs and Desktop clients)**

```bash
.venv/bin/ssmcp
```

**Option B: HTTP or SSE daemon (for remote server deployments)**

```bash
export SSMCP_TRANSPORT=http
.venv/bin/ssmcp
```

The HTTP transport binds to `127.0.0.1:8890` by default. Put it behind a reverse proxy (Caddy or Nginx) when deploying remotely.

---

## Configuration Reference

All settings are configured through environment variables. Every variable has a sensible default.

### Transport

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SSMCP_TRANSPORT` | `stdio` | `stdio` for local AI clients, `http` (streamable HTTP) or `sse` for remote daemons. |
| `SSMCP_HOST` | `127.0.0.1` | Bind address in HTTP/SSE mode. Keep on loopback behind a reverse proxy. |
| `SSMCP_PORT` | `8890` | Bind port in HTTP/SSE mode. |

### Authentication (Remote HTTP/SSE Mode)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SSMCP_BEARER_TOKEN` | *(empty)* | Static token for `Authorization: Bearer <token>` or `?token=<token>`. |
| `SSMCP_PUBLIC_URL` | *(auto)* | Public HTTPS base URL (e.g. `https://mcp.yourdomain.com`) for OAuth 2.1 metadata discovery. |
| `SSMCP_AUTH_PASSWORD` | *(empty)* | Password for the built-in OAuth 2.1 login screen. |
| `SSMCP_GOOGLE_CLIENT_ID` | *(empty)* | Google OAuth Client ID for OAuth 2.1 login. |
| `SSMCP_GOOGLE_CLIENT_SECRET` | *(empty)* | Google OAuth Client Secret for OAuth 2.1 login. |
| `SSMCP_ALLOWED_EMAILS` | *(empty)* | Comma-separated email whitelist for Google Sign-In. |

### Network and Scraping

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SSMCP_PROXY` | *(empty)* | HTTP or SOCKS5 proxy URL (e.g. `socks5://user:pass@host:port` or `http://host:port`). |
| `SSMCP_COOKIES_FILE` | *(empty)* | Path to a Netscape-format `cookies.txt` file for restricted content. |

### Transcription and Visual OCR

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SSMCP_WHISPER_MODEL` | `small` | Faster-Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3`. |
| `SSMCP_WHISPER_DEVICE` | `cpu` | Inference device: `cpu` or `cuda`. |
| `SSMCP_WHISPER_COMPUTE_TYPE` | `int8` | Precision: `int8` for CPU, `float16` for GPU. |
| `SSMCP_WHISPER_CACHE_DIR` | `~/.cache/ssmcp/whisper` | Directory for caching Whisper model weights. |
| `SSMCP_WHISPER_MAX_DURATION` | `1800` | Maximum video length in seconds accepted for Whisper fallback. |
| `SSMCP_OCR_ENABLED` | `true` | Enable Tesseract OCR for video keyframes and image carousels. |
| `SSMCP_OCR_LANGUAGES` | `rus+eng` | Tesseract language pack string (e.g. `rus+eng`, `eng`). |
| `SSMCP_OCR_MAX_FRAMES` | `16` | Maximum keyframes sampled per video. |

### Media Publishing (`tiktok_publish_video`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SSMCP_UPLOAD_MODE` | `local` | `local` (copies to a local web directory) or `scp` (uploads via SSH). |
| `SSMCP_UPLOAD_LOCAL_DIR` | none | Local folder served by Nginx or Caddy. |
| `SSMCP_UPLOAD_PUBLIC_URL_PREFIX` | none | Public URL prefix returned for uploaded assets. |
| `SSMCP_UPLOAD_SSH_HOST` | none | Remote host for SCP upload mode. |
| `SSMCP_UPLOAD_SSH_PORT` | `22` | SSH port for SCP mode. |
| `SSMCP_UPLOAD_SSH_USER` | `root` | SSH user for SCP mode. |
| `SSMCP_UPLOAD_SSH_KEY` | none | Path to private SSH key for SCP mode. |
| `SSMCP_UPLOAD_REMOTE_DIR` | none | Destination folder on remote server in SCP mode. |

### YouTube Data API v3 (Optional)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `YOUTUBE_API_KEY` | *(empty)* | API key for `youtube_search` and `youtube_get_videos`. |
| `YOUTUBE_CLIENT_ID` | *(empty)* | Google OAuth Client ID for playlist management tools. |
| `YOUTUBE_CLIENT_SECRET` | *(empty)* | Google OAuth Client Secret for playlist management tools. |
| `YOUTUBE_REFRESH_TOKEN` | *(empty)* | OAuth refresh token (generate via `scripts/get_refresh_token.py`). |
| `YT_DAILY_QUOTA_LIMIT` | `10000` | Daily API quota cap before requests are paused. |
| `YT_QUOTA_WARN_THRESHOLD` | `8000` | Quota warning threshold. |
| `YT_QUOTA_STATE_PATH` | `~/.cache/ssmcp/quota_state.json` | Path where daily quota ledger is persisted. |

### Resource requirements

| Whisper Model | Approx. RAM | Speed on CPU | Best Use Case |
| :--- | :--- | :--- | :--- |
| `small` (default) | ~1.5 GB | Fast (~2-3x real time) | General production, short videos, Reels, TikToks |
| `base` | ~0.7 GB | Very fast | Low-memory VPS instances (1-2 GB RAM) |
| `large-v3` | ~4.0 GB | Roughly real time | Complex multi-speaker speech with noisy backgrounds |

---

## Media Hosting and Downloads

When calling `tiktok_publish_video`, the server downloads the watermark-free video and cover thumbnail and makes them accessible under your own domain.

### Option 1: Local Directory Mode (Self-Hosted on Same Server)

Point a directory served by Caddy or Nginx:

```bash
SSMCP_UPLOAD_MODE=local
SSMCP_UPLOAD_LOCAL_DIR=/var/www/uploads
SSMCP_UPLOAD_PUBLIC_URL_PREFIX=https://cdn.yourdomain.com/uploads
```

Ensure the user running `ssmcp` has write permissions:

```bash
sudo chown ssmcp:www-data /var/www/uploads
sudo chmod 0755 /var/www/uploads
```

### Option 2: Remote SCP Mode (Upload to a CDN or Storage Host)

If `sSmcp` runs on an application server and static assets live on another machine:

```bash
SSMCP_UPLOAD_MODE=scp
SSMCP_UPLOAD_SSH_HOST=cdn.yourdomain.com
SSMCP_UPLOAD_SSH_PORT=22
SSMCP_UPLOAD_SSH_USER=deploy
SSMCP_UPLOAD_SSH_KEY=/home/ssmcp/.ssh/id_ed25519
SSMCP_UPLOAD_REMOTE_DIR=/var/www/uploads
SSMCP_UPLOAD_PUBLIC_URL_PREFIX=https://cdn.yourdomain.com/uploads
```

---

## Security and Authentication

When running in HTTP or SSE mode on a remote server, protect the instance from unauthorized access. An open instance allows arbitrary video downloads, OCR processing, and Whisper inference on your CPU.

### Rules

1. Keep `SSMCP_HOST=127.0.0.1`. Never expose port `8890` directly to the public internet.
2. In Docker, always bind ports to localhost: `-p 127.0.0.1:8890:8890`.
3. Set `SSMCP_BEARER_TOKEN` with a strong token (`openssl rand -hex 24`).
4. Terminate TLS and check the token at your reverse proxy (Caddy or Nginx).
5. Keep `.env` and `cookies.txt` out of version control.

### Nginx Configuration

```nginx
map $http_authorization $mcp_ok {
    default                         0;
    "Bearer YOUR_SSMCP_BEARER_TOKEN" 1;
}

server {
    listen 443 ssl http2;
    server_name mcp.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/mcp.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mcp.yourdomain.com/privkey.pem;

    client_max_body_size 200m;

    location / {
        if ($mcp_ok = 0) { return 401; }

        proxy_pass http://127.0.0.1:8890;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streaming responses must not be buffered
        proxy_buffering off;
        proxy_cache off;

        # Transcription of long videos can take several minutes
        proxy_read_timeout 900s;
        proxy_send_timeout 900s;
    }
}
```

Add the `$connection_upgrade` map in the `http {}` block:

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}
```

### Caddy Configuration

```caddyfile
mcp.yourdomain.com {
    reverse_proxy 127.0.0.1:8890 {
        flush_interval -1
        header_up X-Forwarded-Proto {scheme}
        header_up X-Forwarded-Host {host}
        transport http {
            read_timeout 900s
        }
    }
}
```

### Firewall

```bash
sudo ufw allow 80,443/tcp
sudo ufw deny 8890
```

---

## Production Deployment

### Option A: Docker Compose (Recommended)

The provided [Dockerfile](Dockerfile) bundles Python 3.12, `ffmpeg`, and `tesseract-ocr` with Russian and English dictionaries.

```bash
git clone https://github.com/bledny1099/sSmcp.git /opt/sSmcp
cd /opt/sSmcp

cp .env.example .env
# Set SSMCP_BEARER_TOKEN=$(openssl rand -hex 24) in .env
```

Start the container:

```bash
docker compose up -d --build
```

The named Docker volume `ssmcp_data` persists downloaded Whisper weights and quota state across container rebuilds.

### Option B: Bare-Metal `systemd` Service

Create `/etc/systemd/system/ssmcp.service`:

```ini
[Unit]
Description=sSmcp (socialSimple MCP) Server
After=network-online.target
Wants=network-online.target

[Service]
User=ssmcp
WorkingDirectory=/opt/sSmcp
EnvironmentFile=/opt/sSmcp/.env
ExecStart=/opt/sSmcp/.venv/bin/ssmcp
Restart=always
RestartSec=5
StartLimitIntervalSec=0

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/data /var/www/uploads

MemoryMax=4G
OOMPolicy=continue

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ssmcp.service
sudo systemctl status ssmcp.service
```

### Keeping yt-dlp current

Social platforms change page structures frequently. Schedule a weekly update on your server:

```bash
/opt/sSmcp/.venv/bin/pip install -U yt-dlp && systemctl restart ssmcp.service
```

### Verifying the deployment

```bash
# 1. Healthcheck endpoint
curl -s http://127.0.0.1:8890/health

# 2. Verify token protection over HTTPS
curl -s -o /dev/null -w '%{http_code}\n' https://mcp.yourdomain.com/mcp

# 3. Test MCP initialize handshake
curl -sN https://mcp.yourdomain.com/mcp \
  -H 'Authorization: Bearer YOUR_SSMCP_BEARER_TOKEN' \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"1"}}}'
```

---

## Client Configuration

### 1. Cursor, Antigravity IDE, Windsurf

**Option 1: Local stdio (`mcp_config.json` or `.cursor/mcp.json`)**

```json
{
  "mcpServers": {
    "sSmcp": {
      "command": "/ABSOLUTE/PATH/TO/sSmcp/.venv/bin/ssmcp",
      "env": {
        "SSMCP_TRANSPORT": "stdio",
        "SSMCP_WHISPER_MODEL": "small",
        "SSMCP_OCR_ENABLED": "true",
        "SSMCP_OCR_LANGUAGES": "rus+eng"
      }
    }
  }
}
```

**Option 2: Remote Streamable HTTP**

```json
{
  "mcpServers": {
    "sSmcp": {
      "url": "https://mcp.yourdomain.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_SSMCP_BEARER_TOKEN"
      }
    }
  }
}
```

---

### 2. Claude Desktop

Configure `claude_desktop_config.json`:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sSmcp": {
      "command": "/ABSOLUTE/PATH/TO/sSmcp/.venv/bin/ssmcp",
      "env": {
        "SSMCP_TRANSPORT": "stdio",
        "SSMCP_WHISPER_MODEL": "small",
        "SSMCP_OCR_LANGUAGES": "rus+eng"
      }
    }
  }
}
```

---

### 3. Claude Mobile (iOS / Android) & Claude Web

1. Open Claude and go to **Settings** -> **Connectors**.
2. Click **Add custom connector**.
3. Fill in:
   - **Name:** `sSmcp`
   - **URL:** `https://mcp.yourdomain.com/mcp`
   - **Authentication:** If using Bearer token, select **No sign-in** and click **+ Add header**:
     - Key: `Authorization`
     - Value: `Bearer <YOUR_SSMCP_BEARER_TOKEN>`
   - Alternatively, if using OAuth 2.1, select OAuth and log in with your configured password or Google account.
4. Click **Add connector**. All 20 tools become available in your chats.

---

### 4. Claude Code CLI

```bash
claude mcp add sSmcp -- /ABSOLUTE/PATH/TO/sSmcp/.venv/bin/ssmcp
```

---

## Usage Examples

```text
User: "Summarize this TikTok and give me the text shown on screen: https://vt.tiktok.com/ZSqa7P1oy/"
```

The assistant calls `tiktok_get_transcript(url="...", include_ocr=True)`. sSmcp transcribes the spoken speech with Faster-Whisper, extracts keyframes, and runs Tesseract OCR on any text slides or code snippets.

```text
User: "Find the top 3 most popular recent YouTube videos on 'quantum computing' and get the full transcript of the first one."
```

The assistant calls `youtube_search(query="quantum computing", order="viewCount", max_results=3)`, followed by `youtube_get_transcript(url="...")` to retrieve the video transcript with timestamps.

```text
User: "Read this Instagram carousel and tell me what is written across all slides: https://www.instagram.com/p/DB123456789/"
```

The assistant calls `instagram_get_transcript(url="...")`. sSmcp downloads the carousel slides and runs OCR across every frame.

---

## Troubleshooting

| Symptom | Cause and fix |
| :--- | :--- |
| `Unable to extract webpage video data` | Platform anti-bot challenge triggered. Ensure `yt-dlp` is up to date, or supply a `cookies.txt` file via `SSMCP_COOKIES_FILE`. |
| `ffmpeg not found` or `tesseract not found` | Install `ffmpeg` and `tesseract-ocr` (`tesseract-ocr-rus`, `tesseract-ocr-eng`) through your operating system package manager. |
| Process killed during transcription | System ran out of memory. Switch to a smaller model (`SSMCP_WHISPER_MODEL=base` or `small`) or limit video duration. |
| Empty audio transcript on a video with music | The video has no human speech. If the video contains text slides, check the `ocr_transcript` output. |
| Quota exceeded error on YouTube search | YouTube Data API v3 daily quota reached. Check usage with `youtube_quota_status`. Note that `youtube_get_transcript` never consumes quota units. |
| `publish_video raises Permission denied` | The service account cannot write to `SSMCP_UPLOAD_LOCAL_DIR`. Check directory permissions (`chmod 0755`). |
| Client disconnects during long transcriptions | Increase `proxy_read_timeout` to `900s` in your Caddy or Nginx reverse proxy configuration. |

---

## License

[MIT](LICENSE)
