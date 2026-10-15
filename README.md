# sSmcp (socialSimple MCP)

**socialSimple MCP (`sSmcp`)** is a single, self-contained Model Context Protocol (MCP) server for **YouTube**, **TikTok**, **Instagram**, and **Threads**.

It pulls video metadata, post captions, comments, user feeds, search results, spoken transcripts, and on-screen text from video slides or screenshot posts.

TikTok, Instagram, Threads, and YouTube transcripts work out of the box with **zero API keys required**.

---

## Key Features

- **4 Platforms in 1 Server (20 Tools):**
  - **YouTube (9 tools):** Search, video inspection, playlists, 4-tier transcript extraction (`youtube-transcript.ai` -> `yt-dlp` captions -> `LemnosLife` -> local Whisper + OCR), and daily API quota tracking.
  - **TikTok (5 tools):** Single and parallel batch video inspection (`yt-dlp` -> `tikwm` API -> `embed/v2` -> `oEmbed`), full audio + slide OCR transcripts, creator profile feeds, and downloading/publishing watermark-free `.mp4` videos and `.jpg` covers to your own domain.
  - **Instagram (3 tools):** Reels, posts, and multi-slide carousels without login (`yt-dlp` + `embed/captioned` HTML parser), spoken audio transcription + slide OCR, and public profile grids.
  - **Threads (3 tools):** Single posts with top replies, keyword search, and author feeds without an Instagram/Meta API key, including automatic transcription and OCR on attached videos or images.
- **Screenshot & Slide Video OCR (`ssmcp/ocr.py`):**
  - Many TikToks, Reels, and Shorts have no spoken voiceover (just background music over screenshots, chat logs, code snippets, or text slides).
  - `sSmcp` samples keyframes via `ffmpeg` (scene-change detection + interval fallback), deduplicates static frames using a 256-bit center-crop `dHash`, preprocesses both light and dark-mode text (`Pillow` autocontrast + inversion), and runs `tesseract` OCR (`rus+eng` by default) in parallel with Faster-Whisper.
- **Zero-Noise Whisper Transcription (`ssmcp/whisper_transcriber.py`):**
  - Uses `faster-whisper` (`int8` on CPU by default) with Silero VAD and built-in hallucination filtering so background music beats never turn into fake subtitle loops.
- **Runs Anywhere:**
  - **Local (`stdio`):** Plug directly into Claude Desktop, Cursor, Antigravity IDE, Windsurf, or Claude Code.
  - **Remote Server (`http` / `sse`):** Deploy with Docker Compose or `systemd` behind Caddy or Nginx, secured with a static Bearer token or built-in OAuth 2.1 (Google Sign-In + password login).

---

## All 20 Available Tools

### TikTok (No API Key Required)
| Tool | Description |
| :--- | :--- |
| `tiktok_get_video` | Get views, likes, comments, shares, saves, duration, author stats, music info, and description. Set `fast=true` for sub-second oEmbed metadata. |
| `tiktok_batch_videos` | Fetch metadata for up to 20 TikTok URLs concurrently with a single call. |
| `tiktok_get_transcript` | Extract native TikTok captions or run Faster-Whisper + keyframe/slide OCR (`whisper_model="small"`, custom `languages`). |
| `tiktok_get_user_videos` | List recent videos from a public `@username` profile. |
| `tiktok_publish_video` | Download a TikTok video (`.mp4`) and cover (`.jpg`) and publish them to your own web directory or remote server via SCP. |

### Instagram (No API Key Required)
| Tool | Description |
| :--- | :--- |
| `instagram_get_media` | Fetch metadata, caption, counts, and media URLs for Reels, posts, and multi-slide carousels. |
| `instagram_get_transcript` | Transcribe spoken audio in Reels via Faster-Whisper and read on-screen text from video keyframes or carousel slides via Tesseract OCR. |
| `instagram_get_user_posts` | Fetch recent public posts/Reels from an Instagram `@username`. |

### Threads (No API Key Required)
| Tool | Description |
| :--- | :--- |
| `threads_get_post` | Fetch a Threads post, engagement metrics, top replies, and optionally transcribe/OCR attached video or image media (`include_transcript=True`). |
| `threads_search_posts` | Search public Threads posts by keyword. |
| `threads_get_user_posts` | Fetch recent posts from a public Threads `@username`. |

### YouTube
| Tool | Description | API Key Needed? |
| :--- | :--- | :---: |
| `youtube_get_transcript` | 4-tier transcript extraction (captions -> Whisper + keyframe OCR fallback) with timestamped segments. | **No** |
| `youtube_list_transcript_languages` | List available subtitle tracks (manual vs auto-generated) for a YouTube video. | **No** |
| `youtube_search` | Search YouTube for videos, channels, or playlists with filters (`order`, `duration`, `published_after`). | Yes (`YOUTUBE_API_KEY`) |
| `youtube_get_videos` | Batch lookup up to 50 video IDs with views, likes, duration, and tags. | Yes (`YOUTUBE_API_KEY`) |
| `youtube_list_playlists` | List playlists for a channel ID or the authenticated user (`mine=True`). | Yes (API key or OAuth) |
| `youtube_list_playlist_items` | List videos inside any playlist with `playlist_item_id` for removal. | Yes (API key or OAuth) |
| `youtube_add_to_playlist` | Add a video to a playlist. | Yes (OAuth) |
| `youtube_remove_from_playlist` | Remove an item from a playlist by `playlist_item_id`. | Yes (OAuth) |
| `youtube_quota_status` | Check daily YouTube Data API v3 quota usage and remaining budget. | No |

---

## Part 1: Local Installation (macOS, Linux, Windows WSL)

Running `sSmcp` locally over `stdio` is the fastest way to use it inside desktop AI clients.

### 1. Install System Dependencies (`ffmpeg` and `tesseract`)

`ffmpeg` is required to extract audio and video frames. `tesseract` is required to read text from screenshot/slide videos and photo carousels.

**macOS (Homebrew):**
```bash
brew install ffmpeg tesseract tesseract-lang
```

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus
```

### 2. Clone and Install `sSmcp`

```bash
git clone https://github.com/bledny1099/sSmcp.git
cd sSmcp

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

*(Optional)* Pre-download the default Faster-Whisper `small` model (~460 MB) so your first transcription starts immediately:
```bash
.venv/bin/python scripts/download_model.py small
```

### 3. Connect Your Local MCP Client

Replace `/ABSOLUTE/PATH/TO/sSmcp` below with the actual path where you cloned the repository.

#### Claude Desktop (`claude_desktop_config.json`)
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sSmcp": {
      "command": "/ABSOLUTE/PATH/TO/sSmcp/.venv/bin/ssmcp",
      "env": {
        "SSMCP_TRANSPORT": "stdio",
        "SSMCP_WHISPER_MODEL": "small",
        "SSMCP_OCR_ENABLED": "true",
        "SSMCP_OCR_LANGUAGES": "rus+eng",
        "YOUTUBE_API_KEY": ""
      }
    }
  }
}
```

#### Cursor / Antigravity IDE / Windsurf (`mcp_config.json` or `.cursor/mcp.json`)
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

#### Claude Code CLI
```bash
claude mcp add sSmcp -- /ABSOLUTE/PATH/TO/sSmcp/.venv/bin/ssmcp
```

---

## Part 2: Remote Server Deployment

Deploying `sSmcp` on a Linux VPS lets you connect from multiple devices, web clients, or remote agents over HTTPS.

### Option A: Docker Compose (Recommended)

`Dockerfile` bundles Python 3.12, `ffmpeg`, `tesseract-ocr` (`eng` + `rus`), and all Python dependencies in one container.

```bash
git clone https://github.com/bledny1099/sSmcp.git /opt/sSmcp
cd /opt/sSmcp

cp .env.example .env
```

Generate a random secret token and set it in `.env`:
```bash
TOKEN=$(openssl rand -hex 24)
sed -i "s|^SSMCP_BEARER_TOKEN=.*|SSMCP_BEARER_TOKEN=${TOKEN}|" .env
echo "Your Bearer token: ${TOKEN}"
```

Build and start the container:
```bash
docker compose up -d --build
```

Verify the health endpoint:
```bash
curl http://127.0.0.1:8890/health
```
Expected response:
```json
{
  "status": "ok",
  "service": "sSmcp",
  "full_name": "socialSimple MCP",
  "version": "1.0.0",
  "transport": "http"
}
```

### Option B: Bare-Metal `systemd` Service (Ubuntu / Debian)

If you prefer running directly on the host without Docker:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip ffmpeg tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus

sudo git clone https://github.com/bledny1099/sSmcp.git /opt/sSmcp
cd /opt/sSmcp

sudo python3 -m venv .venv
sudo .venv/bin/pip install --upgrade pip
sudo .venv/bin/pip install .
sudo .venv/bin/python scripts/download_model.py small

sudo cp .env.example .env
sudo nano .env   # Set SSMCP_BEARER_TOKEN and any optional variables
```

Create `/etc/systemd/system/ssmcp.service`:
```ini
[Unit]
Description=sSmcp (socialSimple MCP) Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/sSmcp
EnvironmentFile=/opt/sSmcp/.env
ExecStart=/opt/sSmcp/.venv/bin/ssmcp
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ssmcp.service
sudo systemctl status ssmcp.service
```

---

## Reverse Proxy (HTTPS) & Connecting Remote Clients

Bind `sSmcp` to `127.0.0.1:8890` and expose it over HTTPS with Caddy or Nginx.

### Caddy (`Caddyfile`)
```caddyfile
mcp.example.com {
    reverse_proxy 127.0.0.1:8890 {
        flush_interval -1
        header_up X-Forwarded-Proto {scheme}
        header_up X-Forwarded-Host {host}
    }
}
```

### Nginx
```nginx
server {
    listen 443 ssl http2;
    server_name mcp.example.com;

    location / {
        proxy_pass http://127.0.0.1:8890;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;
    }
}
```

### Connecting to Your Remote Server

1. **With Static Bearer Header (Cursor, Antigravity, Windsurf, custom agents):**
   ```json
   {
     "mcpServers": {
       "sSmcp": {
         "url": "https://mcp.example.com/mcp",
         "headers": {
           "Authorization": "Bearer YOUR_SSMCP_BEARER_TOKEN"
         }
       }
     }
   }
   ```
2. **With Query Parameter (Clients that only accept a URL):**
   ```text
   https://mcp.example.com/mcp?token=YOUR_SSMCP_BEARER_TOKEN
   ```
3. **With OAuth 2.1 Login Flow (Claude.ai Web / Desktop Custom Connectors):**
   Set `SSMCP_PUBLIC_URL=https://mcp.example.com` and `SSMCP_AUTH_PASSWORD=your_password` (or `SSMCP_GOOGLE_CLIENT_ID` / `SSMCP_GOOGLE_CLIENT_SECRET` / `SSMCP_ALLOWED_EMAILS`) in `.env`, then enter `https://mcp.example.com/mcp` in Claude.

---

## Optional Features Configuration

### 1. YouTube Search & Playlists (`YOUTUBE_API_KEY` / OAuth)
`youtube_get_transcript` and `youtube_list_transcript_languages` work **without** any keys.
If you also want `youtube_search`, `youtube_get_videos`, or playlist tools:
1. Create a project in [Google Cloud Console](https://console.cloud.google.com/), enable **YouTube Data API v3**, and put the key in `YOUTUBE_API_KEY`.
2. For personal playlist modification (`youtube_add_to_playlist`, `youtube_remove_from_playlist`), create an OAuth 2.0 Desktop Client ID and run:
   ```bash
   .venv/bin/python scripts/get_refresh_token.py
   ```

### 2. Publishing TikTok Videos to Your Own Domain (`tiktok_publish_video`)
The `tiktok_publish_video` tool downloads a TikTok video (`.mp4`) and its thumbnail (`.jpg`) and hosts them under a clean public URL on your server:
- **Local Directory Mode (`SSMCP_UPLOAD_MODE=local`):**
  ```env
  SSMCP_UPLOAD_MODE=local
  SSMCP_UPLOAD_LOCAL_DIR=/var/www/uploads
  SSMCP_UPLOAD_PUBLIC_URL_PREFIX=https://cdn.example.com/uploads
  ```
- **Remote SCP Mode (`SSMCP_UPLOAD_MODE=scp`):**
  ```env
  SSMCP_UPLOAD_MODE=scp
  SSMCP_UPLOAD_SSH_HOST=cdn.example.com
  SSMCP_UPLOAD_SSH_PORT=22
  SSMCP_UPLOAD_SSH_USER=deploy
  SSMCP_UPLOAD_SSH_KEY=/home/user/.ssh/id_ed25519
  SSMCP_UPLOAD_REMOTE_DIR=/var/www/uploads
  SSMCP_UPLOAD_PUBLIC_URL_PREFIX=https://cdn.example.com/uploads
  ```

---

## Environment Variables Reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SSMCP_TRANSPORT` | `stdio` | Transport protocol: `stdio`, `http` (`streamable-http`), or `sse`. |
| `SSMCP_HOST` | `127.0.0.1` | Bind host in `http` / `sse` mode. |
| `SSMCP_PORT` | `8890` | Bind port in `http` / `sse` mode. |
| `SSMCP_BEARER_TOKEN` | *(empty)* | Static token for `Authorization: Bearer <token>` or `?token=<token>`. |
| `SSMCP_PUBLIC_URL` | *(auto)* | Public HTTPS base URL for OAuth 2.1 metadata discovery. |
| `SSMCP_AUTH_PASSWORD` | *(empty)* | Password for the built-in OAuth 2.1 login screen. |
| `SSMCP_GOOGLE_CLIENT_ID` | *(empty)* | Google OAuth Client ID for OAuth 2.1 Sign-In. |
| `SSMCP_GOOGLE_CLIENT_SECRET` | *(empty)* | Google OAuth Client Secret for OAuth 2.1 Sign-In. |
| `SSMCP_ALLOWED_EMAILS` | *(empty)* | Comma-separated email whitelist for Google Sign-In. |
| `SSMCP_WHISPER_MODEL` | `small` | Default Faster-Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3`). |
| `SSMCP_WHISPER_DEVICE` | `cpu` | Inference device (`cpu` or `cuda`). |
| `SSMCP_WHISPER_COMPUTE_TYPE` | `int8` | Compute precision (`int8`, `float16`, `float32`). |
| `SSMCP_WHISPER_CACHE_DIR` | `~/.cache/ssmcp/whisper` | Directory where Whisper model weights are cached. |
| `SSMCP_WHISPER_MAX_DURATION` | `1800` | Maximum video duration (seconds) allowed for Whisper fallback. |
| `SSMCP_OCR_ENABLED` | `true` | Enable Tesseract OCR on video keyframes and image carousels. |
| `SSMCP_OCR_LANGUAGES` | `rus+eng` | Tesseract language pack string (e.g., `rus+eng`, `eng`). |
| `SSMCP_OCR_MAX_FRAMES` | `16` | Maximum keyframes sampled per video for OCR. |
| `SSMCP_PROXY` | *(empty)* | Optional HTTP/SOCKS5 proxy URL for outbound extraction requests. |
| `SSMCP_COOKIES_FILE` | *(empty)* | Optional path to a Netscape-format `cookies.txt` file. |
| `YOUTUBE_API_KEY` | *(empty)* | YouTube Data API v3 key (for `youtube_search` / `youtube_get_videos`). |
| `YOUTUBE_CLIENT_ID` | *(empty)* | Google OAuth Client ID for YouTube playlist write tools. |
| `YOUTUBE_CLIENT_SECRET` | *(empty)* | Google OAuth Client Secret for YouTube playlist write tools. |
| `YOUTUBE_REFRESH_TOKEN` | *(empty)* | Google OAuth Refresh Token for YouTube playlist write tools. |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
