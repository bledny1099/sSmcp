#!/usr/bin/env bash
# Export browser cookies locally and upload to a remote sSmcp server.
# Usage: ./scripts/sync_cookies.sh user@your-server.example.com /opt/sSmcp/cookies.txt chrome

set -euo pipefail

REMOTE_HOST="${1:-}"
REMOTE_PATH="${2:-/opt/sSmcp/cookies.txt}"
BROWSER="${3:-chrome}"

if [[ -z "$REMOTE_HOST" ]]; then
  echo "Usage: $0 <user@remote-host> [remote-cookies-path] [browser]"
  exit 1
fi

TMP_COOKIES="$(mktemp -t ssmcp_cookies.XXXXXX.txt)"
trap 'rm -f "$TMP_COOKIES"' EXIT

echo "Exporting cookies from $BROWSER via yt-dlp..."
yt-dlp --cookies-from-browser "$BROWSER" --cookies "$TMP_COOKIES" --skip-download "https://www.youtube.com" >/dev/null 2>&1 || true

echo "Uploading cookies to ${REMOTE_HOST}:${REMOTE_PATH}..."
scp "$TMP_COOKIES" "${REMOTE_HOST}:${REMOTE_PATH}"
echo "Cookies synced."
