#!/usr/bin/env python3
"""Interactive helper to generate a YouTube Data API v3 OAuth2 refresh token."""

from __future__ import annotations

import http.server
import os
import urllib.parse
import webbrowser
import httpx

SCOPES = "https://www.googleapis.com/auth/youtube"
REDIRECT_PORT = 8765
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"


def main() -> None:
    client_id = os.getenv("YOUTUBE_CLIENT_ID") or input("Enter YOUTUBE_CLIENT_ID: ").strip()
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET") or input("Enter YOUTUBE_CLIENT_SECRET: ").strip()

    if not client_id or not client_secret:
        raise SystemExit("Both YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET are required.")

    auth_params = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": SCOPES,
            "access_type": "offline",
            "prompt": "consent",
        }
    )
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{auth_params}"
    print(f"\nOpening browser for Google OAuth consent:\n{auth_url}\n")
    webbrowser.open(auth_url)

    captured_code: dict[str, str] = {}

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            if "code" in qs:
                captured_code["code"] = qs["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<h2>Authorization complete. You may close this tab.</h2>")
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing code parameter.")

        def log_message(self, format: str, *args: object) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", REDIRECT_PORT), CallbackHandler)
    print(f"Waiting for redirect on {REDIRECT_URI} ...")
    server.handle_request()

    code = captured_code.get("code")
    if not code:
        raise SystemExit("Failed to capture authorization code.")

    resp = httpx.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        },
        timeout=20.0,
    )
    resp.raise_for_status()
    tokens = resp.json()
    refresh_token = tokens.get("refresh_token")

    print("\nSuccess! Add these lines to your .env file:\n")
    print(f"YOUTUBE_CLIENT_ID={client_id}")
    print(f"YOUTUBE_CLIENT_SECRET={client_secret}")
    print(f"YOUTUBE_REFRESH_TOKEN={refresh_token}")


if __name__ == "__main__":
    main()
