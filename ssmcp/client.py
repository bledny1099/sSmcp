"""YouTube Data API v3 client: OAuth refresh, quota accounting, error mapping."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional

import httpx

API_BASE = "https://www.googleapis.com/youtube/v3"
TOKEN_URL = "https://oauth2.googleapis.com/token"

QUOTA_COSTS: Dict[str, int] = {
    "search.list": 100,
    "videos.list": 1,
    "channels.list": 1,
    "playlists.list": 1,
    "playlistItems.list": 1,
    "playlistItems.insert": 50,
    "playlistItems.delete": 50,
    "playlists.insert": 50,
    "subscriptions.list": 1,
}

DAILY_QUOTA = int(os.getenv("YT_DAILY_QUOTA", "10000"))
REQUEST_TIMEOUT = float(os.getenv("YT_HTTP_TIMEOUT", "20"))


class YouTubeError(RuntimeError):
    """API error carrying a message already phrased for an agent."""


class QuotaLedger:
    """In-process estimate of quota spent today (resets at midnight US/Pacific)."""

    def __init__(self, daily_limit: int = DAILY_QUOTA) -> None:
        self.daily_limit = daily_limit
        self._spent = 0
        self._period = self._current_period()

    @staticmethod
    def _current_period() -> int:
        return int((time.time() - 8 * 3600) // 86400)

    def _roll(self) -> None:
        period = self._current_period()
        if period != self._period:
            self._period = period
            self._spent = 0

    def charge(self, method: str) -> None:
        self._roll()
        self._spent += QUOTA_COSTS.get(method, 1)

    @property
    def spent(self) -> int:
        self._roll()
        return self._spent

    @property
    def remaining(self) -> int:
        return max(0, self.daily_limit - self.spent)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "daily_limit": self.daily_limit,
            "estimated_spent": self.spent,
            "estimated_remaining": self.remaining,
            "searches_left_if_only_searching": self.remaining // QUOTA_COSTS["search.list"],
            "playlist_writes_left": self.remaining // QUOTA_COSTS["playlistItems.insert"],
            "note": "Local estimate; resets at midnight US/Pacific. Other clients of the same Google Cloud project are not counted.",
        }


class YouTubeClient:
    """Thin async wrapper over the Data API with a cached OAuth access token."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        ledger: Optional[QuotaLedger] = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._access_token: Optional[str] = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()
        self.ledger = ledger or QuotaLedger()
        self._http = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

    @classmethod
    def from_env(cls, ledger: Optional[QuotaLedger] = None) -> "YouTubeClient":
        missing = [
            name
            for name in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN")
            if not os.getenv(name)
        ]
        if missing:
            raise YouTubeError(
                "Missing environment variables: "
                + ", ".join(missing)
                + ". Run scripts/get_refresh_token.py locally to mint a refresh token."
            )
        return cls(
            os.environ["YT_CLIENT_ID"],
            os.environ["YT_CLIENT_SECRET"],
            os.environ["YT_REFRESH_TOKEN"],
            ledger=ledger,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _token(self) -> str:
        async with self._lock:
            if self._access_token and time.time() < self._expires_at - 60:
                return self._access_token
            resp = await self._http.post(
                TOKEN_URL,
                data={
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "refresh_token": self._refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            if resp.status_code != 200:
                raise YouTubeError(
                    f"OAuth refresh failed ({resp.status_code}). The refresh token is likely revoked or the "
                    "OAuth consent screen is still in Testing mode (those tokens expire "
                    "after 7 days). Re-run scripts/get_refresh_token.py."
                )
            payload = resp.json()
            self._access_token = payload["access_token"]
            self._expires_at = time.time() + float(payload.get("expires_in", 3600))
            return self._access_token

    async def call(
        self,
        method: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call `resource.verb` (e.g. 'playlistItems.list') and return parsed JSON."""
        resource, verb = method.split(".", 1)
        url = f"{API_BASE}/{resource}"
        headers = {"Authorization": f"Bearer {await self._token()}"}

        if self.ledger.remaining < QUOTA_COSTS.get(method, 1):
            raise YouTubeError(
                f"Refusing to call {method}: estimated daily quota is exhausted "
                f"({self.ledger.spent}/{self.ledger.daily_limit} units). Quota resets at "
                "midnight US/Pacific. Use youtube_list_playlist_items (1 unit) instead of "
                "youtube_search (100 units) where possible."
            )

        try:
            if verb == "list":
                resp = await self._http.get(url, params=params, headers=headers)
            elif verb == "insert":
                resp = await self._http.post(url, params=params, json=body, headers=headers)
            elif verb == "delete":
                resp = await self._http.delete(url, params=params, headers=headers)
            else:
                raise YouTubeError(f"Unsupported verb: {verb}")
        except httpx.TimeoutException as exc:
            raise YouTubeError(f"{method} timed out after {REQUEST_TIMEOUT}s. Retry once.") from exc
        except httpx.HTTPError as exc:
            raise YouTubeError(f"{method} network error: {type(exc).__name__}") from exc

        self.ledger.charge(method)

        if resp.status_code == 204 or not resp.content:
            return {}
        if resp.status_code >= 400:
            raise YouTubeError(_explain(method, resp))
        return resp.json()


def _explain(method: str, resp: httpx.Response) -> str:
    """Turn a Google API error into something an agent can act on."""
    try:
        err = resp.json().get("error", {})
        reason = (err.get("errors") or [{}])[0].get("reason", "")
        detail = err.get("message", resp.text[:200])
    except Exception:
        reason, detail = "", resp.text[:200]

    if reason == "quotaExceeded":
        return (
            f"{method} failed: daily API quota exhausted. It resets at midnight US/Pacific. "
            "Prefer playlistItems.list (1 unit) over search.list (100 units)."
        )
    if reason in {"playlistNotFound", "playlistItemsNotAccessible"}:
        return (
            f"{method} failed: playlist not found or not accessible. Note that the real "
            "Watch Later playlist (id 'WL') and watch history ('HL') cannot be read through "
            "the API at all; use the migrated playlist ID instead."
        )
    if resp.status_code == 403:
        return f"{method} forbidden ({reason or 'no reason'}): {detail}"
    if resp.status_code == 404:
        return f"{method} not found: {detail}. Check the video or playlist ID."
    if resp.status_code == 401:
        return f"{method} unauthorized: the access token was rejected. Refresh token may be revoked."
    return f"{method} failed with HTTP {resp.status_code}: {detail}"
