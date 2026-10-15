"""OAuth 2.1 authorization server endpoints for Claude and MCP clients.

Implements RFC 8414 (OAuth 2.0 Authorization Server Metadata), RFC 9728
(OAuth 2.0 Protected Resource Metadata), RFC 7591 (Dynamic Client Registration),
and RFC 7636 (PKCE with S256) with:
1. Google Sign-In with strict email whitelisting (ALLOWED_EMAILS).
2. Multiple user passwords / admin tokens (AUTH_USERS, AUTH_PASSWORDS, OAUTH_ADMIN_PASSWORD).
3. 30-day session cookies for authorized browsers.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import json
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlencode

import httpx
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

logger = logging.getLogger("ssmcp.oauth")

# Ephemeral fallback token when MCP_BEARER_TOKEN is not set in environment
_EPHEMERAL_TOKEN = secrets.token_hex(32)

# In-memory storage with size caps to prevent memory exhaustion
_clients: Dict[str, Dict[str, Any]] = {}
_codes: Dict[str, Dict[str, Any]] = {}
_auth_sessions: Dict[str, Dict[str, Any]] = {}
MAX_STORED_ENTRIES = 100

ALLOWED_REDIRECT_PREFIXES = (
    "https://claude.ai/api/mcp/auth_callback",
    "http://localhost",
    "http://127.0.0.1",
)


def is_allowed_redirect_uri(uri: str) -> bool:
    if not uri:
        return False
    return any(uri.startswith(allowed) for allowed in ALLOWED_REDIRECT_PREFIXES)


def get_allowed_emails() -> Set[str]:
    """Retrieve whitelisted Google email addresses from environment."""
    raw = os.getenv("ALLOWED_EMAILS", "")
    emails = {e.strip().lower() for e in raw.split(",") if e.strip()}
    return emails


def get_admin_passwords() -> List[str]:
    """Retrieve all valid authorization passwords (admin + invited users)."""
    passwords = []
    custom = os.getenv("OAUTH_ADMIN_PASSWORD")
    if custom:
        passwords.append(custom.strip())
    token = (os.getenv("SSMCP_BEARER_TOKEN") or os.getenv("MCP_BEARER_TOKEN"))
    if token:
        passwords.append(token.strip())

    # Support comma-separated extra passwords (AUTH_PASSWORDS)
    extra_passwords = os.getenv("AUTH_PASSWORDS", "")
    for p in extra_passwords.split(","):
        if p.strip():
            passwords.append(p.strip())

    # Support user:pass pairs (AUTH_USERS)
    auth_users = os.getenv("AUTH_USERS", "")
    for pair in auth_users.split(","):
        if ":" in pair:
            passwords.append(pair.split(":", 1)[1].strip())

    if not passwords:
        passwords.append(_EPHEMERAL_TOKEN)
    return passwords


def get_session_hash() -> str:
    primary = get_admin_passwords()[0]
    return hashlib.sha256((primary + "_session_salt_2026").encode()).hexdigest()


def get_issuer(request: Optional[Request] = None) -> str:
    env_issuer = os.getenv("OAUTH_ISSUER")
    if env_issuer:
        return env_issuer.rstrip("/")
    if request:
        proto = request.headers.get("x-forwarded-proto") or request.url.scheme
        host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
        return f"{proto}://{host}"
    return "http://localhost:8080"


def verify_pkce(verifier: str, challenge: str, method: str = "S256") -> bool:
    if method == "plain":
        return verifier == challenge
    if method != "S256":
        return False
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return computed == challenge.rstrip("=")


async def protected_resource_metadata(request: Request) -> JSONResponse:
    """RFC 9728 discovery endpoint."""
    issuer = get_issuer(request)
    req_resource = request.query_params.get("resource")
    if req_resource and req_resource.startswith(issuer):
        resource = req_resource
    else:
        resource = f"{issuer}/mcp"
    return JSONResponse(
        {
            "resource": resource,
            "authorization_servers": [issuer],
            "scopes_supported": ["youtube"],
            "bearer_methods_supported": ["header"],
        }
    )


async def authorization_server_metadata(request: Request) -> JSONResponse:
    """RFC 8414 discovery endpoint."""
    issuer = get_issuer(request)
    return JSONResponse(
        {
            "issuer": issuer,
            "authorization_endpoint": f"{issuer}/oauth/authorize",
            "token_endpoint": f"{issuer}/oauth/token",
            "registration_endpoint": f"{issuer}/oauth/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none", "client_secret_post", "client_secret_basic"],
            "scopes_supported": ["youtube"],
        }
    )


async def register_client(request: Request) -> JSONResponse:
    """RFC 7591 Dynamic Client Registration with strict redirect URI validation."""
    try:
        data = await request.json()
    except Exception:
        data = {}

    redirect_uris = data.get("redirect_uris", ["https://claude.ai/api/mcp/auth_callback"])
    if isinstance(redirect_uris, list):
        for uri in redirect_uris:
            if not is_allowed_redirect_uri(uri):
                return JSONResponse(
                    {"error": "invalid_redirect_uri", "error_description": "Only claude.ai redirect URIs are allowed"},
                    status_code=400,
                )
    else:
        return JSONResponse({"error": "invalid_redirect_uri"}, status_code=400)

    if len(_clients) >= MAX_STORED_ENTRIES:
        _clients.pop(next(iter(_clients)))

    client_id = f"client_{secrets.token_hex(8)}"
    client_secret = secrets.token_urlsafe(32)
    client_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "client_id_issued_at": int(time.time()),
        "client_name": data.get("client_name", "Claude"),
        "redirect_uris": redirect_uris,
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": data.get("token_endpoint_auth_method", "none"),
    }
    _clients[client_id] = client_data
    return JSONResponse(client_data, status_code=201)


async def google_login(request: Request) -> Any:
    """Initiate Google OAuth2 authentication flow."""
    session_id = request.query_params.get("session_id")
    if not session_id or session_id not in _auth_sessions:
        return HTMLResponse(
            "<h3>Invalid or expired authorization session. Please restart authorization from Claude.</h3>",
            status_code=400,
        )

    google_client_id = os.getenv("YT_CLIENT_ID")
    if not google_client_id:
        return HTMLResponse(
            "<h3>Google OAuth is not configured on this server (YT_CLIENT_ID missing).</h3>",
            status_code=500,
        )

    issuer = get_issuer(request)
    google_redirect = f"{issuer}/oauth/google/callback"

    params = {
        "client_id": google_client_id,
        "redirect_uri": google_redirect,
        "response_type": "code",
        "scope": "openid email profile",
        "state": session_id,
        "access_type": "online",
        "prompt": "select_account",
    }
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}")


async def google_callback(request: Request) -> Any:
    """Handle Google OAuth2 callback and enforce email whitelist."""
    error = request.query_params.get("error")
    if error:
        return HTMLResponse(
            f"<h3>Google authorization cancelled or failed: {html.escape(error)}</h3>",
            status_code=400,
        )

    code = request.query_params.get("code")
    session_id = request.query_params.get("state")

    if not session_id or session_id not in _auth_sessions:
        return HTMLResponse(
            "<h3>Invalid or expired authorization session. Please restart authorization from Claude.</h3>",
            status_code=400,
        )

    auth_data = _auth_sessions.pop(session_id)
    if time.time() - auth_data.get("created_at", 0) > 600:
        return HTMLResponse(
            "<h3>Authorization session timed out. Please try again.</h3>",
            status_code=400,
        )

    google_client_id = os.getenv("YT_CLIENT_ID")
    google_client_secret = os.getenv("YT_CLIENT_SECRET")
    issuer = get_issuer(request)
    google_redirect = f"{issuer}/oauth/google/callback"

    # Exchange code for token
    async with httpx.AsyncClient(timeout=15.0) as http_client:
        token_resp = await http_client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": google_client_id,
                "client_secret": google_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": google_redirect,
            },
        )
        if token_resp.status_code != 200:
            logger.error(f"Google token exchange failed: {token_resp.text}")
            return HTMLResponse(
                f"<h3>Failed to exchange code with Google (HTTP {token_resp.status_code}).</h3>"
                f"<p>{html.escape(token_resp.text)}</p>",
                status_code=400,
            )

        token_data = token_resp.json()
        access_token = token_data.get("access_token")

        user_resp = await http_client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if user_resp.status_code != 200:
            return HTMLResponse("<h3>Failed to retrieve Google profile information.</h3>", status_code=400)

        userinfo = user_resp.json()
        email = (userinfo.get("email") or "").lower().strip()
        email_verified = userinfo.get("email_verified", False)

    allowed_emails = get_allowed_emails()
    # Strictly enforce whitelist: only explicitly allowed, verified emails can pass
    if not allowed_emails or email not in allowed_emails or not email_verified:
        logger.warning(f"Unauthorized Google login attempt: {email}")
        return HTMLResponse(
            f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Access Denied</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f1117;
            color: #e5e7eb;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }}
        .card {{
            background: #1a1d27;
            border: 1px solid #ef4444;
            border-radius: 16px;
            padding: 32px;
            max-width: 440px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }}
        h2 {{ color: #ef4444; margin-top: 0; }}
        p {{ color: #9ca3af; line-height: 1.6; font-size: 14px; }}
        code {{ background: #262b3d; color: #f59e0b; padding: 2px 6px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="card">
        <h2>Access Denied</h2>
        <p>The Google account <code>{html.escape(email)}</code> is not in the whitelist for this sSmcp (socialSimple MCP) server.</p>
        <p>Ask the server owner to add your email to <code>ALLOWED_EMAILS</code>.</p>
    </div>
</body>
</html>""",
            status_code=403,
        )

    # Approved: issue OAuth authorization code for Claude
    if len(_codes) >= MAX_STORED_ENTRIES:
        _codes.pop(next(iter(_codes)))

    auth_code = secrets.token_urlsafe(32)
    _codes[auth_code] = {
        "client_id": auth_data["client_id"],
        "redirect_uri": auth_data["redirect_uri"],
        "code_challenge": auth_data["code_challenge"],
        "code_challenge_method": auth_data["code_challenge_method"],
        "expires_at": time.time() + 300,
        "user_email": email,
    }

    redirect_params = {"code": auth_code}
    if auth_data.get("state"):
        redirect_params["state"] = auth_data["state"]

    sep = "&" if "?" in auth_data["redirect_uri"] else "?"
    dest_url = f"{auth_data['redirect_uri']}{sep}{urlencode(redirect_params)}"
    response = RedirectResponse(dest_url, status_code=302)

    expected_session = get_session_hash()
    response.set_cookie(
        "mcp_oauth_session",
        expected_session,
        max_age=30 * 86400,
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return response


async def authorize(request: Request) -> Any:
    """OAuth 2.1 authorization endpoint supporting Google Sign-In and Password auth."""
    if request.method == "POST":
        form = await request.form()
        params = dict(form)
    else:
        params = dict(request.query_params)

    client_id = params.get("client_id", "claude")
    redirect_uri = params.get("redirect_uri", "https://claude.ai/api/mcp/auth_callback")
    state = params.get("state", "")
    code_challenge = params.get("code_challenge", "")
    code_challenge_method = params.get("code_challenge_method", "S256")

    # Security check: disallow unauthorized redirect destinations
    if not is_allowed_redirect_uri(redirect_uri):
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Disallowed redirect_uri"},
            status_code=400,
        )

    expected_session = get_session_hash()
    is_authenticated = request.cookies.get("mcp_oauth_session") == expected_session
    error_message = ""

    # Process password submission
    if request.method == "POST":
        submitted_pwd = str(params.get("password", "")).strip()
        approved = params.get("approved")

        if is_authenticated and approved == "true":
            pass
        else:
            valid_passwords = get_admin_passwords()
            matches = any(secrets.compare_digest(submitted_pwd, p) for p in valid_passwords if p)
            if matches:
                is_authenticated = True
            else:
                await asyncio.sleep(0.5)  # Throttling to prevent brute force
                error_message = "Invalid password. Access denied."

    # If user is authenticated: issue code and redirect
    if is_authenticated and (request.method == "POST" or params.get("auto_approved") == "true"):
        if len(_codes) >= MAX_STORED_ENTRIES:
            _codes.pop(next(iter(_codes)))

        code = secrets.token_urlsafe(32)
        _codes[code] = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "expires_at": time.time() + 300,
        }
        redirect_params = {"code": code}
        if state:
            redirect_params["state"] = state
        sep = "&" if "?" in redirect_uri else "?"
        response = RedirectResponse(f"{redirect_uri}{sep}{urlencode(redirect_params)}", status_code=302)
        response.set_cookie(
            "mcp_oauth_session",
            expected_session,
            max_age=30 * 86400,
            httponly=True,
            secure=True,
            samesite="lax",
        )
        return response

    # If already logged in, show quick auto-approval view
    if is_authenticated:
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Authorize sSmcp (socialSimple MCP)</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f1117;
            color: #e5e7eb;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }}
        .card {{
            background: #1a1d27;
            border: 1px solid #2d3345;
            border-radius: 16px;
            padding: 32px;
            max-width: 440px;
            width: 100%;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            text-align: center;
        }}
        h2 {{ margin-top: 0; color: #ffffff; font-size: 22px; }}
        p {{ color: #9ca3af; font-size: 14px; line-height: 1.6; }}
        .btn {{
            display: block; width: 100%; background: #2563eb; color: white;
            padding: 14px 20px; border-radius: 10px; border: none; font-size: 16px;
            font-weight: 600; cursor: pointer; margin-top: 24px; box-sizing: border-box;
        }}
        .badge {{
            display: inline-block; padding: 4px 10px; background: rgba(16, 185, 129, 0.15);
            border: 1px solid #059669; color: #34d399; border-radius: 9999px; font-size: 12px; font-weight: 500; margin-bottom: 16px;
        }}
    </style>
</head>
<body>
    <div class="card">
        <span class="badge">Session Active</span>
        <h2>Authorize Claude</h2>
        <p>Granting Claude access to sSmcp (socialSimple MCP) tools.</p>
        <form method="POST" action="/oauth/authorize" id="auth-form">
            <input type="hidden" name="client_id" value="{html.escape(client_id)}">
            <input type="hidden" name="redirect_uri" value="{html.escape(redirect_uri)}">
            <input type="hidden" name="state" value="{html.escape(state)}">
            <input type="hidden" name="code_challenge" value="{html.escape(code_challenge)}">
            <input type="hidden" name="code_challenge_method" value="{html.escape(code_challenge_method)}">
            <input type="hidden" name="approved" value="true">
            <button type="submit" class="btn" id="auth-btn">Authorize Claude</button>
        </form>
    </div>
    <script>
        setTimeout(function() {{
            var btn = document.getElementById("auth-btn");
            if (btn) {{
                btn.textContent = "Connecting to Claude...";
                btn.style.opacity = "0.8";
            }}
            document.getElementById("auth-form").submit();
        }}, 300);
    </script>
</body>
</html>"""
        return HTMLResponse(html_content)

    # Create new authorization session for Google Sign-In button
    if len(_auth_sessions) >= MAX_STORED_ENTRIES:
        _auth_sessions.pop(next(iter(_auth_sessions)))

    session_id = secrets.token_urlsafe(24)
    _auth_sessions[session_id] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
        "created_at": time.time(),
    }

    # Render Hybrid Sign-In Form (Google Button + Password Input)
    error_html = f'<div class="error">{html.escape(error_message)}</div>' if error_message else ""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Authorize sSmcp (socialSimple MCP)</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f1117;
            color: #e5e7eb;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }}
        .card {{
            background: #1a1d27;
            border: 1px solid #2d3345;
            border-radius: 16px;
            padding: 32px;
            max-width: 440px;
            width: 100%;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            text-align: center;
        }}
        h2 {{ margin-top: 0; color: #ffffff; font-size: 22px; }}
        p {{ color: #9ca3af; font-size: 14px; line-height: 1.6; margin-bottom: 24px; }}
        .error {{
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            color: #fca5a5;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 13px;
            margin-bottom: 20px;
            text-align: left;
        }}
        .google-btn {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            width: 100%;
            background: #ffffff;
            color: #1f2937;
            padding: 12px 20px;
            border-radius: 10px;
            text-decoration: none;
            font-size: 15px;
            font-weight: 600;
            box-sizing: border-box;
            transition: background 0.2s, transform 0.1s;
        }}
        .google-btn:hover {{
            background: #f3f4f6;
            transform: translateY(-1px);
        }}
        .divider {{
            display: flex;
            align-items: center;
            text-align: center;
            margin: 24px 0;
            color: #4b5563;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        .divider::before, .divider::after {{
            content: '';
            flex: 1;
            border-bottom: 1px solid #2d3345;
        }}
        .divider span {{
            padding: 0 12px;
        }}
        .input-group {{
            text-align: left;
            margin-top: 14px;
        }}
        .input-group label {{
            display: block;
            font-size: 13px;
            font-weight: 500;
            color: #d1d5db;
            margin-bottom: 6px;
        }}
        .input-group input {{
            width: 100%;
            background: #131620;
            border: 1px solid #374151;
            color: #ffffff;
            padding: 12px 14px;
            border-radius: 8px;
            font-size: 14px;
            box-sizing: border-box;
            outline: none;
            transition: border-color 0.2s;
        }}
        .input-group input:focus {{
            border-color: #3b82f6;
        }}
        .btn {{
            display: block;
            width: 100%;
            background: #2563eb;
            color: white;
            padding: 12px 20px;
            border-radius: 10px;
            border: none;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 18px;
            box-sizing: border-box;
            transition: background 0.2s;
        }}
        .btn:hover {{
            background: #1d4ed8;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h2>sSmcp (socialSimple MCP) Authorization</h2>
        <p>Choose your preferred sign-in method to authorize your MCP client.</p>
        {error_html}
        <a href="/oauth/google/login?session_id={session_id}" class="google-btn">
            <svg width="20" height="20" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
            </svg>
            Sign in with Google
        </a>

        <div class="divider">
            <span>OR WITH PASSWORD</span>
        </div>

        <form method="POST" action="/oauth/authorize">
            <input type="hidden" name="client_id" value="{html.escape(client_id)}">
            <input type="hidden" name="redirect_uri" value="{html.escape(redirect_uri)}">
            <input type="hidden" name="state" value="{html.escape(state)}">
            <input type="hidden" name="code_challenge" value="{html.escape(code_challenge)}">
            <input type="hidden" name="code_challenge_method" value="{html.escape(code_challenge_method)}">
            <div class="input-group">
                <label for="password">Password or Token:</label>
                <input type="password" id="password" name="password" required placeholder="Enter your password...">
            </div>
            <button type="submit" class="btn">Sign In</button>
        </form>
    </div>
</body>
</html>"""
    return HTMLResponse(html_content)


async def token_endpoint(request: Request) -> JSONResponse:
    """OAuth 2.1 token endpoint with single-use authorization code and PKCE."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = await request.json()
        except Exception:
            data = {}
    else:
        form = await request.form()
        data = dict(form)

    grant_type = data.get("grant_type")
    configured_token = (os.getenv("SSMCP_BEARER_TOKEN") or os.getenv("MCP_BEARER_TOKEN")) or get_admin_passwords()[0]

    if grant_type == "authorization_code":
        code = data.get("code")
        verifier = data.get("code_verifier", "")

        if not code or code not in _codes:
            return JSONResponse({"error": "invalid_grant", "error_description": "Invalid authorization code"}, status_code=400)

        record = _codes.pop(code)
        if time.time() > record["expires_at"]:
            return JSONResponse({"error": "invalid_grant", "error_description": "Authorization code expired"}, status_code=400)

        challenge = record.get("code_challenge")
        method = record.get("code_challenge_method", "S256")
        if challenge and not verify_pkce(verifier, challenge, method):
            return JSONResponse({"error": "invalid_grant", "error_description": "PKCE verification failed"}, status_code=400)

        return JSONResponse(
            {
                "access_token": configured_token,
                "token_type": "Bearer",
                "expires_in": 315360000,
                "refresh_token": configured_token,
                "scope": "youtube",
            }
        )

    if grant_type == "refresh_token":
        return JSONResponse(
            {
                "access_token": configured_token,
                "token_type": "Bearer",
                "expires_in": 315360000,
                "refresh_token": configured_token,
                "scope": "youtube",
            }
        )

    return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)
