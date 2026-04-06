"""Google OAuth2 routes, session management, and API key CRUD endpoints."""
import os
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from authlib.integrations.starlette_client import OAuth
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from pydantic import BaseModel

from auth.config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, SECRET_KEY
from auth.db import upsert_user, get_user, store_api_key, get_api_key, delete_api_key, has_api_key

router = APIRouter(prefix="/auth")

# ── OAuth setup ─────────────────────────────────────────────────────────────
oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

signer = URLSafeTimedSerializer(SECRET_KEY)
COOKIE_NAME = "rag_session"
MAX_AGE = 60 * 60 * 24 * 7  # 7 days


# ── Session helpers ─────────────────────────────────────────────────────────
def create_session_cookie(user_id: str) -> str:
    return signer.dumps({"user_id": user_id})


def decode_session_cookie(token: str) -> str | None:
    try:
        data = signer.loads(token, max_age=MAX_AGE)
        return data.get("user_id")
    except (BadSignature, SignatureExpired):
        return None


def get_current_user(request: Request) -> dict | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    user_id = decode_session_cookie(token)
    if not user_id:
        return None
    return get_user(user_id)


def require_auth(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ── OAuth routes ─────────────────────────────────────────────────────────────
@router.get("/login")
async def login(request: Request):
    """Redirect to Google OAuth."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        from fastapi.responses import HTMLResponse
        return HTMLResponse(
            """<html><body style="font-family:sans-serif;background:#0f1117;color:#e4e6f0;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
            <div style="text-align:center;padding:40px;background:#1a1d27;border-radius:16px;max-width:480px">
            <h2 style="color:#fc8181;margin-bottom:12px">&#9888; Google OAuth Not Configured</h2>
            <p style="color:#8892a4;line-height:1.6;margin-bottom:16px">
              <code>GOOGLE_CLIENT_ID</code> and <code>GOOGLE_CLIENT_SECRET</code> are missing from <code>.env</code>.
            </p>
            <p style="color:#8892a4;font-size:13px">
              1. Go to <a href="https://console.cloud.google.com/" style="color:#63b3ed" target="_blank">console.cloud.google.com</a><br/>
              2. Create a project &rarr; OAuth 2.0 Client ID (Web Application)<br/>
              3. Add <code>http://localhost:8000/auth/callback</code> as redirect URI<br/>
              4. Paste the credentials into <code>.env</code> and restart
            </p>
            </div></body></html>""",
            status_code=503,
        )
    redirect_uri = str(request.base_url) + "auth/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def auth_callback(request: Request):
    """Handle Google OAuth callback."""
    try:
        token = await oauth.google.authorize_access_token(request)
        userinfo = token.get("userinfo") or await oauth.google.userinfo(token=token)
        user_id = userinfo["sub"]
        email = userinfo.get("email", "")
        name = userinfo.get("name", email)
        picture = userinfo.get("picture", "")

        upsert_user(user_id, email, name, picture)

        session_token = create_session_cookie(user_id)
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            COOKIE_NAME,
            session_token,
            max_age=MAX_AGE,
            httponly=True,
            samesite="lax",
        )
        return response
    except Exception as e:
        return RedirectResponse(url=f"/login?error={str(e)}", status_code=302)


@router.get("/logout")
async def logout():
    """Clear session cookie."""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(COOKIE_NAME)
    return response


@router.get("/me")
async def me(request: Request):
    """Return current user info."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "picture": user["picture"],
        "has_api_key": has_api_key(user["id"]),
    }


# ── API Key CRUD ─────────────────────────────────────────────────────────────
class ApiKeyRequest(BaseModel):
    api_key: str


@router.post("/api-key")
async def save_api_key(body: ApiKeyRequest, user: dict = Depends(require_auth)):
    """Store an encrypted API key for the current user."""
    store_api_key(user["id"], body.api_key)
    return {"status": "saved"}


@router.get("/api-key/status")
async def api_key_status(user: dict = Depends(require_auth)):
    """Check if user has an API key stored."""
    return {"has_api_key": has_api_key(user["id"])}


@router.delete("/api-key")
async def remove_api_key(user: dict = Depends(require_auth)):
    """Delete the user's stored API key."""
    delete_api_key(user["id"])
    return {"status": "deleted"}
