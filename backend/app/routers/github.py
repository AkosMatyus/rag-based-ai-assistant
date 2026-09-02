import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import (
    FRONTEND_URL,
    GITHUB_CLIENT_ID,
    GITHUB_CLIENT_SECRET,
    GITHUB_REDIRECT_URI,
)
from database.database import get_db
from models.user import User
from routers.user import get_current_user
from utils import create_access_token


router = APIRouter()


async def make_unique_username(db: AsyncSession, base_username: str) -> str:
    candidate = base_username
    counter = 1

    while True:
        result = await db.execute(
            select(User).where(User.username == candidate)
        )

        existing_user = result.scalar_one_or_none()

        if existing_user is None:
            return candidate

        candidate = f"{base_username}_{counter}"
        counter += 1


@router.get("/login")
async def github_login():
    if not GITHUB_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="GitHub Client ID is missing"
        )

    state = secrets.token_urlsafe(32)

    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": GITHUB_REDIRECT_URI,
        "scope": "read:user repo",
        "state": state,
    }

    github_url = f"https://github.com/login/oauth/authorize?{urlencode(params)}"

    response = RedirectResponse(github_url)

    response.set_cookie(
        key="github_oauth_state",
        value=state,
        httponly=True,
        samesite="lax",
    )

    return response


@router.get("/callback")
async def github_callback(
    request: Request,
    code: str = Query(...),
    state: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    saved_state = request.cookies.get("github_oauth_state")

    if not saved_state or saved_state != state:
        raise HTTPException(
            status_code=400,
            detail="Invalid GitHub OAuth state"
        )

    if not GITHUB_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="GitHub Client Secret is missing"
        )

    async with httpx.AsyncClient(timeout=10) as client:
        token_response = await client.post(
            "https://github.com/login/oauth/access_token",
            headers={
                "Accept": "application/json",
            },
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": GITHUB_REDIRECT_URI,
            },
        )

        token_data = token_response.json()
        github_access_token = token_data.get("access_token")

        if not github_access_token:
            raise HTTPException(
                status_code=400,
                detail="Could not get GitHub access token"
            )

        github_user_response = await client.get(
            "https://api.github.com/user",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {github_access_token}",
            },
        )

    github_user = github_user_response.json()

    github_id = str(github_user["id"])
    github_username = github_user["login"]

    result = await db.execute(
        select(User).where(User.github_id == github_id)
    )

    user = result.scalar_one_or_none()

    if user is None:
        username = await make_unique_username(db, github_username)

        fallback_email = f"{github_id}+{github_username}@github.local"

        user = User(
            first_name=github_user.get("name") or github_username,
            last_name="",
            username=username,
            email=github_user.get("email") or fallback_email,
            hashed_password=None,
            github_id=github_id,
            github_username=github_username,
            github_access_token=github_access_token,
        )

        db.add(user)

    else:
        user.github_username = github_username
        user.github_access_token = github_access_token

    await db.commit()
    await db.refresh(user)

    app_token = create_access_token(data={"sub": user.username})

    frontend_params = urlencode(
        {
            "token": app_token,
            "username": user.username,
        }
    )

    response = RedirectResponse(
        f"{FRONTEND_URL}/github/callback?{frontend_params}"
    )

    response.delete_cookie("github_oauth_state")

    print("Redirecting to frontend:", f"{FRONTEND_URL}/github/callback?{frontend_params}")

    return response


@router.get("/repos")
async def get_github_repos(
    current_user: User = Depends(get_current_user),
):
    if not current_user.github_access_token:
        raise HTTPException(
            status_code=400,
            detail="This user is not connected to GitHub"
        )

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            "https://api.github.com/user/repos",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {current_user.github_access_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            params={
                "visibility": "all",
                "affiliation": "owner,collaborator,organization_member",
                "sort": "updated",
                "per_page": 50,
            },
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail="Could not fetch GitHub repositories"
        )

    repos = response.json()

    return [
        {
            "id": repo["id"],
            "name": repo["name"],
            "full_name": repo["full_name"],
            "private": repo["private"],
            "html_url": repo["html_url"],
            "language": repo.get("language"),
            "updated_at": repo.get("updated_at"),
            "affiliation": (
                "owner"
                if repo.get("owner", {}).get("login") == current_user.github_username
                else "collaborator"
            ),
        }
        for repo in repos
    ]