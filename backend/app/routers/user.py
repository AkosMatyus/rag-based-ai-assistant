from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import get_db
from models.user import User, UserLoginRequest, UserLoginResponse, UserRequest, UserResponse
from utils import create_access_token, decode_access_token, hash_password, verify_password

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")


@router.post("/register", response_model=UserResponse)
async def create_user(user: UserRequest, db: AsyncSession = Depends(get_db)):
    existing_username_result = await db.execute(select(User).where(User.username == user.username))
    existing_username = existing_username_result.scalar_one_or_none()
    if existing_username:
        raise HTTPException(status_code=409, detail="This username is already used by someone")

    existing_email_result = await db.execute(select(User).where(User.email == user.email))
    existing_email = existing_email_result.scalar_one_or_none()
    if existing_email:
        raise HTTPException(status_code=409, detail="This email is already used by someone")

    new_user = User(
        first_name=user.first_name,
        last_name=user.last_name,
        username=user.username,
        email=user.email,
        hashed_password=hash_password(user.password),
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return UserResponse(username=new_user.username, email=new_user.email)


@router.post("/login", response_model=UserLoginResponse)
async def login(user_req: UserLoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == user_req.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(user_req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = create_access_token(data={"sub": user.username})

    return UserLoginResponse(
        message="Login successful",
        username=user.username,
        access_token=access_token,
    )


async def get_current_user(token: str = Security(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        username: str = payload.get("sub")

        if username is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception

    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    return user


def verify_access_token(token: str):
    try:
        payload = decode_access_token(token)
        if payload is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_username(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    username = verify_access_token(token)
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user.username