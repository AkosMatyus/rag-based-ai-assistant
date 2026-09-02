from pydantic import BaseModel, EmailStr, Field
from database.database import Base
from sqlalchemy import Column, Integer, String

class UserRequest(BaseModel):
    first_name: str
    last_name: str
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    username: str
    email: str

class UserLoginRequest(BaseModel):
    username: str
    password: str

class UserLoginResponse(BaseModel):
    message: str
    username: str
    access_token: str
    access_token_type: str = "bearer"

class ResponseMessage(BaseModel):
    message: str

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)

    username = Column(String, nullable=False)
    email = Column(String, nullable=False)

    hashed_password = Column(String, nullable=True)
    
    github_id = Column(String, nullable=True)
    github_username = Column(String, nullable=True)
    github_access_token = Column(String, nullable=True)

