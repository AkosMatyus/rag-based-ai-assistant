from database.database import get_db
from fastapi import APIRouter, Depends
from models.chat import ChatRequest, ChatResponse
from models.user import User
from routers.user import get_current_user
from services.rag_service import answer_question
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await answer_question(db, request.question, request.repo_full_name, current_user)
