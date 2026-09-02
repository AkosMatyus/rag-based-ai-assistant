from pydantic import BaseModel


class SourceChunk(BaseModel):
    start_line: int | None = None
    end_line: int | None = None
    score: float
    url: str | None = None
    content: str


class SourceGroup(BaseModel):
    file_path: str
    chunks: list[SourceChunk]


class ChatRequest(BaseModel):
    question: str
    repo_full_name: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceGroup] = []