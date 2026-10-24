"""Answer generation with a local LLM served by Ollama (``/api/chat``).

The model (LLM_MODEL_NAME, e.g. ``qwen3.5:4b``) runs inside Ollama, which
loads it onto the GPU when one is available - no external API is called.
The request is a plain async HTTP call, so generating an answer doesn't
block the FastAPI event loop.
"""

import logging

from core.config import (
    LLM_CONTEXT_LENGTH,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    OLLAMA_KEEP_ALIVE,
)
from rag.ollama_client import ollama_request

logger = logging.getLogger(__name__)

# Each sentence ends with a space on purpose: the pieces are concatenated,
# and without it the sentences would run into each other ("...helpful.Only").
SYSTEM_PROMPT = (
    "You are a helpful software development assistant who answers questions "
    "based on context taken from a given GitHub codebase. "
    "Rely exclusively on the provided context; do not use outside knowledge "
    "and do not make up anything that is not present in it. "
    "If the answer cannot be found in the context, clearly say that you cannot "
    "answer the question based on the available code snippets. "
    "Answer concisely, in the same language as the question. "
    "Only mention file paths that appear in the provided context, exactly as "
    "written there - do not invent, rename or normalize them. "
    "If the question asks where something is implemented, defined, handled or "
    "executed, prioritize source code files over documentation files, and use "
    "documentation only as secondary explanation."
)

# Sampling settings follow Qwen's recommendation for non-thinking mode, with
# a lower temperature because answers must stick to the context. They are
# sent with every request because they override the model's own defaults in
# Ollama - qwen3.5's Ollama build, for example, ships with temperature 1 and
# presence_penalty 1.5. Repetition/presence penalties are switched off on
# purpose: they also penalize tokens copied from the prompt, which garbled
# file paths and identifiers (e.g. "reranker.py" -> "reanker.py").
GENERATION_OPTIONS = {
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": 20,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
}


def _build_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context of codebase:\n\n{context}\n\n"
                f"Question: {question}"
            ),
        },
    ]


def _strip_thinking(text: str) -> str:
    """Safety net: if the model still emitted a <think>...</think> block
    inside the answer, keep only what comes after it."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


async def generate_answer(question: str, context: str) -> str:
    data = await ollama_request(
        "POST",
        "/api/chat",
        {
            "model": LLM_MODEL_NAME,
            "messages": _build_messages(question, context),
            "stream": False,
            # Thinking models (Qwen3, Qwen3.5, ...) think by default in
            # Ollama: they write a long reasoning block before the answer,
            # which used up the token budget and cut the real answer off.
            # Ollama only rejects think=true for models that can't think, so
            # think=false is safe to send to any model.
            "think": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                **GENERATION_OPTIONS,
                "num_predict": LLM_MAX_NEW_TOKENS,
                # Without this, Ollama uses its own default context size
                # (4096 tokens on GPUs under 24 GB) and silently cuts a
                # longer prompt.
                "num_ctx": LLM_CONTEXT_LENGTH,
            },
        },
    )

    prompt_tokens = data.get("prompt_eval_count") or 0
    if prompt_tokens + LLM_MAX_NEW_TOKENS > LLM_CONTEXT_LENGTH:
        logger.warning(
            "The prompt (%d tokens) plus the answer budget (%d) exceeds "
            "LLM_CONTEXT_LENGTH (%d); part of the context may have been cut. "
            "Raise LLM_CONTEXT_LENGTH or lower RAG_TOP_K / RAG_CHUNK_SIZE.",
            prompt_tokens,
            LLM_MAX_NEW_TOKENS,
            LLM_CONTEXT_LENGTH,
        )
    if data.get("done_reason") == "length":
        logger.info("The answer was cut off at LLM_MAX_NEW_TOKENS=%d.", LLM_MAX_NEW_TOKENS)

    answer = _strip_thinking(str((data.get("message") or {}).get("content") or ""))
    return answer or "The model returned an empty answer. Try asking again."
