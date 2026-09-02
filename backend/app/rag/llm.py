"""Answer generation using a locally-running Hugging Face model.

The model is downloaded from the Hugging Face Hub on first use (cached
under the usual ``~/.cache/huggingface`` directory afterwards) and runs
entirely locally via ``transformers`` - no external API calls. Loading
and generation are both synchronous/CPU-heavy, so they run in a worker
thread through ``asyncio.to_thread``.
"""

import asyncio
import threading

from core.config import LLM_DEVICE, LLM_MAX_NEW_TOKENS, LLM_MODEL_NAME

_pipeline = None
_pipeline_lock = threading.Lock()

SYSTEM_PROMPT = (
    "You are a helpful software development assistant who answers "
    "questions based on context taken from a given GitHub codebase. "
    "Rely exclusively on the provided context, and do not make up anything "
    "that is not present in it. If the answer cannot be found in the context,"
    " clearly state that you cannot answer the question based on the available"
    " code snippets. Respond concisely, in the given language, and reference relevant"
    " filenames when helpful." \
    "Only mention file paths that appear in the provided context metadata."
    "Do not invent, normalize, rename, or infer file paths."
    "If a file path is not present in the retrieved sources, do not include it in the answer."
    "If the user asks where something is implemented, defined, handled, or executed,"
    " prioritize source code files over documentation files. Documentation files may "
    "only be used as secondary explanation."
    "When the question asks where something is implemented in source code, prioritize files under `src/` "
    "and clearly separate source files from documentation files."
    "DO NOT RELY ON YOUR BASIC KNOWLEDGE, ONLY ANSWER VIA THE GIVEN SOURCES," \
    "IF YOU DO NOT KNOW THE ANSWER SAY THAT"
)


def _resolve_device() -> int:
    """Returns the device index expected by transformers' pipeline() call."""
    setting = LLM_DEVICE.strip().lower()

    if setting == "cpu":
        return -1
    if setting == "cuda":
        return 0
    if setting == "auto":
        import torch

        return 0 if torch.cuda.is_available() else -1

    try:
        return int(setting)
    except ValueError:
        return -1


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from transformers import pipeline

                device = _resolve_device()
                _pipeline = pipeline(
                    "text-generation",
                    model=LLM_MODEL_NAME,
                    device=device,
                    torch_dtype="auto",
                )

                if _pipeline.tokenizer.pad_token_id is None:
                    _pipeline.tokenizer.pad_token_id = _pipeline.tokenizer.eos_token_id
    return _pipeline


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


def _generate(question: str, context: str) -> str:
    text_generator = _get_pipeline()
    messages = _build_messages(question, context)

    outputs = text_generator(
        messages,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=True,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
    )

    generated = outputs[0]["generated_text"]

    # Chat-style input returns the full conversation (list of messages);
    # the newly generated reply is the last one.
    if isinstance(generated, list):
        return str(generated[-1].get("content", "")).strip()

    return str(generated).strip()


async def generate_answer(question: str, context: str) -> str:
    return await asyncio.to_thread(_generate, question, context)
