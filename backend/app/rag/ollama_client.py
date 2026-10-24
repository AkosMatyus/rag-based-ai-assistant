"""Thin async client for the local Ollama server's REST API.

Both models of the pipeline - the embedding model and the LLM - run inside
Ollama, a separate program that loads the models (on the GPU when there is
one) and serves them over HTTP on ``OLLAMA_BASE_URL``. This module is the
only place that talks to it; embeddings.py and llm.py build the request
bodies. The calls are plain async HTTP (httpx), so a slow model call never
blocks the FastAPI event loop.

Every failure is turned into an ``OllamaError`` whose message is written
for the user (it ends up in a repository's error message or in the chat
UI), e.g. "Ollama isn't running" or "this model hasn't been downloaded".
"""

import httpx
from core.config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT_SECONDS


class OllamaError(RuntimeError):
    """Ollama couldn't be reached or refused a request. The message is
    meant to be shown to the user as-is."""


def _timeout() -> httpx.Timeout:
    # Fail fast if nothing is listening, but give generation plenty of time.
    return httpx.Timeout(OLLAMA_TIMEOUT_SECONDS, connect=5.0)


def _error_text(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error") or response.text)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


async def ollama_request(method: str, path: str, payload: dict | None = None) -> dict:
    """Sends one request to Ollama and returns the decoded JSON response."""

    try:
        async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=_timeout()) as client:
            response = await client.request(method, path, json=payload)
    except httpx.ConnectError as exc:
        raise OllamaError(
            f"Can't reach Ollama at {OLLAMA_BASE_URL}. Make sure Ollama is running "
            "(start the Ollama app, or run `ollama serve`)."
        ) from exc
    except httpx.TimeoutException as exc:
        raise OllamaError(
            f"Ollama didn't respond within {OLLAMA_TIMEOUT_SECONDS:g} seconds."
        ) from exc
    except httpx.HTTPError as exc:
        raise OllamaError(f"The request to Ollama failed: {exc}") from exc

    model = (payload or {}).get("model")
    if response.status_code == 404 and model:
        raise OllamaError(
            f'The model "{model}" isn\'t available in Ollama. '
            f"Download it with: ollama pull {model}"
        )
    if response.is_error:
        raise OllamaError(
            f"Ollama returned an error (HTTP {response.status_code}): {_error_text(response)}"
        )

    return response.json()


def _with_tag(name: str) -> str:
    """Ollama treats "model" and "model:latest" as the same model."""
    last_segment = name.rsplit("/", 1)[-1]
    return name if ":" in last_segment else f"{name}:latest"


async def missing_models(names: list[str]) -> list[str]:
    """Returns the models from ``names`` that haven't been pulled into Ollama
    yet. Raises OllamaError if Ollama itself can't be reached."""

    data = await ollama_request("GET", "/api/tags")
    available = {
        _with_tag(str(entry.get("name") or entry.get("model") or ""))
        for entry in data.get("models", [])
    }
    return [name for name in names if _with_tag(name) not in available]
