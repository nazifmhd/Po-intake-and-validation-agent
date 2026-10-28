"""Provider-agnostic structured-output LLM client (Anthropic or OpenAI), with vision.

* Structured output uses each provider's native JSON-schema mode, then Pydantic validation.
  A response that fails validation is re-prompted with the validation error (bounded), and
  only then surfaces as an error - which the graph routes to human review.
* Transient failures (rate limits, timeouts, 5xx, connection) retry with exponential backoff.
* A refusal or truncated response is a hard error, never a silently-accepted partial result.
"""
from __future__ import annotations

import base64
import copy
import json
import logging
import mimetypes
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

# Models that accept the server-side refusal fallback ("fallbacks": "default")
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(Exception):
    """Non-retryable LLM failure (refusal, truncation, invalid output after retries)."""


def strict_json_schema(model: Type[BaseModel]) -> dict:
    """Pydantic schema -> strict JSON schema accepted by both providers' structured-output modes:
    every property required (nullability via anyOf-null), no extra properties, no defaults or
    numeric bounds (bounds are enforced by Pydantic validation after the call)."""
    schema = copy.deepcopy(model.model_json_schema())

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("default", "title", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                node.pop(key, None)
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"].keys())
                node["additionalProperties"] = False
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return schema


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
                "OverloadedError", "ServiceUnavailableError", "RetryableError", "DeadlineExceededError"}:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


def _image_b64(path: str) -> tuple[str, str]:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as fh:
        return mime, base64.b64encode(fh.read()).decode("ascii")


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.provider = self.settings.llm_provider
        if self.provider == "anthropic":
            import anthropic

            kwargs: dict[str, Any] = {"max_retries": 0, "timeout": self.settings.llm_timeout_seconds}
            if self.settings.anthropic_api_key:
                kwargs["api_key"] = self.settings.anthropic_api_key
            self._client = anthropic.Anthropic(**kwargs)
            self.model = self.settings.anthropic_model
        elif self.provider == "openai":
            import openai

            self._client = openai.OpenAI(
                api_key=self.settings.openai_api_key or None,
                max_retries=0,
                timeout=self.settings.llm_timeout_seconds,
            )
            self.model = self.settings.openai_model
        else:
            raise LLMError(f"LLM provider '{self.provider}' has no LLM client")

    # ------------------------------------------------------------------ public
    def structured(
        self,
        *,
        system: str,
        text: str,
        response_model: Type[T],
        image_paths: list[str] | None = None,
        max_tokens: int = 16000,
    ) -> T:
        """One structured call; re-prompts with the validation error on malformed output."""
        schema = strict_json_schema(response_model)
        feedback: str | None = None
        attempts = self.settings.extraction_validation_retries + 1
        last_err: Exception | None = None
        for attempt in range(1, attempts + 1):
            user_text = text if feedback is None else (
                f"{text}\n\nYour previous answer was rejected by schema validation:\n{feedback}\n"
                "Return a corrected JSON object that satisfies the schema exactly."
            )
            raw = self._call_with_backoff(system, user_text, schema, response_model.__name__,
                                          image_paths or [], max_tokens)
            try:
                return response_model.model_validate_json(raw)
            except ValidationError as exc:
                last_err = exc
                feedback = str(exc)[:2000]
                logger.warning("LLM output failed validation", extra={"attempt": attempt, "error": feedback[:300]})
        raise LLMError(f"Output failed schema validation after {attempts} attempts: {last_err}")

    # ----------------------------------------------------------------- private
    def _call_with_backoff(self, system, text, schema, name, image_paths, max_tokens) -> str:
        @retry(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(self.settings.llm_max_retries),
            wait=wait_exponential_jitter(initial=1, max=30),
            reraise=True,
            before_sleep=lambda rs: logger.warning(
                "Transient LLM error, backing off",
                extra={"attempt": rs.attempt_number, "error": str(rs.outcome.exception())[:200]},
            ),
        )
        def _do() -> str:
            if self.provider == "anthropic":
                return self._anthropic(system, text, schema, image_paths, max_tokens)
            return self._openai(system, text, schema, name, image_paths, max_tokens)

        return _do()

    def _anthropic(self, system, text, schema, image_paths, max_tokens) -> str:
        content: list[dict] = []
        for p in image_paths:
            mime, data = _image_b64(p)
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
        content.append({"type": "text", "text": text})

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.settings.anthropic_effort,
                           "format": {"type": "json_schema", "schema": schema}},
        )
        if self.settings.anthropic_server_fallback and self.model in _FALLBACK_MODELS:
            kwargs["extra_headers"] = {"anthropic-beta": _FALLBACK_BETA}
            kwargs["extra_body"] = {"fallbacks": "default"}

        response = self._client.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMError(f"Model refused the request: {details}")
        if response.stop_reason == "max_tokens":
            raise LLMError("Model output was truncated (max_tokens)")
        texts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        if not texts:
            raise LLMError("Model returned no text content")
        return texts[0]

    def _openai(self, system, text, schema, name, image_paths, max_tokens) -> str:
        content: list[dict] = [{"type": "text", "text": text}]
        for p in image_paths:
            mime, data = _image_b64(p)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}", "detail": "high"}})
        response = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
            response_format={"type": "json_schema",
                             "json_schema": {"name": name, "schema": schema, "strict": True}},
        )
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMError(f"Model refused the request: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise LLMError("Model output was truncated (max_tokens)")
        if not choice.message.content:
            raise LLMError("Model returned no content")
        return choice.message.content


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=str, indent=2)
