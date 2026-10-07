"""Environment-only OpenAI-compatible client for wording realization.

This module is deliberately downstream of :mod:`limo4si.semantic_gt`.  It
transports an answer-blind, signed language request and returns provider JSON;
it has no access to annotations, media, correct-option ids, or task geometry.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


class LanguageTransportError(RuntimeError):
    """Transient provider/network failure that is safe to retry verbatim."""


def _post_json(
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    timeout_sec: float,
) -> Mapping[str, Any]:
    request = Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        # Never include request headers or credentials in an exception.
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        error_type = LanguageTransportError if exc.code in {408, 429, 500, 502, 503, 504} else RuntimeError
        raise error_type(f"language provider returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise LanguageTransportError(f"language provider connection failed: {exc.reason}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("language provider response must be a JSON object")
    return payload


@dataclass
class OpenAICompatibleStructuredClient:
    """Minimal strict-JSON client implementing ``StructuredOutputClient``."""

    api_key: str
    base_url: str
    model: str
    timeout_sec: float = 90.0
    transport: Transport = _post_json
    maximum_transport_attempts: int = 4
    retry_delay_sec: float = 1.0

    def create_structured_output(
        self,
        *,
        system_prompt: str,
        payload: Mapping[str, Any],
        json_schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not self.api_key.strip():
            raise RuntimeError("language API key is empty")
        if not self.model.strip():
            raise RuntimeError("language model name is empty")
        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
            "temperature": 0.7,
            "max_tokens": 600,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "limo4si_language_realization",
                    "strict": True,
                    "schema": dict(json_schema),
                },
            },
        }
        response = None
        for attempt in range(self.maximum_transport_attempts):
            try:
                response = self.transport(
                    endpoint,
                    {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    request_body,
                    self.timeout_sec,
                )
                break
            except LanguageTransportError:
                if attempt + 1 >= self.maximum_transport_attempts:
                    raise
                time.sleep(self.retry_delay_sec * (2 ** attempt))
        if response is None:
            raise RuntimeError("language provider produced no response")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("language provider response has no choices")
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("language provider response has no text content")
        parsed = json.loads(content)
        if not isinstance(parsed, Mapping):
            raise RuntimeError("language provider content must decode to a JSON object")
        return parsed


def create_client() -> OpenAICompatibleStructuredClient:
    """Create a client without persisting secrets in code or generated files."""

    api_key = os.environ.get("LIMO4SI_LANGUAGE_API_KEY", "")
    key_file = os.environ.get("LIMO4SI_LANGUAGE_API_KEY_FILE", "")
    if not api_key and key_file:
        api_key = Path(key_file).read_text(encoding="utf-8").strip()
    base_url = os.environ.get("LIMO4SI_LANGUAGE_BASE_URL", "https://starwithcoding.com/v1")
    model = os.environ.get("LIMO4SI_LANGUAGE_MODEL", "")
    if not api_key:
        raise RuntimeError(
            "set LIMO4SI_LANGUAGE_API_KEY or LIMO4SI_LANGUAGE_API_KEY_FILE"
        )
    if not model:
        raise RuntimeError("set LIMO4SI_LANGUAGE_MODEL in the process environment")
    return OpenAICompatibleStructuredClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
