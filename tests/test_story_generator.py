"""Unit tests for configuration, secret redaction, and AI story generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.config import AppConfig, ConfigurationError, SecretRedactingFilter, setup_logging
from src.history_manager import HistoryManager
from src.story_generator import (
    StoryAPIError,
    StoryGenerationError,
    StoryGenerator,
    StoryValidationError,
    _CompletionResponse,
    extract_json_from_text,
    validate_story_payload,
)


def _sample_valid_story_dict(title: str = "The Last Umbrella on Platform Four") -> Dict[str, Any]:
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_story.json"
    data = json.loads(sample_path.read_text(encoding="utf-8"))
    data["title"] = title
    return data


class FakeCompletions:
    def __init__(self, responses: List[Any]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            raise StoryAPIError("No more fake responses configured", status_code=500, is_transient=True)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _CompletionResponse(content=item, model=kwargs.get("model", "test:free"))


class FakeClient:
    def __init__(self, responses: List[Any]) -> None:
        self.completions = FakeCompletions(responses)
        self.chat = type("_Chat", (), {"completions": self.completions})()


def test_extract_json_handles_markdown_code_fences_and_think_tags() -> None:
    payload = _sample_valid_story_dict()
    fenced = f"<think>Reasoning about story...</think>\n```json\n{json.dumps(payload, indent=2)}\n```\n"
    parsed = extract_json_from_text(fenced)
    assert parsed["title"] == payload["title"]
    assert len(parsed["scenes"]) == 6


def test_extract_json_rejects_malformed_json() -> None:
    with pytest.raises(StoryValidationError, match="Malformed JSON"):
        extract_json_from_text("```json\n{\"title\": \"Broken\", }\n```")

    with pytest.raises(StoryValidationError, match="No JSON object"):
        extract_json_from_text("Plain text with no braces at all")

    with pytest.raises(StoryValidationError, match="empty response"):
        extract_json_from_text("   ")


def test_validate_story_payload_accepts_valid_story() -> None:
    raw = _sample_valid_story_dict()
    validated = validate_story_payload(raw, min_words=100, max_words=130)
    assert validated.title == "The Last Umbrella on Platform Four"
    assert 100 <= validated.word_count <= 130
    assert len(validated.scenes) == 6


def test_validate_story_payload_rejects_out_of_bounds_word_count() -> None:
    raw = _sample_valid_story_dict()
    raw["story"] = "Too short story."
    raw["scenes"][0]["narration"] = "Too short story."
    with pytest.raises(StoryValidationError, match="word count"):
        validate_story_payload(raw, min_words=100, max_words=130)


def test_validate_story_payload_rejects_scene_narration_mismatch() -> None:
    raw = _sample_valid_story_dict()
    for scene in raw["scenes"]:
        scene["narration"] = "Completely unrelated words that do not appear in the story text."
    with pytest.raises(StoryValidationError, match="do not accurately cover"):
        validate_story_payload(raw, min_words=100, max_words=130)


def test_story_generator_retries_rate_limit_and_succeeds(tmp_path: Path) -> None:
    cfg = AppConfig(
        openrouter_api_key="sk-or-v1-secretkey123456",
        openrouter_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        openrouter_max_retries=3,
        history_file=tmp_path / "history.json",
    )
    valid_json_str = json.dumps(_sample_valid_story_dict())
    fake_client = FakeClient(
        [
            StoryAPIError(
                "HTTP 429 rate limited",
                status_code=429,
                retry_after=1.5,
                is_transient=True,
            ),
            valid_json_str,
        ]
    )
    slept: List[float] = []
    gen = StoryGenerator(
        config=cfg,
        history_manager=HistoryManager(cfg.history_file),
        client=fake_client,
        sleep_fn=slept.append,
    )
    result = gen.generate_story()
    assert result.title == "The Last Umbrella on Platform Four"
    assert slept == [1.5]
    assert len(fake_client.completions.calls) == 2


def test_story_generator_uses_configured_free_fallback_model(tmp_path: Path) -> None:
    cfg = AppConfig(
        openrouter_api_key="sk-or-v1-secretkey123456",
        openrouter_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        openrouter_fallback_model="meta-llama/llama-3.3-70b-instruct:free",
        openrouter_max_retries=2,
        history_file=tmp_path / "history.json",
    )
    valid_json_str = json.dumps(_sample_valid_story_dict("Fallback Free Model Story"))
    fake_client = FakeClient(
        [
            StoryAPIError("Model offline", status_code=404, is_transient=False),
            valid_json_str,
        ]
    )
    gen = StoryGenerator(config=cfg, client=fake_client, sleep_fn=lambda _: None)
    result = gen.generate_story()
    assert result.title == "Fallback Free Model Story"
    assert result.model_used == "meta-llama/llama-3.3-70b-instruct:free"


def test_zero_budget_policy_rejects_paid_model() -> None:
    cfg = AppConfig(
        openrouter_api_key="sk-or-v1-secretkey123456",
        openrouter_model="openai/gpt-4o",  # Paid model without :free suffix
    )
    with pytest.raises(ConfigurationError, match="Zero-budget safeguard violation"):
        cfg.validate_core_settings()


def test_logging_never_leaks_api_keys_or_tokens(caplog: pytest.LogCaptureFixture) -> None:
    secret_key = "sk-or-v1-supersecret999888777"
    yt_secret = "GOCSPX-my_youtube_secret_token"
    cfg = AppConfig(
        openrouter_api_key=secret_key,
        youtube_client_secret=yt_secret,
    )
    logger = setup_logging(config=cfg)
    redactor = SecretRedactingFilter(secrets=cfg.get_sensitive_values())

    raw_msg = f"Failed request with key={secret_key} and Authorization: Bearer {yt_secret}"
    redacted_msg = redactor.redact(raw_msg)
    assert secret_key not in redacted_msg
    assert yt_secret not in redacted_msg
    assert "[REDACTED]" in redacted_msg
