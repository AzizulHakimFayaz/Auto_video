"""AI Story Generation module using OpenRouter's free Nemotron 3 Ultra endpoint.

Implements an OpenAI-compatible client interface, Markdown/JSON extraction,
strict schema and word-count validation, scene-to-story alignment checks,
history deduplication, bounded exponential backoff, Retry-After header handling,
and strict zero-budget free-model enforcement.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

from src.config import AppConfig, ConfigurationError, is_free_model_identifier
from src.history_manager import HistoryManager, compute_story_similarity

logger = logging.getLogger(__name__)

SUPPORTED_THEMES: List[str] = [
    "family",
    "kindness",
    "sacrifice",
    "loneliness",
    "friendship",
    "loss",
    "hope",
    "reunion",
    "unexpected generosity",
]

VALID_VISUAL_MOODS = {
    "sadness",
    "melancholic",
    "love",
    "warm",
    "family",
    "hope",
    "hopeful",
    "uplifting",
    "tragedy",
    "tense",
    "suspenseful",
    "peace",
    "peaceful",
    "calm",
    "nature",
    "city",
    "lonely",
}


class StoryGenerationError(RuntimeError):
    """Base exception for story generation failures."""


class StoryValidationError(StoryGenerationError):
    """Raised when the LLM response fails JSON, schema, word-count, or scene alignment validation."""


class StoryAPIError(StoryGenerationError):
    """Raised on HTTP/API errors from OpenRouter."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
        is_transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.is_transient = is_transient


@dataclass
class SceneSegment:
    """Single narrated scene in the short story."""

    narration: str
    image_query: str
    visual_mood: str
    duration_hint_seconds: float = 7.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StoryPayload:
    """Validated story output structure."""

    title: str
    language: str
    story: str
    mood: str
    caption: str
    hashtags: List[str]
    scenes: List[SceneSegment]
    theme: str = "hope"
    protagonist: str = ""
    setting: str = ""
    model_used: str = ""
    word_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["scenes"] = [s.to_dict() for s in self.scenes]
        return data


class _ChatCompletionsNamespace:
    """OpenAI-compatible `client.chat.completions.create(...)` implementation over HTTP."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 1600,
        **kwargs: Any,
    ) -> Any:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/auto-story-video",
            "X-Title": "Free Daily AI Story Video Automation",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if "response_format" in kwargs and kwargs["response_format"]:
            payload["response_format"] = kwargs["response_format"]

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
        except requests.Timeout as exc:
            raise StoryAPIError(
                f"OpenRouter request timed out after {self.timeout}s.",
                is_transient=True,
            ) from exc
        except requests.RequestException as exc:
            raise StoryAPIError(
                f"Network error communicating with OpenRouter: {type(exc).__name__}",
                is_transient=True,
            ) from exc

        status = response.status_code
        retry_after = _parse_retry_after(response.headers.get("Retry-After"))

        if status == 401:
            raise StoryAPIError(
                "HTTP 401 Unauthorized from OpenRouter: Invalid or expired OPENROUTER_API_KEY.",
                status_code=401,
                is_transient=False,
            )
        if status == 402:
            raise StoryAPIError(
                "HTTP 402 Payment Required: Free quota exhausted or endpoint requires credits. "
                "Refusing paid fallback per zero-budget policy.",
                status_code=402,
                is_transient=False,
            )
        if status == 403:
            raise StoryAPIError(
                "HTTP 403 Forbidden from OpenRouter: Access denied or model restricted.",
                status_code=403,
                is_transient=False,
            )
        if status == 404:
            raise StoryAPIError(
                f"HTTP 404 Not Found: Free model endpoint '{model}' is currently unavailable.",
                status_code=404,
                is_transient=False,
            )
        if status == 429:
            raise StoryAPIError(
                f"HTTP 429 Too Many Requests: Rate limit reached for model '{model}'.",
                status_code=429,
                retry_after=retry_after,
                is_transient=True,
            )
        if status >= 500:
            raise StoryAPIError(
                f"HTTP {status} Server Error from OpenRouter provider.",
                status_code=status,
                retry_after=retry_after,
                is_transient=True,
            )
        if status != 200:
            raise StoryAPIError(
                f"Unexpected HTTP {status} from OpenRouter.",
                status_code=status,
                is_transient=False,
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise StoryAPIError(
                "OpenRouter returned non-JSON HTTP response body.",
                status_code=status,
                is_transient=True,
            ) from exc

        if "error" in body and body["error"]:
            err_obj = body["error"]
            err_code = int(err_obj.get("code", 500)) if isinstance(err_obj, dict) else 500
            err_msg = (
                err_obj.get("message", "Unknown provider error")
                if isinstance(err_obj, dict)
                else str(err_obj)
            )
            is_transient = err_code in {429, 500, 502, 503, 504}
            raise StoryAPIError(
                f"OpenRouter provider error ({err_code}): {err_msg}",
                status_code=err_code,
                retry_after=retry_after,
                is_transient=is_transient,
            )

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise StoryAPIError(
                "OpenRouter response contained no completion choices.",
                status_code=200,
                is_transient=True,
            )

        message_obj = choices[0].get("message", {})
        content = message_obj.get("content") or ""
        return _CompletionResponse(content=content, model=body.get("model", model))


@dataclass
class _MessageObj:
    content: str


@dataclass
class _ChoiceObj:
    message: _MessageObj


class _CompletionResponse:
    def __init__(self, content: str, model: str) -> None:
        self.choices = [_ChoiceObj(message=_MessageObj(content=content))]
        self.model = model


class OpenAICompatibleClient:
    """OpenAI SDK compatible client wrapper.

    Uses the official `openai.OpenAI` client if installed and requested, or falls back
    seamlessly to the built-in OpenAI-compatible REST transport with full status-code
    and Retry-After header inspection.
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 60) -> None:
        self.chat = type(
            "_ChatNamespace",
            (),
            {"completions": _ChatCompletionsNamespace(base_url, api_key, timeout)},
        )()


def _parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    if not header_value:
        return None
    try:
        val = float(header_value.strip())
        return max(0.0, min(val, 120.0))
    except ValueError:
        return None


def count_words(text: str) -> int:
    """Count words in a narration string."""
    words = re.findall(r"\b[\w'-]+\b", text)
    return len(words)


def extract_json_from_text(raw_text: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from raw LLM output, stripping Markdown fences and reasoning."""
    if not raw_text or not raw_text.strip():
        raise StoryValidationError("Model returned an empty response.")

    cleaned = raw_text.strip()

    # Strip complete and unclosed <think>...</think> blocks emitted by reasoning models
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()

    # Check for ```json ... ``` or ``` ... ``` fenced code block (greedy to capture outermost braces)
    fence_match = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        candidate = fence_match.group(1).strip()
    else:
        # Locate outermost JSON object braces
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            snippet = cleaned[:250].replace("\n", " ")
            raise StoryValidationError(
                f"No JSON object found in model output. Model returned: {snippet!r}"
            )
        candidate = cleaned[first_brace : last_brace + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StoryValidationError(f"Malformed JSON in model output: {exc}") from exc

    if not isinstance(parsed, dict):
        raise StoryValidationError("Top-level JSON value must be an object.")
    return parsed


def validate_story_payload(
    raw_dict: Dict[str, Any],
    min_words: int = 100,
    max_words: int = 130,
    min_scenes: int = 4,
    max_scenes: int = 8,
    expected_language: str = "English",
    model_used: str = "",
    theme: str = "hope",
) -> StoryPayload:
    """Validate structured JSON story output against all content and structural rules."""
    required_keys = {"title", "language", "story", "mood", "caption", "hashtags", "scenes"}
    missing = required_keys - set(raw_dict.keys())
    if missing:
        raise StoryValidationError(
            f"Story JSON is missing required fields: {sorted(missing)}"
        )

    title = str(raw_dict.get("title") or "").strip()
    if not title or len(title) < 3 or len(title) > 100:
        raise StoryValidationError(
            f"Invalid story title {title!r} (must be 3-100 characters)."
        )

    language = str(raw_dict.get("language") or expected_language).strip()
    story_text = str(raw_dict.get("story") or "").strip()
    mood = str(raw_dict.get("mood") or "").strip()
    caption = str(raw_dict.get("caption") or "").strip()

    if not story_text:
        raise StoryValidationError("Field 'story' cannot be empty.")
    if not mood:
        raise StoryValidationError("Field 'mood' cannot be empty.")
    if not caption:
        raise StoryValidationError("Field 'caption' cannot be empty.")

    raw_hashtags = raw_dict.get("hashtags")
    if not isinstance(raw_hashtags, list) or not raw_hashtags:
        raise StoryValidationError("Field 'hashtags' must be a non-empty list of strings.")

    hashtags: List[str] = []
    for tag in raw_hashtags:
        tag_str = str(tag).strip()
        if not tag_str:
            continue
        if not tag_str.startswith("#"):
            tag_str = f"#{tag_str}"
        hashtags.append(tag_str)
    if not hashtags:
        raise StoryValidationError("At least one valid hashtag is required.")

    raw_scenes = raw_dict.get("scenes")
    if not isinstance(raw_scenes, list) or len(raw_scenes) < min_scenes:
        raise StoryValidationError(
            f"Field 'scenes' must contain at least {min_scenes} scene objects "
            f"(got {len(raw_scenes) if isinstance(raw_scenes, list) else 0})."
        )
    if len(raw_scenes) > max_scenes:
        raise StoryValidationError(
            f"Field 'scenes' exceeds maximum allowed scenes ({len(raw_scenes)} > {max_scenes})."
        )

    scenes: List[SceneSegment] = []
    for idx, scene_obj in enumerate(raw_scenes):
        if not isinstance(scene_obj, dict):
            raise StoryValidationError(f"Scene at index {idx} must be a JSON object.")
        narration = str(scene_obj.get("narration") or "").strip()
        image_query = str(scene_obj.get("image_query") or "").strip()
        visual_mood = str(scene_obj.get("visual_mood") or "").strip().lower()
        duration_hint = scene_obj.get("duration_hint_seconds", 7)

        if not narration:
            raise StoryValidationError(f"Scene {idx + 1} is missing 'narration'.")
        if not image_query:
            raise StoryValidationError(f"Scene {idx + 1} is missing 'image_query'.")
        if not visual_mood:
            raise StoryValidationError(f"Scene {idx + 1} is missing 'visual_mood'.")
        try:
            duration_val = float(duration_hint)
        except (TypeError, ValueError) as exc:
            raise StoryValidationError(
                f"Scene {idx + 1} has non-numeric 'duration_hint_seconds': {duration_hint!r}"
            ) from exc
        if duration_val <= 1.0 or duration_val > 30.0:
            duration_val = max(3.0, min(duration_val, 20.0))

        scenes.append(
            SceneSegment(
                narration=narration,
                image_query=image_query,
                visual_mood=visual_mood,
                duration_hint_seconds=duration_val,
            )
        )

    # Validate word count of full story
    total_words = count_words(story_text)
    if total_words < min_words or total_words > max_words:
        raise StoryValidationError(
            f"Story word count ({total_words}) is outside configured bounds "
            f"[{min_words}, {max_words}]."
        )

    # Validate that scene narrations correspond to and cover the complete story text
    combined_scene_text = " ".join(s.narration for s in scenes)
    scene_words = count_words(combined_scene_text)
    alignment_score = compute_story_similarity(story_text, combined_scene_text)
    if alignment_score < 0.80 or abs(scene_words - total_words) > max(15, int(total_words * 0.15)):
        raise StoryValidationError(
            f"Scene narrations do not accurately cover the story text "
            f"(alignment={alignment_score:.2f}, story_words={total_words}, scene_words={scene_words})."
        )

    return StoryPayload(
        title=title,
        language=language,
        story=story_text,
        mood=mood,
        caption=caption,
        hashtags=hashtags,
        scenes=scenes,
        theme=str(raw_dict.get("theme") or theme),
        protagonist=str(raw_dict.get("protagonist") or ""),
        setting=str(raw_dict.get("setting") or ""),
        model_used=model_used,
        word_count=total_words,
    )


class StoryGenerator:
    """Generates and validates original daily emotional short stories via OpenRouter."""

    def __init__(
        self,
        config: AppConfig,
        history_manager: Optional[HistoryManager] = None,
        client: Optional[Any] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.history_manager = history_manager
        self.sleep_fn = sleep_fn
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        return OpenAICompatibleClient(
            base_url=self.config.openrouter_base_url,
            api_key=self.config.openrouter_api_key,
            timeout=self.config.openrouter_timeout_seconds,
        )

    def _build_messages(
        self,
        theme: str,
        excluded_titles: List[str],
        attempt: int = 1,
        feedback: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        min_w = self.config.story_min_words
        max_w = self.config.story_max_words
        min_s = self.config.story_min_scenes
        max_s = self.config.story_max_scenes
        lang = self.config.story_language

        avoid_clause = ""
        if excluded_titles:
            joined = ", ".join(f'"{t}"' for t in excluded_titles[-15:])
            avoid_clause = f"\nDo NOT reuse or closely imitate any of these recent titles: {joined}."

        feedback_clause = ""
        if feedback:
            feedback_clause = (
                f"\nIMPORTANT CORRECTION FROM PREVIOUS ATTEMPT: {feedback} "
                f"Ensure the total word count of 'story' is strictly between {min_w} and {max_w} words, "
                "and that concatenating all 'scenes[].narration' strings matches 'story' word-for-word."
            )

        system_prompt = (
            "You are an award-winning short-form emotional story writer for vertical video narration. "
            "Write completely original, deeply moving fictional stories with human warmth. "
            "Never copy existing stories, famous IP, or recognizable characters. "
            "Never present sensational misinformation or fake real-world news claims as facts. "
            "CRITICAL: Do NOT output any thinking process, reasoning, internal monologue, or markdown chatter. "
            "Respond ONLY with a single raw valid JSON object starting with '{' and ending with '}'."
        )

        user_prompt = f"""Create an original emotional short story in {lang} centered on the theme of "{theme}" (variation #{attempt}).

Story & Structural Requirements:
1. Opening hook: Start with an immediate, compelling sentence in the first few seconds.
2. Clear protagonist and vivid setting.
3. Relatable emotional conflict and progressive storytelling.
4. Meaningful emotional turning point and memorable ending.
5. Word count: The complete "story" field MUST be between {min_w} and {max_w} words.
6. Scene breakdown: Split the exact story into {min_s} to {max_s} chronological scenes.
   Concatenating all `scenes[i].narration` texts in order MUST equal the complete `story` text word-for-word.{avoid_clause}{feedback_clause}

Return ONLY valid JSON with this exact structure:
{{
  "title": "Short evocative title",
  "language": "{lang}",
  "protagonist": "Brief protagonist description",
  "setting": "Brief setting description",
  "theme": "{theme}",
  "story": "The complete narration text ({min_w}-{max_w} words)...",
  "mood": "sad then hopeful",
  "caption": "A short emotional social media description...",
  "hashtags": ["#shortstory", "#emotionalstory", "#kindness", "#lifelessons"],
  "scenes": [
    {{
      "narration": "Exact opening sentence(s) from the story.",
      "image_query": "3 to 6 descriptive English keywords for stock photography (e.g. old house rainy village)",
      "visual_mood": "sadness",
      "duration_hint_seconds": 8
    }}
  ]
}}"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_story(self, preferred_theme: Optional[str] = None) -> StoryPayload:
        """Generate and validate a unique emotional story using configured free model(s)."""
        self.config.validate_core_settings(require_api_key=True)

        models_to_try: List[str] = []
        if self.config.openrouter_model:
            models_to_try.append(self.config.openrouter_model)

        if self.config.openrouter_fallback_model:
            for fallback in self.config.openrouter_fallback_model.split(","):
                fallback = fallback.strip()
                if fallback and fallback not in models_to_try:
                    models_to_try.append(fallback)

        excluded_titles: List[str] = []
        if self.history_manager is not None:
            excluded_titles.extend(self.history_manager.get_recent_titles(limit=30))

        last_error: Optional[Exception] = None

        for model_id in models_to_try:
            if not is_free_model_identifier(model_id):
                raise ConfigurationError(
                    f"Refusing to call non-free model {model_id!r}. Only free endpoints are permitted."
                )

            logger.info("Starting story generation using OpenRouter model: %s", model_id)
            feedback: Optional[str] = None

            for attempt in range(1, self.config.openrouter_max_retries + 1):
                theme = preferred_theme or SUPPORTED_THEMES[
                    (attempt - 1 + random.randint(0, len(SUPPORTED_THEMES) - 1))
                    % len(SUPPORTED_THEMES)
                ]
                messages = self._build_messages(
                    theme=theme,
                    excluded_titles=excluded_titles,
                    attempt=attempt,
                    feedback=feedback,
                )
                temperature = min(1.0, 0.75 + (attempt - 1) * 0.08)

                try:
                    client = self._get_client()
                    completion = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=3500,
                    )
                    raw_content = (
                        completion.choices[0].message.content
                        if completion and getattr(completion, "choices", None)
                        else ""
                    )

                    parsed_json = extract_json_from_text(raw_content)
                    payload = validate_story_payload(
                        parsed_json,
                        min_words=self.config.story_min_words,
                        max_words=self.config.story_max_words,
                        min_scenes=self.config.story_min_scenes,
                        max_scenes=self.config.story_max_scenes,
                        expected_language=self.config.story_language,
                        model_used=model_id,
                        theme=theme,
                    )

                    if self.history_manager is not None:
                        rejection = self.history_manager.check_story_uniqueness(
                            payload.title, payload.story
                        )
                        if rejection:
                            excluded_titles.append(payload.title)
                            raise StoryValidationError(rejection)

                    logger.info(
                        "Successfully generated story %r (%d words, %d scenes, mood=%r).",
                        payload.title,
                        payload.word_count,
                        len(payload.scenes),
                        payload.mood,
                    )
                    return payload

                except StoryValidationError as exc:
                    last_error = exc
                    feedback = str(exc)
                    logger.warning(
                        "Story validation failed on attempt %d/%d for model %s: %s",
                        attempt,
                        self.config.openrouter_max_retries,
                        model_id,
                        exc,
                    )
                    if attempt < self.config.openrouter_max_retries:
                        backoff = min(15.0, float(2 ** (attempt - 1)))
                        self.sleep_fn(backoff)

                except StoryAPIError as exc:
                    last_error = exc
                    logger.warning(
                        "OpenRouter API error on attempt %d/%d for model %s (status=%s): %s",
                        attempt,
                        self.config.openrouter_max_retries,
                        model_id,
                        exc.status_code,
                        exc,
                    )
                    # Non-transient auth/quota errors on API key should fail or switch to fallback if 404
                    if exc.status_code in {401, 402}:
                        raise
                    if not exc.is_transient:
                        break
                    if attempt < self.config.openrouter_max_retries:
                        wait_seconds = (
                            exc.retry_after
                            if exc.retry_after is not None
                            else min(30.0, float(2**attempt))
                        )
                        logger.info(
                            "Backing off for %.1f seconds before retry...", wait_seconds
                        )
                        self.sleep_fn(wait_seconds)

                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "Unexpected error during story generation attempt %d/%d: %s",
                        attempt,
                        self.config.openrouter_max_retries,
                        exc,
                    )
                    if attempt < self.config.openrouter_max_retries:
                        self.sleep_fn(float(2 ** (attempt - 1)))

            if len(models_to_try) > 1 and model_id != models_to_try[-1]:
                logger.warning(
                    "Primary free model %s failed; trying configured free fallback model.",
                    model_id,
                )

        raise StoryGenerationError(
            f"All story generation attempts exhausted across configured free models. Last error: {last_error}"
        )
