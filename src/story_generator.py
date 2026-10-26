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
    "betrayal and broken trust",
    "cheating and hidden messages",
    "uncovering painful secrets",
    "heartbreak and walking away",
    "lies and double life",
    "silent revenge and karma",
    "forbidden love and deceit",
    "regret after betrayal",
]

VALID_VISUAL_MOODS = {
    "betrayal",
    "cheating",
    "heartbreak",
    "drama",
    "dark",
    "shock",
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
    pause_after: float = 1.0

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
    theme: str = "betrayal"
    music_style: str = "tense emotional dark cello and sad piano with rising dramatic suspense"
    music_mood: str = "betrayal"
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


PAUSE_MARKER_REGEX = re.compile(r"\[pause(?::\s*[0-9.]+\s*s?)?\]", re.IGNORECASE)


def strip_pause_markers(text: str) -> str:
    """Strip [pause: Xs] or [pause] tags from text for word count and plain text comparison."""
    if not text:
        return ""
    stripped = PAUSE_MARKER_REGEX.sub("", text)
    return re.sub(r"\s+", " ", stripped).strip()


def count_words(text: str) -> int:
    """Count spoken words in a narration string, excluding pause tags."""
    cleaned = strip_pause_markers(text)
    words = re.findall(r"\b[\w'-]+\b", cleaned)
    return len(words)


def repair_json_string(text: str) -> str:
    """Repair common LLM JSON syntax mistakes before decoding."""
    if not text:
        return ""
    # 1. Normalize unicode quotation marks and smart quotes
    s = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    # 2. Fix missing commas between objects in an array: } \s* { -> },\n{
    s = re.sub(r"\}\s*\{", "},\n{", s)

    # 3. Fix missing commas between array elements: ] \s* [ -> ],\n[
    s = re.sub(r"\]\s*\[", "],\n[", s)

    # 4. Fix missing commas between string items in arrays: "item1" \s* "item2" -> "item1",\n"item2"
    s = re.sub(r'(["\d])\s*\n\s*(")', r"\1,\n\2", s)

    # 5. Fix missing commas between key-value pairs where next line starts with a key
    s = re.sub(
        r'("|\b(?:true|false|null|\d+(?:\.\d+)?))\s*\n\s*("[a-zA-Z0-9_]+"\s*:)',
        r"\1,\n\2",
        s,
    )

    # 6. Remove trailing commas before } or ]
    s = re.sub(r",\s*([\}\]])", r"\1", s)

    return s


def extract_fields_fallback(raw: str) -> Dict[str, Any]:
    """Extract fields using regex when JSON has syntax errors that json.loads cannot parse."""
    res: Dict[str, Any] = {}

    m_title = re.search(r'"title"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw)
    if m_title:
        try:
            res["title"] = (
                m_title.group(1).encode("utf-8").decode("unicode_escape", errors="ignore")
            )
        except Exception:
            res["title"] = m_title.group(1)

    m_lang = re.search(r'"language"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw)
    res["language"] = m_lang.group(1) if m_lang else "English"

    m_story = re.search(
        r'"story"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"(?:mood|scenes|caption|music_style)',
        raw,
        re.DOTALL,
    )
    if not m_story:
        m_story = re.search(r'"story"\s*:\s*"((?:[^"]|"(?!,\s*"\w+"))*)"', raw, re.DOTALL)
    if m_story:
        res["story"] = m_story.group(1).replace('\\"', '"').replace("\\n", " ")

    m_mood = re.search(r'"mood"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw)
    res["mood"] = m_mood.group(1) if m_mood else "betrayal"

    m_cap = re.search(r'"caption"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw)
    res["caption"] = m_cap.group(1) if m_cap else "Watch till the end to see the truth."

    m_music_style = re.search(r'"music_style"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw)
    if m_music_style:
        res["music_style"] = m_music_style.group(1)

    m_music_mood = re.search(r'"music_mood"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw)
    if m_music_mood:
        res["music_mood"] = m_music_mood.group(1)

    m_theme = re.search(r'"theme"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw)
    if m_theme:
        res["theme"] = m_theme.group(1)

    m_tags = re.search(r'"hashtags"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if m_tags:
        tags = re.findall(r'"(#[^"]+)"', m_tags.group(1))
        res["hashtags"] = tags or ["#betrayal", "#storytime", "#drama"]
    else:
        res["hashtags"] = ["#betrayal", "#storytime", "#drama"]

    # Extract scenes
    scenes: List[Dict[str, Any]] = []
    scene_matches = re.findall(r'\{\s*"narration".*?\}', raw, re.DOTALL)
    for sm in scene_matches:
        narr = re.search(r'"narration"\s*:\s*"((?:[^"\\]|\\.)*)"', sm)
        query = re.search(r'"image_query"\s*:\s*"([^"]*)"', sm)
        vmood = re.search(r'"visual_mood"\s*:\s*"([^"]*)"', sm)
        pause = re.search(r'"pause_after"\s*:\s*([0-9.]+)', sm)
        dur = re.search(r'"duration_hint_seconds"\s*:\s*([0-9.]+)', sm)
        if narr:
            scenes.append(
                {
                    "narration": narr.group(1).replace('\\"', '"'),
                    "image_query": query.group(1) if query else "emotional drama cinematic",
                    "visual_mood": vmood.group(1) if vmood else "betrayal",
                    "pause_after": float(pause.group(1)) if pause else 1.0,
                    "duration_hint_seconds": float(dur.group(1)) if dur else 7.0,
                }
            )
    if scenes:
        res["scenes"] = scenes

    return res


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
            salvaged = extract_fields_fallback(cleaned)
            if salvaged.get("story") and salvaged.get("scenes"):
                return salvaged
            snippet = cleaned[:250].replace("\n", " ")
            raise StoryValidationError(
                f"No JSON object found in model output. Model returned: {snippet!r}"
            )
        candidate = cleaned[first_brace : last_brace + 1]

    # 1. Attempt direct JSON parse
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Attempt repaired JSON parse
    repaired = repair_json_string(candidate)
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3. Attempt closing unclosed braces/brackets if truncated
    try:
        open_curly = repaired.count("{") - repaired.count("}")
        open_square = repaired.count("[") - repaired.count("]")
        fixed = repaired
        fixed = re.sub(r",\s*$", "", fixed.strip())
        fixed = re.sub(r'"[^"]*"\s*:\s*$', "", fixed.strip())
        fixed += "]" * max(0, open_square)
        fixed += "}" * max(0, open_curly)
        parsed = json.loads(repair_json_string(fixed))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 4. Fallback regex field extraction
    salvaged = extract_fields_fallback(candidate)
    if salvaged.get("title") and salvaged.get("story") and salvaged.get("scenes"):
        return salvaged

    # 5. If everything failed, raise descriptive error
    try:
        json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StoryValidationError(f"Malformed JSON in model output: {exc}") from exc

    raise StoryValidationError("Top-level JSON value must be an object.")


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
        pause_after_raw = scene_obj.get("pause_after", 1.0)

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

        try:
            pause_after_val = max(0.2, min(3.0, float(pause_after_raw)))
        except (TypeError, ValueError):
            pause_after_val = 1.0

        scenes.append(
            SceneSegment(
                narration=narration,
                image_query=image_query,
                visual_mood=visual_mood,
                duration_hint_seconds=duration_val,
                pause_after=pause_after_val,
            )
        )

    # Validate word count of full story (ignoring pause tags)
    total_words = count_words(story_text)
    # Allow a reasonable tolerance window so minor variations (e.g. 149 words against 150)
    # don't fail an otherwise high quality generation, while still ensuring a 1+ minute video.
    under_tolerance = min(30, max(5, int(min_words * 0.15)))
    over_tolerance = min(35, max(10, int(max_words * 0.20)))
    effective_min = max(90, min_words - under_tolerance)
    effective_max = max_words + over_tolerance

    if total_words < effective_min or total_words > effective_max:
        raise StoryValidationError(
            f"Story word count ({total_words}) is outside allowed bounds "
            f"[{effective_min}, {effective_max}] (target was {min_words}-{max_words})."
        )

    # Validate that scene narrations correspond to and cover the complete story text
    clean_story = strip_pause_markers(story_text)
    combined_scene_text = " ".join(strip_pause_markers(s.narration) for s in scenes)
    scene_words = count_words(combined_scene_text)
    alignment_score = compute_story_similarity(clean_story, combined_scene_text)
    if alignment_score < 0.78 or abs(scene_words - total_words) > max(20, int(total_words * 0.18)):
        raise StoryValidationError(
            f"Scene narrations do not accurately cover the story text "
            f"(alignment={alignment_score:.2f}, story_words={total_words}, scene_words={scene_words})."
        )

    music_style = str(
        raw_dict.get("music_style")
        or "tense emotional dark cello and sad piano with rising dramatic suspense"
    ).strip()
    music_mood = str(raw_dict.get("music_mood") or mood or "betrayal").strip().lower()

    return StoryPayload(
        title=title,
        language=language,
        story=story_text,
        mood=mood,
        caption=caption,
        hashtags=hashtags,
        scenes=scenes,
        theme=str(raw_dict.get("theme") or theme),
        music_style=music_style,
        music_mood=music_mood,
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
            "You are a master drama and emotional relationship storyteller specializing in viral vertical video scripts. "
            "Write gripping, realistic fictional stories centered on betrayal, cheating, hidden secrets, heartbreak, and karma. "
            "Deliver intense emotional suspense, shocking discoveries, human vulnerability, and unforgettable closure. "
            "Never copy existing IP or famous celebrities. "
            "CRITICAL: Do NOT output any thinking process, reasoning, internal monologue, or markdown chatter. "
            "Respond ONLY with a single raw valid JSON object starting with '{' and ending with '}'."
        )

        user_prompt = f"""Create a gripping emotional drama short story in {lang} centered on "{theme}" (variation #{attempt}).

Dramatic & Video Pacing Requirements:
1. Opening Hook: The first sentence MUST immediately seize the audience with a revelation, an alarming text message, an unexpected discovery, or a confrontation.
2. Narrative Arc: High stakes emotional conflict. Build the tension piece by piece as the truth comes to light.
3. Turning Point: An intense confrontation or moment of quiet realization where the protagonist takes back their dignity or walks away.
4. Video Length (1 minute or more): The complete "story" field MUST be between {min_w} and {max_w} words.
5. Dramatic Pauses: Insert [pause: 0.8s] or [pause: 1.2s] at 2 to 4 crucial moments of shock or suspense in the narration (e.g. "I checked the name on the second phone. [pause: 1.0s] It was my best friend.").
6. Scene breakdown: Split the story into {min_s} to {max_s} chronological scenes.
   Each scene must include:
   - "narration": The spoken text for this scene (including any [pause: Xs] markers). Concatenating all scene narrations in order must equal the complete story text.
   - "pause_after": Float between 0.6 and 1.5 seconds representing the dramatic silence after this scene before transitioning.
   - "image_query": 3 to 6 descriptive English keywords for atmospheric vertical imagery (e.g. phone lock screen late night, crying woman rain window, empty dining table candlelight).
   - "visual_mood": One of: betrayal, heartbreak, tension, dark, sadness, drama.
7. Background Music Direction:
   - "music_style": Detailed description of the desired background music (e.g. "tense emotional dark cello and sad piano with rising dramatic suspense").
   - "music_mood": One of: betrayal, heartbreak, suspense, drama, sad.
8. JSON Formatting Requirements:
   - Output valid JSON only, starting with {{ and ending with }}.
   - When quoting dialogue inside narration strings, ALWAYS use single quotes (e.g. He whispered, 'Why did you do it?') to prevent broken JSON syntax.
   - Separate all properties and array items with commas. Never leave trailing commas.{avoid_clause}{feedback_clause}

Return ONLY valid JSON with this exact structure:
{{
  "title": "Short dramatic title (e.g. The Second Key, Messages at 2 AM)",
  "language": "{lang}",
  "protagonist": "Brief protagonist description",
  "setting": "Brief setting description",
  "theme": "{theme}",
  "story": "The complete narration text with [pause: Xs] markers ({min_w}-{max_w} words)...",
  "mood": "betrayal and heartbreak",
  "music_style": "tense emotional dark cello and sad piano with rising dramatic suspense",
  "music_mood": "betrayal",
  "caption": "A dramatic viral caption ending with a question for viewers...",
  "hashtags": ["#betrayal", "#cheating", "#storytime", "#drama", "#heartbreak", "#relationship"],
  "scenes": [
    {{
      "narration": "First scene narration text with optional [pause: 0.8s]...",
      "pause_after": 1.0,
      "image_query": "unlocked smartphone notifications dark bedroom",
      "visual_mood": "betrayal",
      "duration_hint_seconds": 12
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
