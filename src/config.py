"""Centralized configuration and secret-safe logging setup.

All environment variables, paths, model identifiers, and platform credentials
are defined and validated in this single module.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"Expected integer configuration value, got: {value!r}") from exc


def _parse_float(value: Optional[str], default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ValueError(f"Expected float configuration value, got: {value!r}") from exc


class ConfigurationError(ValueError):
    """Raised when configuration or required credentials are invalid or missing."""


class SecretRedactingFilter(logging.Filter):
    """Logging filter that scrubs configured secrets and bearer/API tokens from log records."""

    TOKEN_PATTERNS = [
        re.compile(r"(Bearer\s+)[A-Za-z0-9._\-~+/]+=*", re.IGNORECASE),
        re.compile(r"(OAuth\s+)[A-Za-z0-9._\-~+/]+=*", re.IGNORECASE),
        re.compile(r"(sk-or-v1-[A-Za-z0-9]+)", re.IGNORECASE),
        re.compile(r"(access_token=)[^&\s\"']+", re.IGNORECASE),
        re.compile(r"(client_secret=)[^&\s\"']+", re.IGNORECASE),
        re.compile(r"(refresh_token=)[^&\s\"']+", re.IGNORECASE),
        re.compile(r"(key=)[^&\s\"']+", re.IGNORECASE),
    ]

    def __init__(self, secrets: Optional[List[str]] = None) -> None:
        super().__init__()
        self._secrets = [s for s in (secrets or []) if s and len(s.strip()) >= 4]

    def add_secret(self, secret: Optional[str]) -> None:
        if secret and len(secret.strip()) >= 4 and secret not in self._secrets:
            self._secrets.append(secret)

    def redact(self, text: str) -> str:
        if not text:
            return text
        redacted = text
        for secret in self._secrets:
            if secret in redacted:
                redacted = redacted.replace(secret, "[REDACTED]")
        for pattern in self.TOKEN_PATTERNS:
            if pattern.groups:
                redacted = pattern.sub(
                    lambda m: (
                        f"{m.group(1)}[REDACTED]"
                        if not m.group(1).lower().startswith("sk-or-")
                        else "[REDACTED]"
                    ),
                    redacted,
                )
        return redacted

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self.redact(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self.redact(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )
        return True


def is_free_model_identifier(model_name: str) -> bool:
    """Return True if model_name represents a verified zero-cost free endpoint."""
    cleaned = (model_name or "").strip().lower()
    return (
        cleaned.endswith(":free")
        or cleaned.endswith("/free")
        or cleaned == "openrouter/free"
    )


@dataclass
class AppConfig:
    """Single source of truth for application settings and credentials."""

    # Base paths
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    image_directory: Path = field(default_factory=lambda: PROJECT_ROOT / "assets" / "backgrounds")
    music_directory: Path = field(default_factory=lambda: PROJECT_ROOT / "assets" / "music")
    font_directory: Path = field(default_factory=lambda: PROJECT_ROOT / "assets" / "fonts")
    license_directory: Path = field(default_factory=lambda: PROJECT_ROOT / "assets" / "licenses")
    output_directory: Path = field(default_factory=lambda: PROJECT_ROOT / "output")
    history_file: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "history.json")

    # OpenRouter / AI Story Generation (Strictly Free Models Only)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openrouter/free"
    openrouter_fallback_model: str = (
        "google/gemma-2-9b-it:free,meta-llama/llama-3.1-8b-instruct:free,"
        "mistralai/mistral-7b-instruct:free,nvidia/nemotron-3-ultra-550b-a55b:free"
    )
    openrouter_timeout_seconds: int = 60
    openrouter_max_retries: int = 3

    # Story configuration
    story_language: str = "English"
    story_min_words: int = 150
    story_max_words: int = 190
    story_min_scenes: int = 5
    story_max_scenes: int = 8
    story_similarity_threshold: float = 0.65

    # Image providers
    pixabay_api_key: str = ""
    pixabay_base_url: str = "https://pixabay.com/api/"
    openverse_enabled: bool = False
    openverse_base_url: str = "https://api.openverse.org/v1/images/"
    openverse_client_id: str = ""
    openverse_client_secret: str = ""
    image_download_timeout_seconds: int = 20
    image_max_bytes: int = 15 * 1024 * 1024  # 15 MB

    # Piper Local TTS
    piper_executable: str = "piper"
    piper_voice: str = "en_US-ljspeech-medium"
    piper_model_path: Optional[Path] = None
    piper_config_path: Optional[Path] = None
    piper_speaker_id: Optional[int] = None
    piper_length_scale: float = 1.05  # Speech speed (1.05 = suspenseful deliberate pace)
    piper_volume: float = 1.0
    piper_sentence_silence: float = 0.55  # Breath pause between sentences
    allow_Offline_tts_fallback: bool = False

    # Audio & Background Music
    allow_narration_only_fallback: bool = True
    music_volume: float = 0.28  # Rich, audible cinematic presence
    music_fade_in_seconds: float = 1.5
    music_fade_out_seconds: float = 2.5
    enable_audio_ducking: bool = True
    internet_archive_music_enabled: bool = True
    freesound_api_key: str = ""

    # Video rendering
    video_width: int = 1080
    video_height: int = 1920
    video_fps: int = 30
    crossfade_duration_seconds: float = 0.5
    title_overlay_seconds: float = 3.5
    dark_overlay_opacity: float = 0.28
    subtitle_font_name: str = "Arial"
    subtitle_font_size: int = 16
    subtitle_margin_v: int = 65
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    ffmpeg_preset: str = "medium"
    ffmpeg_crf: int = 21
    keep_temp_files: bool = False

    # Execution & Publishing Flags
    dry_run: bool = True
    enable_youtube: bool = False
    enable_instagram: bool = False
    enable_facebook: bool = False
    enable_tiktok: bool = False
    log_level: str = "INFO"

    # YouTube Official Data API v3 Credentials
    youtube_client_id: str = ""
    youtube_client_secret: str = ""
    youtube_refresh_token: str = ""
    youtube_privacy_status: str = "public"  # public, unlisted, private
    youtube_category_id: str = "24"  # Entertainment

    # Instagram Graph API Credentials (Eligible Professional Business/Creator Account)
    instagram_account_id: str = ""
    instagram_access_token: str = ""
    instagram_video_public_url_base: str = ""
    instagram_graph_version: str = "v22.0"

    # Facebook Page Graph API Credentials
    facebook_page_id: str = ""
    facebook_page_access_token: str = ""
    facebook_graph_version: str = "v22.0"

    # TikTok Official Content Posting API v2 Credentials
    tiktok_access_token: str = ""
    tiktok_privacy_level: str = "SELF_ONLY"  # SELF_ONLY, MUTUAL_FOLLOW_FRIENDS, PUBLIC_TO_EVERYONE

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None, override: bool = False) -> "AppConfig":
        """Load configuration from environment variables and optional .env file."""
        if env_file is not None:
            load_dotenv(dotenv_path=env_file, override=override)
        else:
            default_env = PROJECT_ROOT / ".env"
            if default_env.exists():
                load_dotenv(dotenv_path=default_env, override=override)

        def _resolve_path(env_key: str, default_rel: str) -> Path:
            raw = os.getenv(env_key, "").strip()
            if not raw:
                return PROJECT_ROOT / default_rel
            p = Path(raw)
            return p if p.is_absolute() else (PROJECT_ROOT / p)

        def _resolve_optional_path(env_key: str) -> Optional[Path]:
            raw = os.getenv(env_key, "").strip()
            if not raw:
                return None
            p = Path(raw)
            return p if p.is_absolute() else (PROJECT_ROOT / p)

        speaker_raw = os.getenv("PIPER_SPEAKER_ID", "").strip()
        piper_speaker_id = int(speaker_raw) if speaker_raw else None

        config = cls(
            project_root=PROJECT_ROOT,
            image_directory=_resolve_path("IMAGE_DIRECTORY", "assets/backgrounds"),
            music_directory=_resolve_path("MUSIC_DIRECTORY", "assets/music"),
            font_directory=_resolve_path("FONT_DIRECTORY", "assets/fonts"),
            license_directory=_resolve_path("LICENSE_DIRECTORY", "assets/licenses"),
            output_directory=_resolve_path("OUTPUT_DIRECTORY", "output"),
            history_file=_resolve_path("HISTORY_FILE", "data/history.json"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).strip(),
            openrouter_model=os.getenv(
                "OPENROUTER_MODEL", "openrouter/free"
            ).strip()
            or "openrouter/free",
            openrouter_fallback_model=os.getenv(
                "OPENROUTER_FALLBACK_MODEL",
                "google/gemma-2-9b-it:free,meta-llama/llama-3.1-8b-instruct:free,mistralai/mistral-7b-instruct:free,nvidia/nemotron-3-ultra-550b-a55b:free",
            ).strip()
            or "google/gemma-2-9b-it:free,meta-llama/llama-3.1-8b-instruct:free,mistralai/mistral-7b-instruct:free,nvidia/nemotron-3-ultra-550b-a55b:free",
            openrouter_timeout_seconds=_parse_int(
                os.getenv("OPENROUTER_TIMEOUT_SECONDS"), 60
            ),
            openrouter_max_retries=_parse_int(os.getenv("OPENROUTER_MAX_RETRIES"), 3),
            story_language=os.getenv("STORY_LANGUAGE", "English").strip() or "English",
            story_min_words=_parse_int(os.getenv("STORY_MIN_WORDS"), 150),
            story_max_words=_parse_int(os.getenv("STORY_MAX_WORDS"), 190),
            story_min_scenes=_parse_int(os.getenv("STORY_MIN_SCENES"), 5),
            story_max_scenes=_parse_int(os.getenv("STORY_MAX_SCENES"), 8),
            story_similarity_threshold=_parse_float(
                os.getenv("STORY_SIMILARITY_THRESHOLD"), 0.65
            ),
            pixabay_api_key=os.getenv("PIXABAY_API_KEY", "").strip(),
            openverse_enabled=_parse_bool(os.getenv("OPENVERSE_ENABLED"), False),
            openverse_client_id=os.getenv("OPENVERSE_CLIENT_ID", "").strip(),
            openverse_client_secret=os.getenv("OPENVERSE_CLIENT_SECRET", "").strip(),
            image_download_timeout_seconds=_parse_int(
                os.getenv("IMAGE_DOWNLOAD_TIMEOUT_SECONDS"), 20
            ),
            image_max_bytes=_parse_int(os.getenv("IMAGE_MAX_BYTES"), 15 * 1024 * 1024),
            piper_executable=os.getenv("PIPER_EXECUTABLE", "piper").strip() or "piper",
            piper_voice=os.getenv("PIPER_VOICE", "en_US-ljspeech-medium").strip(),
            piper_model_path=_resolve_optional_path("PIPER_MODEL_PATH"),
            piper_config_path=_resolve_optional_path("PIPER_CONFIG_PATH"),
            piper_speaker_id=piper_speaker_id,
            piper_length_scale=_parse_float(os.getenv("PIPER_LENGTH_SCALE"), 1.05),
            piper_volume=_parse_float(os.getenv("PIPER_VOLUME"), 1.0),
            piper_sentence_silence=_parse_float(os.getenv("PIPER_SENTENCE_SILENCE"), 0.55),
            allow_Offline_tts_fallback=_parse_bool(
                os.getenv("ALLOW_OFFLINE_TTS_FALLBACK"), False
            ),
            allow_narration_only_fallback=_parse_bool(
                os.getenv("ALLOW_NARRATION_ONLY_FALLBACK"), True
            ),
            music_volume=_parse_float(os.getenv("MUSIC_VOLUME"), 0.28),
            music_fade_in_seconds=_parse_float(os.getenv("MUSIC_FADE_IN_SECONDS"), 1.5),
            music_fade_out_seconds=_parse_float(os.getenv("MUSIC_FADE_OUT_SECONDS"), 2.5),
            enable_audio_ducking=_parse_bool(os.getenv("ENABLE_AUDIO_DUCKING"), True),
            internet_archive_music_enabled=_parse_bool(
                os.getenv("INTERNET_ARCHIVE_MUSIC_ENABLED"), True
            ),
            freesound_api_key=os.getenv("FREESOUND_API_KEY", "").strip(),
            video_width=_parse_int(os.getenv("VIDEO_WIDTH"), 1080),
            video_height=_parse_int(os.getenv("VIDEO_HEIGHT"), 1920),
            video_fps=_parse_int(os.getenv("VIDEO_FPS"), 30),
            crossfade_duration_seconds=_parse_float(
                os.getenv("CROSSFADE_DURATION_SECONDS"), 0.5
            ),
            title_overlay_seconds=_parse_float(os.getenv("TITLE_OVERLAY_SECONDS"), 3.5),
            dark_overlay_opacity=_parse_float(os.getenv("DARK_OVERLAY_OPACITY"), 0.28),
            subtitle_font_name=os.getenv("SUBTITLE_FONT_NAME", "Arial").strip() or "Arial",
            subtitle_font_size=_parse_int(os.getenv("SUBTITLE_FONT_SIZE"), 16),
            subtitle_margin_v=_parse_int(os.getenv("SUBTITLE_MARGIN_V"), 65),
            ffmpeg_binary=os.getenv("FFMPEG_BINARY", "ffmpeg").strip() or "ffmpeg",
            ffprobe_binary=os.getenv("FFPROBE_BINARY", "ffprobe").strip() or "ffprobe",
            ffmpeg_preset=os.getenv("FFMPEG_PRESET", "medium").strip() or "medium",
            ffmpeg_crf=_parse_int(os.getenv("FFMPEG_CRF"), 21),
            keep_temp_files=_parse_bool(os.getenv("KEEP_TEMP_FILES"), False),
            dry_run=_parse_bool(os.getenv("DRY_RUN"), True),
            enable_youtube=_parse_bool(os.getenv("ENABLE_YOUTUBE"), False),
            enable_instagram=_parse_bool(os.getenv("ENABLE_INSTAGRAM"), False),
            enable_facebook=_parse_bool(os.getenv("ENABLE_FACEBOOK"), False),
            enable_tiktok=_parse_bool(os.getenv("ENABLE_TIKTOK"), False),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
            youtube_client_id=os.getenv("YOUTUBE_CLIENT_ID", "").strip(),
            youtube_client_secret=os.getenv("YOUTUBE_CLIENT_SECRET", "").strip(),
            youtube_refresh_token=os.getenv("YOUTUBE_REFRESH_TOKEN", "").strip(),
            youtube_privacy_status=os.getenv("YOUTUBE_PRIVACY_STATUS", "public").strip() or "public",
            youtube_category_id=os.getenv("YOUTUBE_CATEGORY_ID", "24").strip(),
            instagram_account_id=os.getenv("INSTAGRAM_ACCOUNT_ID", "").strip(),
            instagram_access_token=os.getenv("INSTAGRAM_ACCESS_TOKEN", "").strip(),
            instagram_video_public_url_base=os.getenv(
                "INSTAGRAM_VIDEO_PUBLIC_URL_BASE", ""
            ).strip(),
            instagram_graph_version=os.getenv("INSTAGRAM_GRAPH_VERSION", "v22.0").strip(),
            facebook_page_id=os.getenv("FACEBOOK_PAGE_ID", "").strip(),
            facebook_page_access_token=os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN", "").strip(),
            facebook_graph_version=os.getenv("FACEBOOK_GRAPH_VERSION", "v22.0").strip(),
            tiktok_access_token=os.getenv("TIKTOK_ACCESS_TOKEN", "").strip(),
            tiktok_privacy_level=os.getenv("TIKTOK_PRIVACY_LEVEL", "SELF_ONLY").strip(),
        )
        return config

    def get_sensitive_values(self) -> List[str]:
        """Return all configured secret values for log redaction."""
        candidates = [
            self.openrouter_api_key,
            self.pixabay_api_key,
            self.openverse_client_secret,
            self.youtube_client_secret,
            self.youtube_refresh_token,
            self.instagram_access_token,
            self.facebook_page_access_token,
            self.tiktok_access_token,
        ]
        return [val for val in candidates if val and len(val) >= 4]

    def validate_core_settings(self, require_api_key: bool = True) -> None:
        """Validate core pipeline settings and enforce strict zero-budget free-model rules."""
        if self.story_min_words <= 10:
            raise ConfigurationError("STORY_MIN_WORDS must be greater than 10.")
        if self.story_max_words < self.story_min_words:
            raise ConfigurationError(
                f"STORY_MAX_WORDS ({self.story_max_words}) must be >= STORY_MIN_WORDS ({self.story_min_words})."
            )
        if not (0.0 <= self.music_volume <= 1.0):
            raise ConfigurationError("MUSIC_VOLUME must be between 0.0 and 1.0.")
        if self.video_width <= 0 or self.video_height <= 0 or self.video_fps <= 0:
            raise ConfigurationError("Video dimensions and FPS must be positive integers.")

        # Enforce zero-cost policy: OpenRouter models must be explicit :free / free routes
        if not is_free_model_identifier(self.openrouter_model):
            raise ConfigurationError(
                f"Zero-budget safeguard violation: OPENROUTER_MODEL ({self.openrouter_model!r}) "
                "must use an explicit ':free' endpoint suffix. Paid models are strictly prohibited."
            )
        if self.openrouter_fallback_model:
            for fallback in self.openrouter_fallback_model.split(","):
                fallback = fallback.strip()
                if fallback and not is_free_model_identifier(fallback):
                    raise ConfigurationError(
                        f"Zero-budget safeguard violation: OPENROUTER_FALLBACK_MODEL "
                        f"({fallback!r}) must use an explicit ':free' endpoint suffix."
                    )

        if require_api_key and not self.openrouter_api_key:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required for AI story generation. "
                "Set it in your .env file or GitHub Actions Secrets."
            )

    def validate_publisher_requirements(self) -> Dict[str, List[str]]:
        """Check which enabled publishers have missing credentials.

        Returns a dictionary mapping platform name to a list of missing variable names.
        """
        missing: Dict[str, List[str]] = {}
        if self.enable_youtube:
            req = []
            if not self.youtube_client_id:
                req.append("YOUTUBE_CLIENT_ID")
            if not self.youtube_client_secret:
                req.append("YOUTUBE_CLIENT_SECRET")
            if not self.youtube_refresh_token:
                req.append("YOUTUBE_REFRESH_TOKEN")
            if req:
                missing["youtube"] = req

        if self.enable_instagram:
            req = []
            if not self.instagram_account_id:
                req.append("INSTAGRAM_ACCOUNT_ID")
            if not self.instagram_access_token:
                req.append("INSTAGRAM_ACCESS_TOKEN")
            if req:
                missing["instagram"] = req

        if self.enable_facebook:
            req = []
            if not self.facebook_page_id:
                req.append("FACEBOOK_PAGE_ID")
            if not self.facebook_page_access_token:
                req.append("FACEBOOK_PAGE_ACCESS_TOKEN")
            if req:
                missing["facebook"] = req

        if self.enable_tiktok:
            req = []
            if not self.tiktok_access_token:
                req.append("TIKTOK_ACCESS_TOKEN")
            if req:
                missing["tiktok"] = req

        return missing


def setup_logging(
    config: Optional[AppConfig] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure root and application loggers with secret redaction."""
    level_name = config.log_level if config else "INFO"
    level = getattr(logging, level_name.upper(), logging.INFO)
    secrets = config.get_sensitive_values() if config else []
    redactor = SecretRedactingFilter(secrets=secrets)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicate lines across test runs
    root_logger.handlers.clear()
    root_logger.filters.clear()
    root_logger.addFilter(redactor)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(redactor)
    root_logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redactor)
        root_logger.addHandler(file_handler)

    # Silence overly verbose third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "googleapiclient", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("auto_story_video")
