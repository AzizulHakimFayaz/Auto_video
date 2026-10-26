"""Emotional background music selection and license validation module.

Selects mood-matched instrumental tracks from the local `assets/music` library
ONLY when a verified license record exists in `assets/licenses/music_manifest.json`
or a sidecar `.json` file. Supports graceful fallback to narration-only audio
when no licensed music track is available.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import AppConfig

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

STORY_MOOD_CATEGORIES: Dict[str, List[str]] = {
    "sad": ["sad", "sadness", "melancholy", "melancholic", "lonely", "grief", "piano"],
    "hopeful": ["hopeful", "hope", "uplifting", "inspiring", "warm", "redemption"],
    "romantic": ["romantic", "love", "tender", "warm", "affection", "heartfelt"],
    "suspenseful": ["suspenseful", "suspense", "tense", "tragedy", "atmospheric", "mystery"],
    "peaceful": ["peaceful", "peace", "calm", "ambient", "gentle", "serene", "nature"],
}


class MusicSelectionError(RuntimeError):
    """Raised when music selection fails and narration-only fallback is disabled."""


@dataclass
class MusicTrackMetadata:
    """Verified licensing and acoustic metadata for a local soundtrack file."""

    track_id: str
    file_path: str
    title: str
    creator: str
    source_url: str
    license_name: str
    license_url: str
    attribution_instructions: str
    commercial_use_allowed: bool
    mood_tags: List[str] = field(default_factory=list)
    restrictions: str = "None"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def classify_story_mood(mood_description: str) -> str:
    """Map a story's free-form mood string (e.g., 'sad then hopeful') to a canonical music mood."""
    lower = mood_description.lower()
    # If the story transitions to hopeful ("sad then hopeful"), prioritize hopeful or sad
    tokens = re.findall(r"\b[a-z]+\b", lower)
    for category, keywords in STORY_MOOD_CATEGORIES.items():
        if any(tok in keywords for tok in tokens):
            return category
    for category, keywords in STORY_MOOD_CATEGORIES.items():
        if any(kw in lower for kw in keywords):
            return category
    return "peaceful"


class MusicSelector:
    """Selects and validates licensed background music matched to the story mood."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        manifest_path = self.config.license_directory / "music_manifest.json"
        if not manifest_path.exists():
            return {}
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "tracks" in data and isinstance(data["tracks"], dict):
                return data["tracks"]
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not parse music license manifest %s: %s", manifest_path, exc)
        return {}

    def _validate_track_license(
        self, audio_file: Path, manifest: Dict[str, Dict[str, Any]]
    ) -> Optional[MusicTrackMetadata]:
        """Verify that an audio file exists, is non-empty, and has explicit license metadata.

        Never assumes an unspecified soundtrack is copyright-free.
        """
        if not audio_file.exists() or not audio_file.is_file():
            return None
        try:
            if audio_file.stat().st_size < 256:
                logger.warning("Skipping empty/corrupt audio file: %s", audio_file)
                return None
        except OSError:
            return None

        meta = manifest.get(audio_file.name)
        if not meta:
            sidecar = audio_file.with_suffix(audio_file.suffix + ".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    meta = None

        if not isinstance(meta, dict):
            logger.warning(
                "Skipping music file '%s': no verified license entry found in music_manifest.json or sidecar. "
                "Unspecified tracks are never assumed to be copyright-free.",
                audio_file.name,
            )
            return None

        required = {"title", "creator", "source_url", "license_name", "attribution_instructions"}
        missing = [k for k in required if not str(meta.get(k, "")).strip()]
        if missing:
            logger.warning(
                "Skipping music file '%s': incomplete license record (missing %s).",
                audio_file.name,
                missing,
            )
            return None

        mood_tags = [str(t).lower().strip() for t in meta.get("mood_tags", []) if t]
        return MusicTrackMetadata(
            track_id=f"music:{audio_file.name}",
            file_path=str(audio_file.resolve()),
            title=str(meta["title"]).strip(),
            creator=str(meta["creator"]).strip(),
            source_url=str(meta["source_url"]).strip(),
            license_name=str(meta["license_name"]).strip(),
            license_url=str(meta.get("license_url", "")).strip(),
            attribution_instructions=str(meta["attribution_instructions"]).strip(),
            commercial_use_allowed=bool(meta.get("commercial_use_allowed", True)),
            mood_tags=mood_tags,
            restrictions=str(meta.get("restrictions", "None")).strip(),
        )

    def select_music(self, story_mood: str) -> Optional[MusicTrackMetadata]:
        """Select the best licensed background track matching `story_mood`.

        Returns None when no licensed track is available and narration-only fallback is enabled.
        """
        target_category = classify_story_mood(story_mood)
        raw_mood_tokens = set(re.findall(r"\b[a-z]+\b", story_mood.lower()))
        target_keywords = set(STORY_MOOD_CATEGORIES.get(target_category, [])) | raw_mood_tokens

        manifest = self._load_manifest()
        candidates: List[MusicTrackMetadata] = []

        if self.config.music_directory.exists() and self.config.music_directory.is_dir():
            for file_path in sorted(self.config.music_directory.iterdir()):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
                ):
                    validated = self._validate_track_license(file_path, manifest)
                    if validated is not None:
                        candidates.append(validated)

        if not candidates:
            if self.config.allow_narration_only_fallback:
                logger.warning(
                    "No licensed background music tracks found in %s. "
                    "Falling back to narration-only audio per configuration.",
                    self.config.music_directory,
                )
                return None
            raise MusicSelectionError(
                f"No licensed background music tracks available in {self.config.music_directory} "
                "and ALLOW_NARRATION_ONLY_FALLBACK is disabled."
            )

        best_track: Optional[MusicTrackMetadata] = None
        best_score = -1.0

        for track in candidates:
            tag_set = set(track.mood_tags)
            score = 0.0
            if target_category in tag_set:
                score += 5.0
            overlap = len(tag_set & target_keywords)
            score += overlap * 2.0
            if track.commercial_use_allowed:
                score += 1.0

            if score > best_score:
                best_score = score
                best_track = track

        if best_track is not None:
            logger.info(
                "Selected background music '%s' by %s (license=%s, target_mood=%s).",
                best_track.title,
                best_track.creator,
                best_track.license_name,
                target_category,
            )
        return best_track
