"""Platform-specific social media metadata and attribution generator.

Produces tailored titles, descriptions, captions, tags, and mandatory creative
commons / provider license attributions for YouTube Shorts, Instagram Reels,
Facebook Reels, and TikTok.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.image_selector import ImageMetadata
from src.music_selector import MusicTrackMetadata
from src.story_generator import StoryPayload

logger = logging.getLogger(__name__)


@dataclass
class YouTubeMetadata:
    title: str
    description: str
    tags: List[str]
    category_id: str = "24"
    privacy_status: str = "private"


@dataclass
class InstagramMetadata:
    caption: str


@dataclass
class FacebookMetadata:
    title: str
    description: str


@dataclass
class TikTokMetadata:
    title: str
    privacy_level: str = "SELF_ONLY"


@dataclass
class PlatformMetadataBundle:
    production_date: str
    run_id: str
    story_title: str
    language: str
    mood: str
    youtube: YouTubeMetadata
    instagram: InstagramMetadata
    facebook: FacebookMetadata
    tiktok: TikTokMetadata
    attributions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_hashtags(hashtags: List[str], required_tags: Optional[List[str]] = None) -> List[str]:
    seen = set()
    result: List[str] = []
    combined = list(hashtags) + (required_tags or ["#shortstory", "#emotionalstory", "#storytime"])
    for raw in combined:
        clean = raw.strip()
        if not clean:
            continue
        if not clean.startswith("#"):
            clean = f"#{clean}"
        lower = clean.lower()
        if lower not in seen:
            seen.add(lower)
            result.append(clean)
    return result


def build_attribution_lines(
    images: List[ImageMetadata],
    music_track: Optional[MusicTrackMetadata] = None,
    voice_license: str = "",
) -> List[str]:
    """Build deduplicated license attribution lines for all visual and audio assets used."""
    lines: List[str] = []
    seen = set()
    for img in images:
        attr = img.attribution_text.strip()
        if attr and attr not in seen:
            seen.add(attr)
            lines.append(f"Visual: {attr}")
    if music_track is not None:
        m_attr = music_track.attribution_instructions.strip()
        if m_attr and m_attr not in seen:
            seen.add(m_attr)
            lines.append(f"Music: {m_attr} ({music_track.license_name})")
    if voice_license:
        lines.append(f"Narration TTS: Piper TTS ({voice_license})")
    return lines


class MetadataGenerator:
    """Generates platform-compliant titles, captions, tags, and license credits."""

    def __init__(
        self,
        youtube_category_id: str = "24",
        youtube_privacy_status: str = "private",
        tiktok_privacy_level: str = "SELF_ONLY",
    ) -> None:
        self.youtube_category_id = youtube_category_id
        self.youtube_privacy_status = youtube_privacy_status
        self.tiktok_privacy_level = tiktok_privacy_level

    def generate(
        self,
        story: StoryPayload,
        production_date: str,
        run_id: str,
        images: List[ImageMetadata],
        music_track: Optional[MusicTrackMetadata] = None,
        voice_license: str = "",
    ) -> PlatformMetadataBundle:
        """Create metadata tailored to YouTube Shorts, Instagram Reels, Facebook Reels, and TikTok."""
        hashtags = _normalize_hashtags(story.hashtags)
        attributions = build_attribution_lines(
            images=images, music_track=music_track, voice_license=voice_license
        )
        credits_block = "\n".join(attributions)

        # 1. YouTube Shorts (Title <= 100 chars including #Shorts)
        yt_suffix = " #Shorts"
        max_base_title = 100 - len(yt_suffix)
        base_title = story.title.strip()[:max_base_title].rstrip()
        yt_title = f"{base_title}{yt_suffix}"

        yt_hashtags_str = " ".join((hashtags + ["#Shorts"])[:12])
        yt_desc_parts = [
            f"{story.title} — {story.caption.strip()}",
            "",
            yt_hashtags_str,
        ]
        if credits_block:
            yt_desc_parts.extend(["", "Credits & Licenses:", credits_block])
        yt_description = "\n".join(yt_desc_parts)[:4900]
        yt_tags = [t.lstrip("#") for t in (hashtags + ["#Shorts", "#moralstory"])][:20]

        # 2. Instagram Reels (Caption <= 2200 chars, max 30 hashtags)
        ig_tags = " ".join((hashtags + ["#reels", "#emotionalreels"])[:15])
        ig_parts = [
            f"{story.title}",
            "",
            story.caption.strip(),
            "",
            ig_tags,
        ]
        if credits_block:
            ig_parts.extend(["", "Credits:", credits_block])
        ig_caption = "\n".join(ig_parts)[:2150]

        # 3. Facebook Reels
        fb_tags = " ".join((hashtags + ["#FacebookReels", "#ShortStory"])[:12])
        fb_parts = [
            f"{story.title} - {story.caption.strip()}",
            "",
            fb_tags,
        ]
        if credits_block:
            fb_parts.extend(["", "Credits:", credits_block])
        fb_description = "\n".join(fb_parts)[:2150]

        # 4. TikTok (Direct Post title/caption)
        tt_tags = " ".join((hashtags + ["#storytime", "#emotional"])[:8])
        tt_title = f"{story.title} — {story.caption.strip()} {tt_tags}"[:2100]

        return PlatformMetadataBundle(
            production_date=production_date,
            run_id=run_id,
            story_title=story.title,
            language=story.language,
            mood=story.mood,
            youtube=YouTubeMetadata(
                title=yt_title,
                description=yt_description,
                tags=yt_tags,
                category_id=self.youtube_category_id,
                privacy_status=self.youtube_privacy_status,
            ),
            instagram=InstagramMetadata(caption=ig_caption),
            facebook=FacebookMetadata(
                title=story.title[:100],
                description=fb_description,
            ),
            tiktok=TikTokMetadata(
                title=tt_title,
                privacy_level=self.tiktok_privacy_level,
            ),
            attributions=attributions,
        )

    def save_metadata(self, bundle: PlatformMetadataBundle, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(bundle.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved platform metadata bundle to %s", output_path)
        return output_path
