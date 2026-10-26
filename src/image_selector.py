"""Scene-aware image selection system with local library, Pixabay, and Openverse providers.

Implements the strict 5-stage selection hierarchy:
1. Search existing local library for an image tagged with a matching mood and subject.
2. If no good match exists and a valid Pixabay API key is configured, query Pixabay API.
3. If Pixabay is unavailable, optionally query Openverse API with strict license checks.
4. If no remote provider is available, select the best available local fallback image.
5. If no usable image exists, stop rendering with a clear ImageSelectionError.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

from src.config import AppConfig
from src.story_generator import SceneSegment

logger = logging.getLogger(__name__)

THEME_FOLDERS: List[str] = [
    "sadness",
    "love",
    "family",
    "hope",
    "tragedy",
    "peace",
    "nature",
    "city",
]

MOOD_TO_THEME_MAP: Dict[str, str] = {
    "sad": "sadness",
    "sadness": "sadness",
    "melancholic": "sadness",
    "melancholy": "sadness",
    "lonely": "sadness",
    "loneliness": "sadness",
    "grief": "sadness",
    "tearful": "sadness",
    "love": "love",
    "romantic": "love",
    "kindness": "love",
    "warm": "love",
    "warmth": "love",
    "compassion": "love",
    "generosity": "love",
    "family": "family",
    "reunion": "family",
    "nostalgic": "family",
    "home": "family",
    "friendship": "family",
    "hope": "hope",
    "hopeful": "hope",
    "uplifting": "hope",
    "inspiring": "hope",
    "sunrise": "hope",
    "redemption": "hope",
    "tragedy": "tragedy",
    "tragic": "tragedy",
    "loss": "tragedy",
    "storm": "tragedy",
    "tense": "tragedy",
    "suspenseful": "tragedy",
    "crisis": "tragedy",
    "peace": "peace",
    "peaceful": "peace",
    "calm": "peace",
    "serene": "peace",
    "gentle": "peace",
    "quiet": "peace",
    "nature": "nature",
    "forest": "nature",
    "rain": "nature",
    "ocean": "nature",
    "city": "city",
    "urban": "city",
    "street": "city",
    "station": "city",
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

ALLOWED_OPENVERSE_LICENSES = {"cc0", "pdm", "by", "by-sa"}


class ImageSelectionError(RuntimeError):
    """Raised when no valid, licensed image can be found for a scene."""


@dataclass
class ImageMetadata:
    """Comprehensive provenance and licensing metadata for a selected scene image."""

    id: str
    local_path: str
    provider: str  # "local", "pixabay", "openverse", "local_fallback"
    creator: str
    source_url: str
    license_name: str
    license_url: str
    attribution_text: str
    retrieved_at: str
    width: int
    height: int
    mood: str
    tags: List[str] = field(default_factory=list)
    scene_index: int = 0
    query_used: str = ""

    @property
    def is_portrait(self) -> bool:
        return self.height >= self.width

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_mood_to_theme(mood_str: str, query_str: str = "") -> str:
    """Map an arbitrary visual mood or keyword string to one of the 8 canonical theme folders."""
    tokens = re.findall(r"\b[a-z]+\b", f"{mood_str} {query_str}".lower())
    for token in tokens:
        if token in MOOD_TO_THEME_MAP:
            return MOOD_TO_THEME_MAP[token]
    for key, folder in MOOD_TO_THEME_MAP.items():
        if key in mood_str.lower():
            return folder
    return "peace"


def extract_keywords(text: str) -> Set[str]:
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "under",
        "very",
        "scene",
        "shot",
        "view",
    }
    words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    return words - stop_words


def validate_image_file(path: Path) -> Tuple[int, int]:
    """Verify that a file exists, is non-empty, and can be decoded by Pillow without corruption.

    Returns (width, height). Raises ImageSelectionError if invalid or corrupt.
    """
    if not path.exists() or not path.is_file():
        raise ImageSelectionError(f"Image file does not exist: {path}")
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ImageSelectionError(f"Cannot stat image file {path}: {exc}") from exc
    if file_size < 128:
        raise ImageSelectionError(f"Image file is empty or truncated ({file_size} bytes): {path}")

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img2:
            img2.load()
            width, height = img2.size
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageSelectionError(f"Corrupt or unreadable image file {path}: {exc}") from exc

    if width < 64 or height < 64:
        raise ImageSelectionError(
            f"Image dimensions ({width}x{height}) are too small for video rendering: {path}"
        )
    return width, height


def validate_http_url(url: str) -> bool:
    """Ensure a URL uses HTTP/HTTPS and has a valid hostname."""
    if not url or not isinstance(url, str):
        return False
    parsed = urlparse(url.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class ImageProvider(abc.ABC):
    """Abstract interface for scene image providers."""

    @abc.abstractmethod
    def search_image(
        self,
        query: str,
        mood: str,
        excluded_ids: Set[str],
        scene_index: int = 0,
    ) -> Optional[ImageMetadata]:
        """Search and return a validated local ImageMetadata or None if no match."""


class LocalLibraryImageProvider(ImageProvider):
    """Searches `assets/backgrounds/<theme>/` for tagged local images."""

    def __init__(self, image_directory: Path, license_directory: Path) -> None:
        self.image_directory = image_directory
        self.license_directory = license_directory

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        manifest_path = self.license_directory / "images_manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _collect_candidates(self, theme_folder: Optional[str] = None) -> List[Tuple[Path, str]]:
        candidates: List[Tuple[Path, str]] = []
        if not self.image_directory.exists():
            return candidates

        folders = [theme_folder] if theme_folder else THEME_FOLDERS
        for folder in folders:
            if not folder:
                continue
            folder_path = self.image_directory / folder
            if not folder_path.exists() or not folder_path.is_dir():
                continue
            for file_path in sorted(folder_path.iterdir()):
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                ):
                    candidates.append((file_path, folder))
        return candidates

    def _build_metadata_for_local_file(
        self,
        file_path: Path,
        folder: str,
        manifest: Dict[str, Dict[str, Any]],
        provider_label: str,
        scene_index: int,
        query: str,
    ) -> Optional[ImageMetadata]:
        try:
            width, height = validate_image_file(file_path)
        except ImageSelectionError as exc:
            logger.warning("Skipping invalid local image %s: %s", file_path, exc)
            return None

        rel_key = f"{folder}/{file_path.name}"
        sidecar_path = file_path.with_suffix(file_path.suffix + ".json")
        meta_info: Dict[str, Any] = manifest.get(rel_key, {})
        if not meta_info and sidecar_path.exists():
            try:
                meta_info = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta_info = {}

        stem_tokens = re.findall(r"[a-z0-9]+", file_path.stem.lower())
        tags = list(dict.fromkeys([folder] + stem_tokens + meta_info.get("tags", [])))
        creator = str(meta_info.get("creator", "Local Project Asset Library"))
        source_url = str(meta_info.get("source_url", f"local://{rel_key}"))
        license_name = str(meta_info.get("license", "CC0 1.0 Universal (Public Domain)"))
        license_url = str(
            meta_info.get(
                "license_url", "https://creativecommons.org/publicdomain/zero/1.0/"
            )
        )
        attribution = str(
            meta_info.get(
                "attribution",
                f"'{file_path.stem}' by {creator} ({license_name})",
            )
        )

        return ImageMetadata(
            id=f"local:{rel_key}",
            local_path=str(file_path.resolve()),
            provider=provider_label,
            creator=creator,
            source_url=source_url,
            license_name=license_name,
            license_url=license_url,
            attribution_text=attribution,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            width=width,
            height=height,
            mood=folder,
            tags=tags,
            scene_index=scene_index,
            query_used=query,
        )

    def search_image(
        self,
        query: str,
        mood: str,
        excluded_ids: Set[str],
        scene_index: int = 0,
    ) -> Optional[ImageMetadata]:
        """Stage 1: Search local library for an image matching BOTH mood/theme AND query keywords."""
        target_theme = normalize_mood_to_theme(mood, query)
        query_keywords = extract_keywords(query)
        manifest = self._load_manifest()

        # Check target theme folder first, then all folders for strong keyword matches
        candidates = self._collect_candidates(target_theme) + [
            c for c in self._collect_candidates(None) if c[1] != target_theme
        ]

        best_meta: Optional[ImageMetadata] = None
        best_score: float = 0.0

        for file_path, folder in candidates:
            meta = self._build_metadata_for_local_file(
                file_path, folder, manifest, "local", scene_index, query
            )
            if meta is None:
                continue

            tag_set = set(t.lower() for t in meta.tags)
            keyword_overlap = len(query_keywords & tag_set)
            if keyword_overlap == 0:
                continue  # Stage 1 requires an actual subject/tag match

            score = keyword_overlap * 3.0
            if folder == target_theme:
                score += 2.5
            if meta.is_portrait:
                score += 1.0
            if meta.width >= 1080 and meta.height >= 1920:
                score += 0.5
            if meta.id in excluded_ids or meta.local_path in excluded_ids:
                score -= 5.0

            if score > best_score:
                best_score = score
                best_meta = meta

        return best_meta

    def select_fallback_image(
        self,
        query: str,
        mood: str,
        excluded_ids: Set[str],
        scene_index: int = 0,
    ) -> Optional[ImageMetadata]:
        """Stage 4: Select the best available local fallback image by mood or general library."""
        target_theme = normalize_mood_to_theme(mood, query)
        manifest = self._load_manifest()

        ordered_candidates = self._collect_candidates(target_theme) + [
            c for c in self._collect_candidates(None) if c[1] != target_theme
        ]

        best_meta: Optional[ImageMetadata] = None
        best_score: float = -999.0

        for file_path, folder in ordered_candidates:
            meta = self._build_metadata_for_local_file(
                file_path, folder, manifest, "local_fallback", scene_index, query
            )
            if meta is None:
                continue

            score = 0.0
            if folder == target_theme:
                score += 4.0
            if meta.is_portrait:
                score += 1.5
            if meta.width >= 1080 and meta.height >= 1920:
                score += 0.5
            if meta.id not in excluded_ids and meta.local_path not in excluded_ids:
                score += 10.0

            if score > best_score:
                best_score = score
                best_meta = meta

        return best_meta


def download_and_verify_remote_image(
    url: str,
    cache_dir: Path,
    prefix: str,
    timeout: int = 20,
    max_bytes: int = 15 * 1024 * 1024,
    session: Optional[requests.Session] = None,
) -> Tuple[Path, int, int]:
    """Safely download a remote image with redirect, timeout, size, and Pillow verification."""
    if not validate_http_url(url):
        raise ImageSelectionError(f"Invalid or unsupported image URL: {url!r}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    ext = Path(urlparse(url).path).suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        ext = ".jpg"

    dest_path = cache_dir / f"{prefix}_{url_hash}{ext}"
    if dest_path.exists():
        try:
            w, h = validate_image_file(dest_path)
            return dest_path, w, h
        except ImageSelectionError:
            dest_path.unlink(missing_ok=True)

    http = session or requests
    headers = {"User-Agent": "AutoStoryVideoBot/1.0 (https://github.com/auto-story-video)"}
    try:
        resp = http.get(
            url, headers=headers, stream=True, timeout=timeout, allow_redirects=True
        )
    except requests.RequestException as exc:
        raise ImageSelectionError(f"Failed to download image from {url}: {exc}") from exc

    if resp.status_code == 429:
        raise ImageSelectionError("Remote image download rate-limited (HTTP 429).")
    if resp.status_code != 200:
        raise ImageSelectionError(
            f"Remote image download returned HTTP {resp.status_code} for {url}"
        )

    content_length = resp.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                raise ImageSelectionError(
                    f"Remote image exceeds maximum allowed size ({content_length} > {max_bytes} bytes)."
                )
        except ValueError:
            pass

    temp_path = dest_path.with_suffix(".part")
    total_read = 0
    try:
        with open(temp_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total_read += len(chunk)
                if total_read > max_bytes:
                    raise ImageSelectionError(
                        f"Remote image stream exceeded max size limit ({max_bytes} bytes)."
                    )
                fh.write(chunk)
        temp_path.replace(dest_path)
        width, height = validate_image_file(dest_path)
        return dest_path, width, height
    except Exception:
        temp_path.unlink(missing_ok=True)
        dest_path.unlink(missing_ok=True)
        raise


class PixabayImageProvider(ImageProvider):
    """Queries the official Pixabay API when PIXABAY_API_KEY is configured."""

    def __init__(
        self,
        config: AppConfig,
        cache_dir: Path,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.session = session

    def search_image(
        self,
        query: str,
        mood: str,
        excluded_ids: Set[str],
        scene_index: int = 0,
    ) -> Optional[ImageMetadata]:
        if not self.config.pixabay_api_key:
            return None

        http = self.session or requests
        clean_query = " ".join(re.findall(r"\b[a-zA-Z0-9]+\b", query)[:6]) or mood
        params = {
            "key": self.config.pixabay_api_key,
            "q": clean_query[:100],
            "image_type": "photo",
            "orientation": "vertical",
            "safesearch": "true",
            "per_page": 10,
        }

        try:
            resp = http.get(
                self.config.pixabay_base_url,
                params=params,
                timeout=self.config.image_download_timeout_seconds,
            )
        except requests.RequestException as exc:
            logger.warning("Pixabay API request failed: %s", type(exc).__name__)
            return None

        if resp.status_code == 429:
            logger.warning("Pixabay API rate limit reached (HTTP 429).")
            return None
        if resp.status_code != 200:
            logger.warning("Pixabay API returned HTTP %d.", resp.status_code)
            return None

        try:
            payload = resp.json()
        except ValueError:
            logger.warning("Pixabay API returned invalid JSON.")
            return None

        hits = payload.get("hits", [])
        if not isinstance(hits, list) or not hits:
            return None

        for hit in hits:
            if not isinstance(hit, dict):
                continue
            hit_id = f"pixabay:{hit.get('id', '')}"
            img_url = hit.get("largeImageURL") or hit.get("webformatURL") or ""
            if not img_url or hit_id in excluded_ids or img_url in excluded_ids:
                continue

            try:
                local_path, width, height = download_and_verify_remote_image(
                    url=img_url,
                    cache_dir=self.cache_dir,
                    prefix=f"pixabay_{hit.get('id', 'img')}",
                    timeout=self.config.image_download_timeout_seconds,
                    max_bytes=self.config.image_max_bytes,
                    session=self.session,
                )
            except ImageSelectionError as exc:
                logger.warning("Skipping Pixabay candidate %s: %s", hit_id, exc)
                continue

            creator = str(hit.get("user") or "Pixabay Contributor")
            page_url = str(hit.get("pageURL") or "https://pixabay.com")
            tags_raw = str(hit.get("tags") or "")
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

            return ImageMetadata(
                id=hit_id,
                local_path=str(local_path.resolve()),
                provider="pixabay",
                creator=creator,
                source_url=page_url,
                license_name="Pixabay Content License",
                license_url="https://pixabay.com/service/license-summary/",
                attribution_text=f"Image by {creator} from Pixabay ({page_url})",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                width=width,
                height=height,
                mood=normalize_mood_to_theme(mood, query),
                tags=tags,
                scene_index=scene_index,
                query_used=clean_query,
            )

        return None


class OpenverseImageProvider(ImageProvider):
    """Queries the Openverse API v1 images endpoint with strict CC license verification."""

    def __init__(
        self,
        config: AppConfig,
        cache_dir: Path,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.session = session

    def search_image(
        self,
        query: str,
        mood: str,
        excluded_ids: Set[str],
        scene_index: int = 0,
    ) -> Optional[ImageMetadata]:
        if not self.config.openverse_enabled:
            return None

        http = self.session or requests
        clean_query = " ".join(re.findall(r"\b[a-zA-Z0-9]+\b", query)[:5]) or mood
        params = {
            "q": clean_query,
            "license": "cc0,pdm,by,by-sa",
            "aspect_ratio": "tall",
            "page_size": 10,
        }
        headers = {"User-Agent": "AutoStoryVideoBot/1.0"}

        try:
            resp = http.get(
                self.config.openverse_base_url,
                params=params,
                headers=headers,
                timeout=self.config.image_download_timeout_seconds,
            )
        except requests.RequestException as exc:
            logger.warning("Openverse API request failed: %s", type(exc).__name__)
            return None

        if resp.status_code == 429:
            logger.warning("Openverse API rate limit reached (HTTP 429).")
            return None
        if resp.status_code != 200:
            logger.warning("Openverse API returned HTTP %d.", resp.status_code)
            return None

        try:
            payload = resp.json()
        except ValueError:
            logger.warning("Openverse API returned invalid JSON.")
            return None

        results = payload.get("results", [])
        if not isinstance(results, list) or not results:
            return None

        for item in results:
            if not isinstance(item, dict):
                continue
            lic = str(item.get("license") or "").lower().strip()
            if lic not in ALLOWED_OPENVERSE_LICENSES:
                continue

            item_id = f"openverse:{item.get('id', '')}"
            img_url = str(item.get("url") or "")
            if not img_url or item_id in excluded_ids or img_url in excluded_ids:
                continue

            try:
                local_path, width, height = download_and_verify_remote_image(
                    url=img_url,
                    cache_dir=self.cache_dir,
                    prefix=f"openverse_{item.get('id', 'img')}",
                    timeout=self.config.image_download_timeout_seconds,
                    max_bytes=self.config.image_max_bytes,
                    session=self.session,
                )
            except ImageSelectionError as exc:
                logger.warning("Skipping Openverse candidate %s: %s", item_id, exc)
                continue

            creator = str(item.get("creator") or "Unknown Creator")
            title = str(item.get("title") or "Untitled Image")
            lic_version = str(item.get("license_version") or "4.0")
            lic_url = str(
                item.get("license_url")
                or f"https://creativecommons.org/licenses/{lic}/{lic_version}/"
            )
            landing_url = str(item.get("foreign_landing_url") or img_url)
            lic_label = f"CC {lic.upper()} {lic_version}" if lic not in {"cc0", "pdm"} else lic.upper()
            attribution = (
                item.get("attribution")
                or f'"{title}" by {creator} is licensed under {lic_label} ({lic_url})'
            )

            tags_list: List[str] = []
            for t_obj in item.get("tags", []) or []:
                if isinstance(t_obj, dict) and t_obj.get("name"):
                    tags_list.append(str(t_obj["name"]))

            return ImageMetadata(
                id=item_id,
                local_path=str(local_path.resolve()),
                provider="openverse",
                creator=creator,
                source_url=landing_url,
                license_name=lic_label,
                license_url=lic_url,
                attribution_text=str(attribution),
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                width=width,
                height=height,
                mood=normalize_mood_to_theme(mood, query),
                tags=tags_list,
                scene_index=scene_index,
                query_used=clean_query,
            )

        return None


class ImageSelector:
    """Orchestrates scene image selection across local, Pixabay, Openverse, and fallback stages."""

    def __init__(
        self,
        config: AppConfig,
        session: Optional[requests.Session] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir or (config.output_directory / "cache" / "images")
        self.local_provider = LocalLibraryImageProvider(
            image_directory=config.image_directory,
            license_directory=config.license_directory,
        )
        self.pixabay_provider = PixabayImageProvider(
            config=config, cache_dir=self.cache_dir, session=session
        )
        self.openverse_provider = OpenverseImageProvider(
            config=config, cache_dir=self.cache_dir, session=session
        )

    def select_image_for_scene(
        self,
        scene: SceneSegment,
        scene_index: int,
        excluded_ids: Set[str],
    ) -> ImageMetadata:
        """Select a single image for a scene following the strict 5-step priority order."""
        query = scene.image_query
        mood = scene.visual_mood

        # 1. Local library match by mood and subject keywords
        selected = self.local_provider.search_image(
            query=query, mood=mood, excluded_ids=excluded_ids, scene_index=scene_index
        )
        if selected is not None:
            logger.info(
                "Scene %d: Selected tagged local image %s (mood=%s)",
                scene_index + 1,
                Path(selected.local_path).name,
                selected.mood,
            )
            return selected

        # 2. Pixabay API search when key is configured
        if self.config.pixabay_api_key:
            selected = self.pixabay_provider.search_image(
                query=query, mood=mood, excluded_ids=excluded_ids, scene_index=scene_index
            )
            if selected is not None:
                logger.info(
                    "Scene %d: Selected Pixabay image %s by %s",
                    scene_index + 1,
                    selected.id,
                    selected.creator,
                )
                return selected

        # 3. Openverse API search when enabled
        if self.config.openverse_enabled:
            selected = self.openverse_provider.search_image(
                query=query, mood=mood, excluded_ids=excluded_ids, scene_index=scene_index
            )
            if selected is not None:
                logger.info(
                    "Scene %d: Selected Openverse image %s (%s)",
                    scene_index + 1,
                    selected.id,
                    selected.license_name,
                )
                return selected

        # 4. Local mood/library fallback
        selected = self.local_provider.select_fallback_image(
            query=query, mood=mood, excluded_ids=excluded_ids, scene_index=scene_index
        )
        if selected is not None:
            logger.info(
                "Scene %d: Selected local fallback image %s (theme=%s)",
                scene_index + 1,
                Path(selected.local_path).name,
                selected.mood,
            )
            return selected

        # 5. Stop rendering with a clear error rather than generating a broken video
        raise ImageSelectionError(
            f"Scene {scene_index + 1}: No usable image found in local library ({self.config.image_directory}) "
            f"or configured remote providers for query={query!r}, mood={mood!r}."
        )

    def select_images_for_story(
        self,
        scenes: List[SceneSegment],
        recently_used_ids: Optional[Set[str]] = None,
    ) -> List[ImageMetadata]:
        """Select a distinct, licensed image for every scene in the story."""
        if not scenes:
            raise ImageSelectionError("Cannot select images for an empty scene list.")

        excluded: Set[str] = set(recently_used_ids or set())
        selected_list: List[ImageMetadata] = []

        for idx, scene in enumerate(scenes):
            meta = self.select_image_for_scene(
                scene=scene, scene_index=idx, excluded_ids=excluded
            )
            excluded.add(meta.id)
            excluded.add(meta.local_path)
            if meta.source_url:
                excluded.add(meta.source_url)
            selected_list.append(meta)

        self._record_license_log(selected_list)
        return selected_list

    def _record_license_log(self, images: List[ImageMetadata]) -> None:
        """Persist attribution and license provenance for all selected images."""
        self.config.license_directory.mkdir(parents=True, exist_ok=True)
        log_path = self.config.license_directory / "selected_images_attribution.json"
        existing: List[Dict[str, Any]] = []
        if log_path.exists():
            try:
                data = json.loads(log_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    existing = data
            except (json.JSONDecodeError, OSError):
                existing = []

        existing.extend([img.to_dict() for img in images])
        # Keep most recent 200 entries
        log_path.write_text(
            json.dumps(existing[-200:], indent=2, ensure_ascii=False), encoding="utf-8"
        )
