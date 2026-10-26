"""Unit tests for the scene-aware image selection system and provider fallbacks."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, Set

import pytest
import requests
from PIL import Image

from src.config import AppConfig
from src.image_selector import ImageSelectionError, ImageSelector, validate_image_file
from src.story_generator import SceneSegment


def _create_valid_jpeg(path: Path, width: int = 1080, height: int = 1920, color: tuple = (30, 60, 90)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=color)
    img.save(path, format="JPEG")
    return path


def _create_jpeg_bytes(width: int = 1080, height: int = 1920) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(40, 80, 120))
    img.save(buf, format="JPEG")
    return buf.getvalue()


class FakeHTTPResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_data: Any = None,
        content: bytes = b"",
        headers: Dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self._content = content
        self.headers = headers or {}

    def json(self) -> Any:
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data

    def iter_content(self, chunk_size: int = 65536):
        yield self._content


class FakeHTTPSession:
    def __init__(self, handler):
        self.handler = handler

    def get(self, url: str, **kwargs: Any) -> FakeHTTPResponse:
        return self.handler(url, **kwargs)


def test_local_tagged_image_selected_first(tmp_path: Path) -> None:
    bg_dir = tmp_path / "backgrounds"
    lic_dir = tmp_path / "licenses"
    _create_valid_jpeg(bg_dir / "sadness" / "rainy_village_house.jpg")

    cfg = AppConfig(
        image_directory=bg_dir,
        license_directory=lic_dir,
        output_directory=tmp_path / "output",
    )
    selector = ImageSelector(config=cfg)
    scene = SceneSegment(
        narration="An old house stood in the rain.",
        image_query="old house in a rainy village",
        visual_mood="sadness",
        duration_hint_seconds=7.0,
    )
    meta = selector.select_image_for_scene(scene, scene_index=0, excluded_ids=set())
    assert meta.provider == "local"
    assert "rainy_village_house.jpg" in meta.local_path
    assert meta.is_portrait


def test_remote_provider_failure_falls_back_to_local_library(tmp_path: Path) -> None:
    bg_dir = tmp_path / "backgrounds"
    lic_dir = tmp_path / "licenses"
    _create_valid_jpeg(bg_dir / "hope" / "generic_sunrise.jpg")

    cfg = AppConfig(
        image_directory=bg_dir,
        license_directory=lic_dir,
        output_directory=tmp_path / "output",
        pixabay_api_key="test_pixabay_key",
        openverse_enabled=True,
    )

    def failing_handler(url: str, **kwargs: Any) -> FakeHTTPResponse:
        return FakeHTTPResponse(status_code=429, json_data={"error": "rate limit"})

    selector = ImageSelector(config=cfg, session=FakeHTTPSession(failing_handler))
    scene = SceneSegment(
        narration="She looked toward tomorrow.",
        image_query="astronaut telescope nebula",  # No local keyword match -> triggers remote -> falls back to local
        visual_mood="hope",
        duration_hint_seconds=6.0,
    )
    meta = selector.select_image_for_scene(scene, scene_index=0, excluded_ids=set())
    assert meta.provider == "local_fallback"
    assert "generic_sunrise.jpg" in meta.local_path


def test_openverse_provider_records_license_and_attribution(tmp_path: Path) -> None:
    bg_dir = tmp_path / "backgrounds"
    lic_dir = tmp_path / "licenses"
    cfg = AppConfig(
        image_directory=bg_dir,
        license_directory=lic_dir,
        output_directory=tmp_path / "output",
        openverse_enabled=True,
    )
    jpeg_bytes = _create_jpeg_bytes()

    def openverse_handler(url: str, **kwargs: Any) -> FakeHTTPResponse:
        if "api.openverse.org" in url:
            return FakeHTTPResponse(
                status_code=200,
                json_data={
                    "results": [
                        {
                            "id": "ov-123",
                            "title": "Misty Lantern",
                            "creator": "Jane Photographer",
                            "license": "by",
                            "license_version": "4.0",
                            "license_url": "https://creativecommons.org/licenses/by/4.0/",
                            "foreign_landing_url": "https://example.org/photo/123",
                            "url": "https://images.example.org/lantern.jpg",
                            "tags": [{"name": "lantern"}, {"name": "mist"}],
                        }
                    ]
                },
            )
        return FakeHTTPResponse(
            status_code=200,
            content=jpeg_bytes,
            headers={"Content-Length": str(len(jpeg_bytes))},
        )

    selector = ImageSelector(config=cfg, session=FakeHTTPSession(openverse_handler))
    scene = SceneSegment(
        narration="A lantern glowed in the mist.",
        image_query="misty lantern night",
        visual_mood="peace",
        duration_hint_seconds=6.0,
    )
    results = selector.select_images_for_story([scene])
    assert len(results) == 1
    meta = results[0]
    assert meta.provider == "openverse"
    assert meta.creator == "Jane Photographer"
    assert "CC BY 4.0" in meta.license_name
    assert (lic_dir / "selected_images_attribution.json").exists()


def test_corrupt_image_and_empty_library_raises_clear_error(tmp_path: Path) -> None:
    bg_dir = tmp_path / "backgrounds" / "sadness"
    bg_dir.mkdir(parents=True)
    corrupt_file = bg_dir / "corrupt.jpg"
    corrupt_file.write_bytes(b"not-a-valid-jpeg-file-at-all" * 20)

    with pytest.raises(ImageSelectionError, match="Corrupt or unreadable"):
        validate_image_file(corrupt_file)

    cfg = AppConfig(
        image_directory=tmp_path / "backgrounds",
        license_directory=tmp_path / "licenses",
        output_directory=tmp_path / "output",
    )
    selector = ImageSelector(config=cfg)
    scene = SceneSegment(
        narration="Empty scene.",
        image_query="rainy street",
        visual_mood="sadness",
    )
    with pytest.raises(ImageSelectionError, match="No usable image found"):
        selector.select_image_for_scene(scene, scene_index=0, excluded_ids=set())
